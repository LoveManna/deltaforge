"""CPU tests for the tiled GEMV: layout, launch geometry, and the kernel's indexing.

Triton is absent locally, so the numerics of the compiled kernel are the GPU gate's job.
What *is* decidable here is the arithmetic the kernel is a transcription of — the split-K
partial boundaries and the row-0 extraction from the MMA accumulator — and that is where
this kernel is most likely to be wrong. `_emulate` follows the kernel body statement for
statement, and a mutation test proves it can fail.
"""

from __future__ import annotations

import pytest
import torch

from .tiled_gemv import _launch_shape, to_k_major

# -- layout ---------------------------------------------------------------------------


def test_k_major_storage_is_the_transpose_and_is_contiguous():
    weight = torch.randn(2560, 9216)  # [N, K] as nn.Linear stores it

    k_major = to_k_major(weight)

    assert k_major.shape == (9216, 2560)
    assert k_major.is_contiguous()
    assert torch.equal(k_major, weight.t().contiguous())


def test_k_major_does_not_alias_the_parameter_it_was_built_from():
    """It is registered as a buffer beside the shared bf16 Parameter, not instead of it."""
    weight = torch.randn(8, 16)

    k_major = to_k_major(weight)
    k_major[0, 0] += 1.0

    assert not torch.equal(k_major.t(), weight)


# -- launch geometry ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "k", "note"),
    [
        (9216, 2560, "gate_proj / up_proj"),
        (2560, 9216, "down_proj"),
        (32, 2560, "in_proj_a / in_proj_b"),
        (248320, 2560, "the tied lm head"),
    ],
)
def test_every_projection_in_the_model_gets_a_tile_tl_dot_accepts(n, k, note):
    """`tl.dot` needs both tile dims >= 16, so the tile cannot be narrowed for parallelism.

    Batch 003's kernel used 8 rows per program on the narrow projections. That is not
    available here, which is exactly why split-K exists in this kernel and not in that one.
    """
    block_n, block_k, split_k, _warps = _launch_shape(n, k)

    assert block_n >= 16, note
    assert block_k >= 16, note
    assert split_k >= 1


def test_a_32_channel_projection_is_split_because_no_tile_can_fill_the_card():
    """`in_proj_a` is 32 output channels wide: one program at any legal BLOCK_N."""
    _block_n, _block_k, split_k, _warps = _launch_shape(32, 2560)

    assert split_k == 8, "the only parallelism available here is over K"


def test_a_head_wide_enough_to_fill_the_card_is_not_split():
    """248320 / 64 is 3880 programs. Splitting K would buy nothing and cost a reduction."""
    _block_n, _block_k, split_k, _warps = _launch_shape(248320, 2560)

    assert split_k == 1


def test_k_is_never_split_further_than_it_has_blocks():
    """A split with no work still costs a partial-buffer row and a pass over it."""
    _block_n, block_k, split_k, _warps = _launch_shape(32, 128)

    assert split_k <= 128 // block_k


# -- the kernel's indexing, emulated ---------------------------------------------------


def _emulate(x, w_k_major, block_n, block_k, split_k):
    """`_tiled_gemv_bf16_kernel`, statement for statement, in torch on a CPU.

    The two things most likely to be wrong in this kernel are the split-K chunk boundaries
    and the extraction of row 0 from the MMA accumulator, and both are pure indexing.
    """
    k, n = w_k_major.shape
    rows = x.shape[0]
    partials = torch.zeros(split_k, rows, n, dtype=torch.float32)
    chunk = -(-k // split_k)
    for pid_k in range(split_k):
        lo, hi = pid_k * chunk, min((pid_k + 1) * chunk, k)
        for pid_m in range(rows):
            for n0 in range(0, n, block_n):
                cols = torch.arange(n0, min(n0 + block_n, n))
                acc = torch.zeros(16, len(cols), dtype=torch.float32)
                for k0 in range(lo, hi, block_k):
                    ks = torch.arange(k0, min(k0 + block_k, hi))
                    # x in row 0, zeros elsewhere: the MMA carries the partial sums and the
                    # only cross-lane reduction in this kernel is the tl.sum after the loop.
                    x16 = torch.zeros(16, len(ks), dtype=torch.float32)
                    x16[0] = x[pid_m, ks].float()
                    acc += x16 @ w_k_major[ks][:, cols].float()  # tl.dot accumulate
                partials[pid_k, pid_m, cols] = acc.sum(dim=0)  # one reduction, row 0
    return partials.sum(dim=0)


@pytest.mark.parametrize(("n", "k", "split_k"), [(64, 256, 1), (64, 256, 4), (48, 192, 3)])
def test_the_split_k_emulation_reproduces_the_dense_product(n, k, split_k):
    torch.manual_seed(0)
    weight = torch.randn(n, k)
    x = torch.randn(1, k)

    emulated = _emulate(x, to_k_major(weight), block_n=32, block_k=64, split_k=split_k)
    dense = torch.nn.functional.linear(x.float(), weight.float())

    assert torch.allclose(emulated, dense, rtol=1e-5, atol=1e-4)


def test_the_emulation_handles_a_k_that_no_split_divides_evenly():
    """200 over 3 splits is 67, 67, 66. A chunk that assumed even division drops rows."""
    torch.manual_seed(0)
    weight, x = torch.randn(32, 200), torch.randn(1, 200)

    emulated = _emulate(x, to_k_major(weight), block_n=32, block_k=64, split_k=3)

    assert torch.allclose(
        emulated, torch.nn.functional.linear(x.float(), weight.float()), rtol=1e-5, atol=1e-4
    )


def test_more_than_one_row_of_x_is_handled_independently():
    """The grid's third axis is M. A kernel that ignored it would return row 0 everywhere."""
    torch.manual_seed(0)
    weight, x = torch.randn(48, 128), torch.randn(4, 128)

    emulated = _emulate(x, to_k_major(weight), block_n=32, block_k=64, split_k=2)

    assert torch.allclose(
        emulated, torch.nn.functional.linear(x.float(), weight.float()), rtol=1e-5, atol=1e-4
    )


def test_a_split_k_that_drops_the_tail_chunk_is_caught():
    """The emulation only proves something if it can fail. This is the mutation: a `hi` that
    uses the chunk size rather than clamping to K silently loses the last partial rows."""
    torch.manual_seed(0)
    weight, x = torch.randn(32, 200), torch.randn(1, 200)
    w = to_k_major(weight)
    chunk = -(-200 // 3)
    wrong = sum(
        (x[:, p * chunk : (p + 1) * chunk].float() @ w[p * chunk : (p + 1) * chunk].float())
        for p in range(2)  # the bug: two chunks instead of three
    )

    assert not torch.allclose(
        wrong, torch.nn.functional.linear(x.float(), weight.float()), rtol=1e-5, atol=1e-4
    )


def test_taking_the_whole_accumulator_instead_of_row_zero_is_caught():
    """Rows 1-15 of the MMA accumulator are zero by construction. If the zero padding were
    ever dropped -- broadcasting x into all 16 rows, say -- the answer would be 16x."""
    torch.manual_seed(0)
    weight, x = torch.randn(32, 128), torch.randn(1, 128)
    w = to_k_major(weight)
    broadcast = torch.ones(16, 128) * x[0].float()
    wrong = (broadcast @ w.float()).sum(dim=0)

    assert not torch.allclose(
        wrong, torch.nn.functional.linear(x.float(), weight.float())[0], rtol=1e-5, atol=1e-4
    )


# -- installation ---------------------------------------------------------------------


@pytest.fixture
def model():
    from ..config import tiny_config
    from ..reference import ReferenceModel

    return ReferenceModel(tiny_config())


def test_installing_changes_every_projection_in_every_decoder_layer(model):
    """An installer that patches nothing produces a candidate identical to the reference,
    measures 1.00, and is recorded as a well-behaved null. That costs a rental to find."""
    from torch import nn

    from .tiled_gemv import TiledBf16Linear, install_tiled_bf16

    before = [type(m).__name__ for m in model.layers.modules() if isinstance(m, nn.Linear)]
    install_tiled_bf16(model)
    after = [type(m).__name__ for m in model.layers.modules() if isinstance(m, nn.Linear)]

    assert before and set(before) == {"Linear"}
    assert set(after) == {TiledBf16Linear.__name__}


def test_the_k_major_copy_is_a_buffer_so_the_bf16_parameter_stays_shared(model):
    """`cli._assert_parameters_are_shared` walks `named_parameters` and rejects anything
    with no counterpart in the reference. A K-major *parameter* would fail the run."""
    from .tiled_gemv import install_tiled_bf16

    names_before = set(dict(model.named_parameters()))
    install_tiled_bf16(model)

    assert set(dict(model.named_parameters())) == names_before
    assert any(name.endswith("w_k_major") for name, _ in model.named_buffers())


def test_installing_twice_is_a_no_op(model):
    """`apply_champions` may run an installer more than once; a second K-major copy of
    every projection is 8.4 GB this card does not have."""
    from .tiled_gemv import install_tiled_bf16

    install_tiled_bf16(model)
    buffers = {name: t.data_ptr() for name, t in model.named_buffers() if name.endswith("w_k_major")}
    install_tiled_bf16(model)

    assert {name: t.data_ptr() for name, t in model.named_buffers() if name.endswith("w_k_major")} == buffers


def test_the_k_major_buffer_holds_the_transpose_of_the_weight_it_replaces(model):
    """The layout change is the kernel's whole premise; a transpose dropped here would
    produce a kernel that is wrong everywhere and correct in shape."""
    from torch import nn

    from .tiled_gemv import install_tiled_bf16

    install_tiled_bf16(model)

    for module in model.layers.modules():
        if isinstance(module, nn.Linear):
            assert torch.equal(module.w_k_major, module.weight.t().contiguous())


def test_the_kernel_and_its_checks_are_registered_under_the_kernels_own_name(model):
    """`quantised_linear` already exports `bf16_correctness_checks` for a different kernel.
    An op-keyed table would run 009's checks against 015's weights and report `pass`."""
    from . import CHECK_BUILDERS, REGISTRY
    from .tiled_gemv import tiled_bf16_correctness_checks

    assert REGISTRY.get("tiled_gemv_bf16").replaces == "decode_step"
    assert CHECK_BUILDERS["tiled_gemv_bf16"] is tiled_bf16_correctness_checks
    assert CHECK_BUILDERS["gemv_bf16"] is not tiled_bf16_correctness_checks


def test_an_installer_is_registered_for_the_kernel(model):
    """`apply_champions` fails loudly rather than silently running the reference."""
    from ..model import INSTALLERS

    assert "tiled_gemv_bf16" in INSTALLERS
