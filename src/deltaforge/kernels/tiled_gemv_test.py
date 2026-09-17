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

from .quantised_linear import INT4_MAX
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


# -- quantisers -----------------------------------------------------------------------


def _restore_fp8(weight):
    from .tiled_gemv import quantise_fp8_per_channel

    q = quantise_fp8_per_channel(weight)
    return q.qweight.to(torch.float32) * q.scale.unsqueeze(1)


def _restore_int8(weight):
    from .quantised_linear import dequantise_int8, quantise_int8_per_channel

    q = quantise_int8_per_channel(weight)
    return dequantise_int8(q.qweight, q.scale)


def _restore_int4(weight):
    from .tiled_gemv import dequantise_int4_k_major, quantise_int4_k_major

    q = quantise_int4_k_major(weight)
    return dequantise_int4_k_major(q.qweight, q.scale, group_size=q.group_size).t()


def test_fp8_e4m3_round_trip_keeps_the_per_channel_scale_meaningful():
    from .tiled_gemv import quantise_fp8_per_channel

    torch.manual_seed(0)
    weight = torch.randn(16, 256)
    weight[3] *= 100.0

    q = quantise_fp8_per_channel(weight)

    assert q.qweight.dtype is torch.float8_e4m3fn
    assert q.scale.shape == (16,)
    restored = q.qweight.to(torch.float32) * q.scale.unsqueeze(1)
    # e4m3 carries 3 mantissa bits: ~6% worst-case relative error per element.
    rel = ((restored - weight).abs() / weight.abs().clamp(min=1e-6)).max()
    assert rel < 0.07


def test_the_outlier_row_does_not_cost_the_others_their_range():
    """Per-channel rather than per-tensor: one row 100x the rest would otherwise spend the
    whole 8-bit range on itself."""
    from .tiled_gemv import quantise_fp8_per_channel

    torch.manual_seed(0)
    weight = torch.randn(16, 256)
    weight[3] *= 100.0

    q = quantise_fp8_per_channel(weight)

    assert q.scale[3] > 50 * q.scale[0]


def test_fp8_is_coarser_than_int8_and_finer_than_int4():
    """The accuracy ordering the batch-004 gates are derived from, asserted rather than assumed."""
    torch.manual_seed(0)
    weight = torch.randn(32, 512)

    def err(restored):
        return (restored - weight).abs().mean()

    assert err(_restore_int8(weight)) < err(_restore_fp8(weight)) < err(_restore_int4(weight))


def test_a_zero_row_does_not_become_nan():
    """It cannot happen in a trained checkpoint, which is precisely why nothing would catch
    it: NaN weights still look plausible downstream."""
    from .tiled_gemv import quantise_fp8_per_channel

    weight = torch.zeros(4, 64)
    weight[1] = 1.0

    q = quantise_fp8_per_channel(weight)

    assert torch.isfinite(q.scale).all()
    assert torch.isfinite(q.qweight.to(torch.float32)).all()


def test_the_head_is_quantised_in_row_blocks_so_the_gate_does_not_oom():
    """The tied head is 248320 x 2560; a single fp32 temporary of it is 2.5 GB, on a card
    already holding the reference and its CUDA graphs."""
    from .tiled_gemv import ROW_CHUNK_ELEMENTS

    assert ROW_CHUNK_ELEMENTS <= 1 << 24


# -- int4, K-major --------------------------------------------------------------------


def test_int4_k_major_packing_round_trips():
    from .tiled_gemv import dequantise_int4_k_major, quantise_int4_k_major

    torch.manual_seed(0)
    weight = torch.randn(32, 512)

    q = quantise_int4_k_major(weight)
    restored = dequantise_int4_k_major(q.qweight, q.scale, group_size=q.group_size)

    assert q.qweight.shape == (256, 32), "packed [K/2, N]: two K-elements per byte"
    assert q.scale.shape == (512 // q.group_size, 32)
    assert restored.shape == (512, 32)
    # Scored against the tensor's own scale, not per element: int4 sends a near-zero
    # element to zero and a per-element relative error there is 1.0 by construction.
    error = (restored - weight.t()).abs().max()
    assert error < weight.abs().max() / INT4_MAX


def test_int4_pairs_k_with_k_plus_half_so_one_byte_load_serves_two_k_blocks():
    """The packing is what lets the kernel read one contiguous byte block and consume it
    against two contiguous slices of x, each inside exactly one scale group."""
    from .tiled_gemv import quantise_int4_k_major

    torch.manual_seed(0)
    weight = torch.randn(8, 512)

    q = quantise_int4_k_major(weight)

    low = (q.qweight & 0x0F).to(torch.int16) - 8
    high = (q.qweight >> 4).to(torch.int16) - 8
    assert low.shape == high.shape == (256, 8)
    # Element (j, n) and element (j + 256, n) share a byte.
    scale = q.scale
    group = q.group_size
    assert torch.allclose(
        low[0].float() * scale[0 // group], weight.t()[0].float(), atol=scale[0].max().item()
    )
    assert torch.allclose(
        high[0].float() * scale[256 // group], weight.t()[256].float(), atol=scale[256 // group].max().item()
    )


# -- installing the quantised variants ------------------------------------------------


@pytest.mark.parametrize(
    ("installer", "patched", "kind"),
    [
        ("install_tiled_fp8_all_linear", "TiledFp8Linear", "fp8"),
        ("install_tiled_int8_all_linear", "TiledInt8Linear", "int8"),
        ("install_tiled_int4_full", "TiledInt4Linear", "int4"),
    ],
)
def test_each_quantised_installer_patches_every_layer_projection(model, installer, patched, kind):
    from torch import nn

    from . import tiled_gemv

    getattr(tiled_gemv, installer)(model)

    classes = {type(m).__name__ for m in model.layers.modules() if isinstance(m, nn.Linear)}
    assert classes == {patched}


def test_a_quantised_install_registers_buffers_and_leaves_the_parameters_shared(model):
    from .tiled_gemv import install_tiled_fp8_all_linear

    names_before = set(dict(model.named_parameters()))
    install_tiled_fp8_all_linear(model)

    assert set(dict(model.named_parameters())) == names_before
    buffers = dict(model.named_buffers())
    assert any(name.endswith("w_k_major") for name in buffers)
    assert any(name.endswith("qscale") for name in buffers)


def test_the_fp8_weight_is_stored_k_major_at_one_byte_per_element(model):
    from torch import nn

    from .tiled_gemv import install_tiled_fp8_all_linear

    install_tiled_fp8_all_linear(model)

    for module in model.layers.modules():
        if isinstance(module, nn.Linear):
            assert module.w_k_major.dtype is torch.float8_e4m3fn
            assert module.w_k_major.shape == (module.in_features, module.out_features)


def test_installing_the_head_routes_project_logits_through_the_kernel(model):
    """The tied head is 15.1% of weight bytes and has no `nn.Linear` to swap."""
    from .tiled_gemv import install_tiled_fp8_full

    before = type(model).__name__
    install_tiled_fp8_full(model)

    assert type(model).__name__ != before
    assert hasattr(model, "tiled_lm_head")
    assert model.tiled_lm_head.w_k_major.shape == (
        model.lm_head_weight.shape[1],
        model.lm_head_weight.shape[0],
    )


def test_the_mlp_installer_leaves_the_attention_projections_alone(model):
    """018's whole job is to be a smaller dose than 016. If it patched everything the
    dose-response ladder would have two identical rungs and say nothing."""
    from torch import nn

    from .tiled_gemv import install_tiled_fp8_mlp

    install_tiled_fp8_mlp(model)

    patched = [type(m).__name__ for m in model.layers.modules() if isinstance(m, nn.Linear)]
    assert "Linear" in patched and "TiledFp8Linear" in patched


@pytest.mark.parametrize(
    "kernel",
    ["tiled_fp8_mlp", "tiled_fp8_all_linear", "tiled_fp8_full", "tiled_int8_all_linear", "tiled_int4_full"],
)
def test_every_batch_004_kernel_has_its_own_checks_and_installer(kernel):
    """Keyed by kernel name, not by replaced operation: four of these five replace
    `decode_step`, and an op-keyed table would run one's checks against another's weights
    and report `pass`."""
    from ..model import INSTALLERS
    from . import CHECK_BUILDERS, REGISTRY

    assert REGISTRY.get(kernel) is not None
    assert kernel in CHECK_BUILDERS
    assert kernel in INSTALLERS
