"""Hypothesis 001's kernel: structure on CPU, numerics on a GPU.

The CPU half runs everywhere, including CI with no Triton installed, and covers the
things that are cheap to get wrong and expensive to discover on a rented box: that
installing patches one model and not another, that there is no silent PyTorch fallback,
and that the launch geometry is right.

The GPU half is marked `gpu` and skips on a real condition. It compares against the
reference modules themselves rather than a re-derived formula, because a re-derivation
can agree with the kernel while both disagree with `reference.py`.
"""

from __future__ import annotations

import pytest
import torch

from ..config import tiny_config
from ..model import INSTALLERS, apply_champions
from ..reference import ReferenceModel
from . import REGISTRY, KernelStatus
from .fused_rmsnorm_residual import (
    HAS_TRITON,
    _launch_config,
    _next_power_of_two,
    add_rms_norm,
    install,
    is_installed,
    rms_norm,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")


@pytest.fixture
def model():
    return ReferenceModel(tiny_config())


# -- registration ---------------------------------------------------------------------


def test_the_kernel_is_registered_retired_and_still_installable():
    """Retired, not unregistered — and its installer stays wired.

    The kernel was graveyarded on its mechanism (0.018% of per-token bytes, below the
    harness's noise band) without ever being measured. Keeping it registered with a
    working installer is what stops a later session rediscovering the dead end, and it
    means the entry can be promoted for a one-off comparison without rebuilding anything.
    """
    entry = REGISTRY.get("fused_rmsnorm_residual")

    assert entry.replaces == "rms_norm_residual"
    assert entry.status is KernelStatus.RETIRED
    assert entry.hypothesis == "001-fused-rmsnorm-residual"
    assert "fused_rmsnorm_residual" in INSTALLERS


def test_no_champion_ships_so_the_candidate_column_is_the_identity():
    """The shipped registry installs nothing, on purpose.

    With every entry retired, ``apply_champions`` is a no-op and the ``candidate`` column
    is bit-identical to ``eager``. That is the *identity champion* a first GPU session uses
    to calibrate the harness: every column must return 1.00 +/- noise, at zero
    kernel-writing risk. See AGENT.md, "If this is the first session that ever gets a GPU".
    """
    assert REGISTRY.champions() == {}


# -- installation ---------------------------------------------------------------------


def test_install_patches_every_decoder_layer_and_the_final_norm(model):
    assert not is_installed(model)

    install(model)

    assert is_installed(model)
    assert all(type(layer).__name__ == "FusedDecoderLayer" for layer in model.layers)
    assert type(model.norm).__name__ == "FusedRMSNorm"


def test_installing_does_not_touch_a_different_model_instance(model):
    """The eager and compiled columns build their own model. If installation leaked
    through the class object, every column would run the kernel and the benchmark would
    compare the candidate against itself."""
    untouched = ReferenceModel(tiny_config())

    install(model)

    assert not is_installed(untouched)
    assert all(type(layer).__name__ == "DecoderLayer" for layer in untouched.layers)
    assert type(untouched.norm).__name__ == "RMSNorm"


def test_install_is_idempotent(model):
    install(model)
    classes = [type(layer) for layer in model.layers]

    install(model)

    assert [type(layer) for layer in model.layers] == classes


def test_the_installed_model_keeps_the_reference_parameters(model):
    """A class swap and not a reconstruction: the weights must be the same objects, or
    the candidate would be running the right kernel on the wrong numbers."""
    before = {name: param.data_ptr() for name, param in model.named_parameters()}

    apply_champions(model, REGISTRY)

    assert {name: param.data_ptr() for name, param in model.named_parameters()} == before


# -- no silent fallback ---------------------------------------------------------------


@pytest.mark.parametrize("op", ["rms_norm", "add_rms_norm"])
def test_the_ops_refuse_to_run_on_cpu_rather_than_falling_back(op):
    """Falling back to PyTorch here would run the reference while the harness recorded
    the result as the candidate."""
    x = torch.zeros(2, 8)
    weight = torch.zeros(8)

    with pytest.raises(RuntimeError, match="requires a CUDA tensor"):
        if op == "rms_norm":
            rms_norm(x, weight, 1e-6)
        else:
            add_rms_norm(x, x, weight, 1e-6)


# -- input validation -----------------------------------------------------------------


def test_a_mismatched_residual_is_refused_rather_than_broadcast():
    """Broadcasting would launch cleanly and compute something else. The reference adds
    two equal-shaped tensors; anything else is a caller bug, not a shape to reinterpret."""
    x = torch.zeros(2, 8)
    weight = torch.zeros(8)

    with pytest.raises(ValueError, match="residual shape"):
        add_rms_norm(x, torch.zeros(1, 8), weight, 1e-6)

    with pytest.raises(ValueError, match="residual dtype"):
        add_rms_norm(x, torch.zeros(2, 8, dtype=torch.float64), weight, 1e-6)


def test_a_weight_that_is_not_a_contiguous_vector_is_refused():
    """The kernel indexes the weight by column with unit stride."""
    x = torch.zeros(2, 8)

    with pytest.raises(ValueError, match="contiguous 1-D"):
        rms_norm(x, torch.zeros(2, 8), 1e-6)

    with pytest.raises(ValueError, match="activation row is"):
        rms_norm(x, torch.zeros(9), 1e-6)


@requires_cuda
@requires_triton
@pytest.mark.gpu
def test_a_non_contiguous_activation_is_handled_not_refused():
    """The reference norm accepts any layout. Refusing one here would make the candidate
    fail where the baseline works, which is a harness bug wearing a kernel's clothes."""
    from ..reference import RMSNorm

    norm = RMSNorm(2560, eps=1e-6).cuda().to(torch.bfloat16)
    norm.weight.data.normal_(0.0, 0.1)
    wide = torch.randn(4, 2, 2560, device="cuda", dtype=torch.bfloat16)
    view = wide[:, 1]  # strided rows: stride(-2) is 2 * 2560, not 2560

    assert not view.is_contiguous()
    torch.testing.assert_close(
        rms_norm(view, norm.weight, norm.eps).float(), norm(view).float(), rtol=1e-2, atol=1e-2
    )


# -- launch geometry ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, 1), (2, 2), (3, 4), (128, 128), (2560, 4096), (4096, 4096), (4097, 8192)],
)
def test_next_power_of_two(n, expected):
    assert _next_power_of_two(n) == expected


def test_the_block_covers_the_whole_row_so_no_element_is_dropped():
    """One program per row, so BLOCK below the row width would silently normalise a
    prefix of the row and leave the rest unwritten."""
    for n in (128, 256, 2560, 9216):
        block, num_warps, _stages = _launch_config(n)
        assert block >= n
        assert block & (block - 1) == 0
        assert 4 <= num_warps <= 32


def test_a_row_too_wide_for_one_program_is_refused_rather_than_truncated():
    with pytest.raises(ValueError, match="too wide"):
        _launch_config(70000)


# -- numerics (GPU) -------------------------------------------------------------------


@requires_cuda
@requires_triton
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 1, 2560), (32, 1, 2560), (1, 512, 2560), (1, 2048, 2560)])
def test_rms_norm_matches_the_reference_module(shape):
    from ..reference import RMSNorm

    torch.manual_seed(0)
    norm = RMSNorm(shape[-1], eps=1e-6).cuda().to(torch.bfloat16)
    norm.weight.data.normal_(0.0, 0.1)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    expected = norm(x)
    actual = rms_norm(x, norm.weight, norm.eps)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)


@requires_cuda
@requires_triton
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 1, 2560), (32, 1, 2560), (1, 2048, 2560)])
def test_add_rms_norm_matches_add_then_reference_norm(shape):
    from ..reference import RMSNorm

    torch.manual_seed(0)
    norm = RMSNorm(shape[-1], eps=1e-6).cuda().to(torch.bfloat16)
    norm.weight.data.normal_(0.0, 0.1)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    expected_residual = residual + x
    expected_out = norm(expected_residual)
    actual_residual, actual_out = add_rms_norm(x, residual, norm.weight, norm.eps)

    # The residual stream is a plain add and must match bit for bit: it feeds every
    # later layer, so a rounding difference here compounds through the whole model.
    assert torch.equal(actual_residual, expected_residual)
    torch.testing.assert_close(actual_out.float(), expected_out.float(), rtol=1e-2, atol=1e-2)


@requires_cuda
@requires_triton
@pytest.mark.gpu
def test_the_scale_is_one_plus_weight_not_weight():
    """Qwen3.5 stores norm weights centred on zero. A kernel reading them as a plain
    scale produces an all-zero activation on a freshly initialised module and a subtly
    wrong one on real weights."""
    from ..reference import RMSNorm

    norm = RMSNorm(2560, eps=1e-6).cuda().to(torch.bfloat16)  # weight is all zeros
    x = torch.randn(1, 1, 2560, device="cuda", dtype=torch.bfloat16)

    out = rms_norm(x, norm.weight, norm.eps)

    assert out.abs().max() > 0, "zero weight must mean identity scale, not a zeroed output"
    torch.testing.assert_close(out.float(), norm(x).float(), rtol=1e-2, atol=1e-2)


@requires_cuda
@requires_triton
@pytest.mark.gpu
def test_the_installed_model_agrees_with_the_reference_end_to_end():
    """Random weights, tiny config, on the GPU: catches a wiring error in the patched
    layer — the wrong norm weight, the wrong residual, a dropped MLP — without needing
    the 9 GB checkpoint."""
    torch.manual_seed(0)
    config = tiny_config()
    reference = ReferenceModel(config).cuda().to(torch.bfloat16).eval()
    candidate = ReferenceModel(config).cuda().to(torch.bfloat16).eval()
    candidate.load_state_dict(reference.state_dict())
    install(candidate)

    ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")
    with torch.no_grad():
        expected, _ = reference(ids, reference.new_cache(1, 32))
        actual, _ = candidate(ids, candidate.new_cache(1, 32))

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
@requires_triton
@pytest.mark.gpu
def test_the_custom_ops_survive_torch_compile():
    """The scoring column compiles the candidate with max-autotune. If inductor cannot
    handle the custom op, that column silently graph-breaks and the comparison stops
    being like for like."""
    torch.manual_seed(0)
    config = tiny_config()
    candidate = ReferenceModel(config).cuda().to(torch.bfloat16).eval()
    install(candidate)
    ids = torch.randint(0, config.vocab_size, (1, 8), device="cuda")

    # `from torch import _dynamo` rather than `import torch._dynamo`: the latter binds a
    # local name `torch` in this function, which shadows the module-level import and makes
    # every earlier `torch.` reference in the body a use-before-assignment.
    from torch import _dynamo as dynamo

    dynamo.reset()
    explain = dynamo.explain(candidate)(ids, candidate.new_cache(1, 16))

    with torch.no_grad():
        eager_out, _ = candidate(ids, candidate.new_cache(1, 16))
        compiled = torch.compile(candidate, mode="max-autotune")
        compiled_out, _ = compiled(ids, candidate.new_cache(1, 16))

    # What this test is actually for, asserted directly instead of inferred from a number.
    # The docstring's concern is a silent graph break around the custom op, which would
    # make the scoring column stop being like-for-like. Dynamo will tell us that outright.
    assert explain.graph_break_count == 0, (
        f"the custom ops caused {explain.graph_break_count} graph break(s): {explain.break_reasons}"
    )

    # Numerics are checked at a bf16-appropriate scale rather than at 2e-2 absolute.
    # `max-autotune` selects different GEMM kernels than eager — different tile shapes,
    # different accumulation orders — so compiled and eager are not bit-identical by
    # construction, and on a tiny randomly-initialised model the logits are small and their
    # relative differences correspondingly large. Measured 2026-09-07 on an RTX 5090:
    # 0.156 absolute, 4.5% of elements outside 2e-2. Scored against the tensor's own scale,
    # which is what makes the bound mean the same thing on any config.
    diff = (compiled_out.float() - eager_out.float()).abs().max().item()
    scale = eager_out.float().abs().max().item()
    assert diff / max(scale, 1e-6) < 5e-2, (
        f"compiled and eager differ by {diff} against a max magnitude of {scale}"
    )
