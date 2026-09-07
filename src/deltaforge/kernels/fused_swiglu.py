"""Hypothesis 004 — fused SwiGLU activation.

**Mechanism.** `SwiGLUMLP.forward` is ``down_proj(silu(gate_proj(x)) * up_proj(x))``. The
two projections produce ``intermediate_size`` = 9216-wide intermediates; the reference
then makes a separate pass to apply SiLU and another to multiply. Fusing them reads the
two intermediates once and writes one result, instead of reading and writing them across
three elementwise kernels.

**What it does not touch.** The GEMMs. They are cuBLAS territory and 53.9% of the model's
weight bytes sit behind them, but streaming those weights is exactly what the reference
already does once. This kernel replaces only the epilogue.

**Ceiling: 0.026% of per-token bytes**, which is why `docs/HYPOTHESES.md` graveyarded it
without measuring. That was a judgement about *cost* — a whole rental to measure 0.026%.
At three minutes a slot the judgement changes, and a measured null is worth more than a
predicted one. Predicted `inconclusive`.

**Numerics.** SiLU is computed in fp32 and the product is rounded once, on store, matching
`F.silu(gate) * up` on bf16 tensors: `F.silu` on a bf16 input returns bf16, so the
reference rounds the SiLU result *before* multiplying. Computing the whole thing in fp32
would be more accurate than the reference and therefore a different function — and the
gate compares against the reference, not against infinite precision.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._triton import HAS_TRITON, launch_config, require_cuda, tl, triton

__all__ = ["correctness_checks", "install", "is_installed", "silu_mul"]


if HAS_TRITON:

    @triton.jit
    def _silu_mul_kernel(GATE, UP, OUT, n_elements, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        g = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(UP + offs, mask=mask, other=0.0)

        # silu(g) = g * sigmoid(g), in fp32, then rounded to the tensor dtype before the
        # multiply — see the module docstring on why this rounding is deliberate.
        activated = (g * tl.sigmoid(g)).to(UP.dtype.element_ty)
        tl.store(OUT + offs, activated * u, mask=mask)


@torch.library.custom_op("deltaforge::silu_mul", mutates_args=())
def silu_mul(gate: Tensor, up: Tensor) -> Tensor:
    """``F.silu(gate) * up`` in one pass.

    A custom op so `torch.compile` treats it as opaque: inductor may fuse and CUDA-graph
    around it, but may not decompose it back into the elementwise ops this hypothesis
    exists to replace.
    """
    if gate.shape != up.shape:
        raise ValueError(f"gate shape {tuple(gate.shape)} != up shape {tuple(up.shape)}")
    if gate.dtype != up.dtype:
        raise ValueError(f"gate dtype {gate.dtype} != up dtype {up.dtype}")
    require_cuda(gate, "silu_mul")

    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(up)
    n = out.numel()
    if n == 0:  # pragma: no cover - defensive; a zero-width MLP is not a real config
        return out
    block, num_warps = launch_config(min(n, 1024), min_block=256, max_block=1024)
    grid = ((n + block - 1) // block,)
    _silu_mul_kernel[grid](gate, up, out, n, BLOCK=block, num_warps=num_warps)
    return out


@silu_mul.register_fake
def _silu_mul_fake(gate: Tensor, up: Tensor) -> Tensor:
    return torch.empty_like(up)


_PATCHED: dict[str, type] = {}


def _fused_mlp_class() -> type:
    if "mlp" not in _PATCHED:
        from ..reference import SwiGLUMLP

        class FusedSwiGLUMLP(SwiGLUMLP):
            def forward(self, x):
                return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))

        _PATCHED["mlp"] = FusedSwiGLUMLP
    return _PATCHED["mlp"]


def install(model, entry=None) -> None:
    """Swap every layer's MLP. All 32 layers have one, linear-attention or not."""
    if getattr(model, "_deltaforge_fused_swiglu", False):
        return
    patched = _fused_mlp_class()
    for layer in model.layers:
        layer.mlp.__class__ = patched
    model._deltaforge_fused_swiglu = True


def is_installed(model) -> bool:
    return bool(getattr(model, "_deltaforge_fused_swiglu", False))


def correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    import torch.nn.functional as F

    from ..harness.correctness import check_kernel

    width = int(model.layers[0].mlp.gate_proj.out_features)
    target_dtype = dtype if dtype is not None else model.layers[0].mlp.gate_proj.weight.dtype
    generator = torch.Generator(device=device).manual_seed(seed)

    checks = []
    for shape, label in (
        ((1, 1, width), "headline decode step: batch 1, one token"),
        ((32, 1, width), "secondary decode step: batch 32, one token"),
        ((1, 2048, width), "prefill: batch 1, 2048 tokens"),
    ):
        gate = torch.randn(shape, device=device, dtype=target_dtype, generator=generator)
        up = torch.randn(shape, device=device, dtype=target_dtype, generator=generator)
        checks.append(
            check_kernel(
                "fused_swiglu.silu_mul",
                lambda g, u: F.silu(g) * u,
                lambda g, u: silu_mul(g, u),
                args=(gate, up),
                replaces="swiglu_mlp",
                note=label,
            )
        )
    return tuple(checks)
