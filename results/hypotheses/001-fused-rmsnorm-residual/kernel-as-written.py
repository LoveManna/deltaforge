# Frozen snapshot of the Triton kernel for hypothesis 001.
#
# Source of truth: src/deltaforge/kernels/fused_rmsnorm_residual.py
# Commit: ab0864b
# Status: written and gated on CPU, NEVER EXECUTED ON A GPU. See README.md here.

"""Hypothesis 001 — fused residual-add + RMSNorm.

**Mechanism.** In a decoder layer the residual add, the norm's reduction and the norm's
rescale each make a separate pass over the full hidden state. At batch 1 that work is
entirely memory-bound: nothing here is arithmetic-limited, so the cost *is* the traffic.
Fusing the add and the norm into one pass reads the hidden state once and writes it
twice (the updated residual stream, and the normalised activation) instead of reading and
writing it repeatedly across three separate kernels.

**What is replaced.** Only the two `hidden_size`-wide norms in each decoder layer, and the
model's final norm. `q_norm`/`k_norm` inside the attention layers are also RMSNorm but run
on a 256-wide head dimension in a different shape regime; leaving them to the reference
keeps the measured win attributable to one thing.

**Numerics.** Deliberately identical to `reference.RMSNorm`, step for step:

* the residual add happens in the tensor dtype, so the sum is rounded to bf16 *before* the
  norm sees it — matching `residual + hidden_states` in the reference, which produces a
  bf16 tensor that `RMSNorm.forward` then upcasts;
* the reduction and the rescale are computed in fp32;
* the scale is `1 + weight`, Qwen3.5's zero-centred convention. Reading it as plain
  `weight` produces an all-zero activation and is the classic way to get this model
  silently wrong.

**No silent fallback.** These ops raise on a non-CUDA tensor rather than quietly running
PyTorch. A candidate column that fell back to the reference would benchmark the baseline
while labelling it the candidate, which is the worst failure this harness could have.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "HAS_TRITON",
    "add_rms_norm",
    "install",
    "is_installed",
    "rms_norm",
]

try:  # The CPU dev environment has no Triton, and importing the registry must still work.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised by the CPU CI environment
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _add_rms_norm_kernel(
        X,  # activation to add into the residual stream
        R,  # residual stream in
        W,  # norm weight, zero-centred
        RESID,  # residual stream out (X + R, rounded to the tensor dtype)
        OUT,  # normalised activation out
        stride_row,
        N: tl.constexpr,
        EPS,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        offs = row * stride_row + cols

        x = tl.load(X + offs, mask=mask, other=0.0)
        r = tl.load(R + offs, mask=mask, other=0.0)

        # Add in the tensor dtype, exactly as `residual + hidden_states` does, then read
        # that rounded value back for the norm. Normalising the unrounded fp32 sum would
        # be *more* accurate than the reference and would therefore not be the same
        # function — the correctness gate compares against the reference, not against
        # infinite precision.
        summed = (x.to(tl.float32) + r.to(tl.float32)).to(RESID.dtype.element_ty)
        tl.store(RESID + offs, summed, mask=mask)

        s = summed.to(tl.float32)
        var = tl.sum(s * s, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + EPS)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = s * rstd * (1.0 + w)
        tl.store(OUT + offs, y.to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _rms_norm_kernel(
        X,
        W,
        OUT,
        stride_row,
        N: tl.constexpr,
        EPS,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        offs = row * stride_row + cols

        s = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(s * s, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + EPS)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = s * rstd * (1.0 + w)
        tl.store(OUT + offs, y.to(OUT.dtype.element_ty), mask=mask)


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _launch_config(n: int) -> tuple[int, int, int]:
    """``(BLOCK, num_warps, num_stages)`` for a one-row-per-program reduction.

    One program per row keeps the reduction in registers with no cross-block
    communication. 2560 elements at bf16 is 5 KB — comfortably register-resident.
    """
    block = _next_power_of_two(n)
    if block < 128:
        block = 128
    if block > 65536:
        raise ValueError(f"row of {n} elements is too wide for a single-program reduction")
    num_warps = min(32, max(4, block // 256))
    return block, num_warps, 1


def _check(x: Tensor, weight: Tensor, name: str, residual: Tensor | None = None) -> None:
    """Everything the kernels assume, checked before a launch can read the wrong memory.

    Layout is *coerced* rather than rejected — the reference norm accepts any layout, so
    refusing a non-contiguous activation would make the candidate fail where the baseline
    works. Shape and dtype disagreements are refused, because the kernel indexes rows by a
    single stride and would silently read the wrong elements instead of failing.

    The structural checks come before the device check on purpose: a shape bug should be
    reported as a shape bug on any device, and putting the device check first would let a
    CPU test pass without ever reaching the structural ones.
    """
    if weight.dim() != 1 or not weight.is_contiguous():
        raise ValueError("norm weight must be a contiguous 1-D tensor; the kernel indexes it by column")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"weight has {weight.shape[0]} elements, activation row is {x.shape[-1]}")
    if residual is not None:
        # Broadcasting would change what is computed while still launching cleanly, so a
        # mismatch has to be an error rather than a shape the kernel quietly reinterprets.
        if residual.shape != x.shape:
            raise ValueError(f"residual shape {tuple(residual.shape)} != activation shape {tuple(x.shape)}")
        if residual.dtype != x.dtype:
            raise ValueError(f"residual dtype {residual.dtype} != activation dtype {x.dtype}")
    if not x.is_cuda:
        raise RuntimeError(
            f"deltaforge::{name} requires a CUDA tensor and has no PyTorch fallback, on "
            "purpose: falling back here would run the reference while the harness "
            "recorded it as the candidate."
        )


@torch.library.custom_op("deltaforge::add_rms_norm", mutates_args=())
def add_rms_norm(x: Tensor, residual: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """``(residual + x, rms_norm(residual + x))`` in one pass.

    Registered as a custom op so `torch.compile` treats it as opaque: inductor may fuse
    and CUDA-graph everything around it, but it may not decompose it back into the
    elementwise ops this hypothesis exists to replace.
    """
    _check(x, weight, "add_rms_norm", residual=residual)
    x = x.contiguous()
    residual = residual.contiguous()
    rows = x.numel() // x.shape[-1]
    n = x.shape[-1]
    out = torch.empty_like(x)
    resid_out = torch.empty_like(x)
    block, num_warps, num_stages = _launch_config(n)
    _add_rms_norm_kernel[(rows,)](
        x,
        residual,
        weight,
        resid_out,
        out,
        x.stride(-2) if x.dim() > 1 else n,
        N=n,
        EPS=eps,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return resid_out, out


@add_rms_norm.register_fake
def _add_rms_norm_fake(x: Tensor, residual: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


@torch.library.custom_op("deltaforge::rms_norm", mutates_args=())
def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm with the `1 + weight` scale, matching `reference.RMSNorm.forward`."""
    _check(x, weight, "rms_norm")
    x = x.contiguous()
    rows = x.numel() // x.shape[-1]
    n = x.shape[-1]
    out = torch.empty_like(x)
    block, num_warps, num_stages = _launch_config(n)
    _rms_norm_kernel[(rows,)](
        x,
        weight,
        out,
        x.stride(-2) if x.dim() > 1 else n,
        N=n,
        EPS=eps,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@rms_norm.register_fake
def _rms_norm_fake(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return torch.empty_like(x)


# --------------------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------------------
#
# `reference.py` never changes to accommodate a kernel, so the swap happens here, against
# a *built* model. The reference decoder layer runs:
#
#     residual = h;  h = input_layernorm(h);        h = attn(h);  h = residual + h   (A)
#     residual = h;  h = post_attention_layernorm(h); h = mlp(h);  return residual + h
#
# The add at (A) and the `post_attention_layernorm` that immediately follows it are the
# fusable pair. The layer's final add is fusable with the *next* layer's
# `input_layernorm`, but that pair straddles the layer boundary and threading a residual
# through `ReferenceModel.forward` would restructure the model, so this hypothesis leaves
# it alone and says so in the writeup. Per layer this replaces four passes with three.


_FUSED_CLASSES: dict[str, type] = {}


def _fused_classes() -> dict[str, type]:
    """Build the patched subclasses once, on first install.

    Installing by swapping an *instance's* `__class__` rather than by assigning a bound
    method to `layer.forward`: Dynamo then sees an ordinary `nn.Module` with an ordinary
    `forward`, which is what keeps the compiled candidate column free of graph breaks.
    Swapping the class on one instance cannot leak into the reference model — the
    `eager` and `compiled` columns build their own, and they keep the reference classes.
    """
    if _FUSED_CLASSES:
        return _FUSED_CLASSES

    from ..config import LINEAR_ATTENTION
    from ..reference import DecoderLayer, RMSNorm

    class FusedDecoderLayer(DecoderLayer):
        def forward(self, hidden_states, cos, sin, cache=None, cache_offset=0):
            normed = rms_norm(hidden_states, self.input_layernorm.weight, self.input_layernorm.eps)
            if self.layer_type == LINEAR_ATTENTION:
                layer_cache = cache.linear(self.layer_idx) if cache is not None else None
                attn_out = self.linear_attn(normed, layer_cache)
            else:
                layer_cache = cache.attention(self.layer_idx) if cache is not None else None
                attn_out = self.self_attn(normed, cos, sin, layer_cache, cache_offset)

            residual, normed = add_rms_norm(
                attn_out,
                hidden_states,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
            )
            return residual + self.mlp(normed)

    class FusedRMSNorm(RMSNorm):
        def forward(self, x):
            return rms_norm(x, self.weight, self.eps)

    _FUSED_CLASSES.update(layer=FusedDecoderLayer, norm=FusedRMSNorm)
    return _FUSED_CLASSES


def install(model, entry=None) -> None:
    """Splice the fused ops into a built `ReferenceModel`, in place.

    Idempotent: installing twice is a no-op rather than a silent double-patch.
    """
    if getattr(model, "_deltaforge_fused_rmsnorm", False):
        return
    classes = _fused_classes()
    for layer in model.layers:
        layer.__class__ = classes["layer"]
    model.norm.__class__ = classes["norm"]
    model._deltaforge_fused_rmsnorm = True


def is_installed(model) -> bool:
    """True when `install` has patched this model. The correctness gate asserts this
    before trusting a candidate measurement."""
    return bool(getattr(model, "_deltaforge_fused_rmsnorm", False))


# --------------------------------------------------------------------------------------
# Layer-1 correctness checks
# --------------------------------------------------------------------------------------


def correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """Per-kernel `allclose` checks against the reference operation on real shapes.

    The reference side calls the *actual reference modules* off a weight-loaded model
    rather than re-deriving the formula here. A re-derivation could agree with the kernel
    and both be wrong about what `reference.py` does, which is the one thing this gate
    exists to rule out.
    """
    from ..harness.correctness import check_kernel

    layer = model.layers[0]
    w_in = layer.input_layernorm.weight
    w_post = layer.post_attention_layernorm.weight
    eps = layer.input_layernorm.eps
    hidden = int(w_in.shape[0])
    dtype = dtype if dtype is not None else w_in.dtype

    generator = torch.Generator(device=device).manual_seed(seed)
    checks = []
    for shape, label in (
        ((1, 1, hidden), "headline decode step: batch 1, one token"),
        ((32, 1, hidden), "secondary decode step: batch 32, one token"),
        ((1, 2048, hidden), "prefill: batch 1, 2048 tokens"),
    ):
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        residual = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        checks.append(
            check_kernel(
                "fused_rmsnorm_residual.rms_norm",
                lambda t: layer.input_layernorm(t),
                lambda t: rms_norm(t, w_in, eps),
                args=(x,),
                replaces="rms_norm",
                note=label,
            )
        )
        checks.append(
            check_kernel(
                "fused_rmsnorm_residual.add_rms_norm",
                lambda a, b: (b + a, layer.post_attention_layernorm(b + a)),
                lambda a, b: add_rms_norm(a, b, w_post, eps),
                args=(x, residual),
                replaces="rms_norm_residual",
                note=label,
            )
        )
    return tuple(checks)
