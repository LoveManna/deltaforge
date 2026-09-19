"""025 — the 72 launches per token that move 0.05% of the bytes.

Counted out of rental 38's `TORCH_LOGS=output_code` dump, the decode graph issues **508
kernel launches per decoded token** — 483 generated Triton plus 25 `extern_kernels`. Three
of them, per linear-attention layer, belong to a four-tap depthwise causal convolution::

     24  triton_poi_fused__unsafe_view_cat_mm_transpose_5      # cat(history, x)
     24  extern_kernels.convolution                            # F.conv1d, via cuDNN
     24  triton_poi_fused_copy_copy__slice_6                   # cache.conv.copy_(...)

**72 of 508 launches — 14.2% of the dispatch — for 4.6 MB/token, which is 0.05% of the
traffic.** At batch-1 decode that convolution is 8192 channels x 4 taps: 32768 multiplies,
about a microsecond of arithmetic wrapped in three launches and an `extern_kernels` call
into cuDNN, whose dispatch is not cheap and whose algorithm selection is designed for a
problem several orders of magnitude larger.

This is the one place in the model where launch count and byte count are that far apart,
and it is the reason this kernel is worth writing when a fused RMSNorm was not: the
hypothesis is not about the 0.05%, it is about the 14.2%.

## Why inductor leaves it alone, which is the part that makes it a hypothesis

`F.conv1d` is not something inductor generates. It falls out to `extern_kernels.convolution`
— the one `extern` call left in the whole decode graph — and a scheduler cannot fuse a
producer and a consumer across an opaque external call. So the `cat` that builds its input
and the `copy_` that updates the history are stranded on either side of it as their own
kernels, and there is nothing in inductor's repertoire that could join them.

A hand-written kernel can, because at ``seq_len == 1`` the convolution is not a convolution
at all. It is one fused multiply-add of length four per channel:

    out[b, c] = silu( h[b, c, 0]*w[c, 0] + h[b, c, 1]*w[c, 1] + h[b, c, 2]*w[c, 2] + x[b, c]*w[c, 3] )

with the history shifted by one in the same pass. Four loads, three stores, no
communication between programs, and **one launch where there were three**.

## What it is worth

24 layers x 2 launches saved = 48 of 508, or **9.4% of the dispatch**, plus whatever a
cuDNN `extern_kernels.convolution` costs above a Triton launch. Against the 0.9-2.5 ms of
per-launch overhead the dump's arithmetic implies (see `static_cache.py`), that is
**0.08-0.24 ms/token of the baseline's 7.30 — a ratio of 1.011 to 1.033**. The lower end
of that is inside the noise band and the upper end is several times outside it, which is
an honest statement of a hypothesis rather than a prediction of a win, and it is why the
slot also runs composed with `static_decode_cache`: if launches cost nothing once the step
is CUDA-graphed, this kernel should be worth nothing there, and that is a falsifiable pair.

## The prefill takes the reference's path, and says so

`run_interleaved` excludes every `setup` from the timed region and the 2048-token prefill
is setup, so a multi-token call routes back to `GatedDeltaNet._causal_conv` unchanged.
That is the same deliberate boundary `tiled_gemv.GEMV_MAX_ROWS` draws, and it is not the
silent fallback `_triton.require_cuda` forbids: nothing here ever runs the reference inside
a measured region while the harness records it as the candidate.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._triton import HAS_TRITON, require_cuda, tl, triton

__all__ = [
    "FusedCausalConvDeltaNet",
    "fused_causal_conv_correctness_checks",
    "fused_causal_conv_step",
    "install_fused_causal_conv",
]

#: Channels per program. 8192 channels at 256 gives 32 programs per layer, which is small
#: — but this kernel is bound by its launch, not by its bandwidth: it moves 192 KB.
BLOCK_C = 256


if HAS_TRITON:

    @triton.jit
    def _fused_causal_conv_step_kernel(
        X,
        CACHE,
        W,
        OUT,
        C,
        stride_xb,
        stride_xc,
        stride_cb,
        stride_cc,
        stride_ct,
        stride_wc,
        stride_ob,
        stride_oc,
        BLOCK_C: tl.constexpr,
    ):
        """One decode step of a four-tap depthwise causal conv, history shifted in place.

        The rounding is chosen to match the reference exactly rather than to be better than
        it: `F.conv1d` accumulates in fp32 and returns bf16, and `F.silu` is applied to that
        bf16 tensor. So the accumulator is fp32, rounded to the activation dtype, and *then*
        gated. A shared reference is only shared if the implementations share their
        rounding — batch 003 lost a slot to getting that backwards.
        """
        pid_c, pid_b = tl.program_id(0), tl.program_id(1)
        offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = offs < C

        base = CACHE + pid_b * stride_cb + offs * stride_cc
        h0 = tl.load(base + 0 * stride_ct, mask=mask, other=0.0)
        h1 = tl.load(base + 1 * stride_ct, mask=mask, other=0.0)
        h2 = tl.load(base + 2 * stride_ct, mask=mask, other=0.0)
        xv = tl.load(X + pid_b * stride_xb + offs * stride_xc, mask=mask, other=0.0)

        w_base = W + offs * stride_wc
        w0 = tl.load(w_base + 0, mask=mask, other=0.0)
        w1 = tl.load(w_base + 1, mask=mask, other=0.0)
        w2 = tl.load(w_base + 2, mask=mask, other=0.0)
        w3 = tl.load(w_base + 3, mask=mask, other=0.0)

        acc = (
            h0.to(tl.float32) * w0.to(tl.float32)
            + h1.to(tl.float32) * w1.to(tl.float32)
            + h2.to(tl.float32) * w2.to(tl.float32)
            + xv.to(tl.float32) * w3.to(tl.float32)
        )
        rounded = acc.to(X.dtype.element_ty).to(tl.float32)
        out = rounded * tl.sigmoid(rounded)
        tl.store(OUT + pid_b * stride_ob + offs * stride_oc, out.to(X.dtype.element_ty), mask=mask)

        # The shift is the whole reason the history is a cache: [h0, h1, h2] <- [h1, h2, x].
        # Written after every load above, so a program never reads a slot it has overwritten.
        tl.store(base + 0 * stride_ct, h1, mask=mask)
        tl.store(base + 1 * stride_ct, h2, mask=mask)
        tl.store(base + 2 * stride_ct, xv, mask=mask)


@torch.library.custom_op("deltaforge::fused_causal_conv_step", mutates_args=("conv_cache",))
def fused_causal_conv_step(x: Tensor, conv_cache: Tensor, weight: Tensor) -> Tensor:
    """``x`` is ``(B, C, 1)``, ``conv_cache`` ``(B, C, 3)``, ``weight`` ``(C, 1, 4)``.

    Returns ``(B, C, 1)`` and advances the history. Declared as mutating ``conv_cache``
    because it does: the op is opaque to dynamo, and an undeclared mutation would let the
    scheduler reorder a read of the cache across the write.
    """
    # Shapes first, device second: a shape this op does not implement is a programming
    # error on any device, and checking it on a CPU is what lets the CPU suite pin it.
    batch, channels, steps = x.shape
    if steps != 1:
        raise ValueError(f"fused_causal_conv_step is the decode step; got seq_len {steps}")
    if conv_cache.shape != (batch, channels, 3):
        raise ValueError(f"expected a (B, C, 3) history, got {tuple(conv_cache.shape)}")

    w = weight.reshape(channels, -1)
    if w.shape[1] != 4:
        raise ValueError(f"this kernel is the four-tap case; got {w.shape[1]} taps")

    require_cuda(x, "fused_causal_conv_step")

    out = torch.empty_like(x)
    _fused_causal_conv_step_kernel[((channels + BLOCK_C - 1) // BLOCK_C, batch)](
        x,
        conv_cache,
        w,
        out,
        channels,
        x.stride(0),
        x.stride(1),
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        w.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


@fused_causal_conv_step.register_fake
def _fused_causal_conv_step_fake(x: Tensor, conv_cache: Tensor, weight: Tensor) -> Tensor:
    return torch.empty_like(x)


class FusedCausalConvDeltaNet:
    """Mixin body for the patched `GatedDeltaNet`. Never instantiated directly."""

    def _causal_conv(self, x: Tensor, cache):  # type: ignore[override]
        if cache is None or x.shape[-1] != 1:
            # The prefill, which `run_interleaved` excludes from every timed region by
            # design. This is the only condition that routes back to the reference, and it
            # is a *sequence length*, not a device or a dtype: a fallback that fired on
            # "no CUDA" would run the reference inside the measured region while the
            # harness recorded it as the candidate, which is what `_triton.require_cuda`
            # exists to forbid. Anything else this kernel cannot do raises from the op.
            return super()._causal_conv(x, cache)
        return fused_causal_conv_step(x, cache.conv, self.conv1d.weight)


_PATCHED: dict[str, type] = {}


def _patched_delta_net_class() -> type:
    if "conv" not in _PATCHED:
        from ..reference import GatedDeltaNet  # noqa: PLC0415

        class FusedConvGatedDeltaNet(FusedCausalConvDeltaNet, GatedDeltaNet):
            pass

        _PATCHED["conv"] = FusedConvGatedDeltaNet
    return _PATCHED["conv"]


def _delta_nets(model):
    from ..reference import GatedDeltaNet  # noqa: PLC0415

    return [module for module in model.modules() if isinstance(module, GatedDeltaNet)]


def install_fused_causal_conv(model, entry=None) -> None:
    if getattr(model, "_deltaforge_fused_causal_conv", False):
        return
    model._deltaforge_fused_causal_conv = True
    patched = _patched_delta_net_class()
    for module in _delta_nets(model):
        module.__class__ = patched


def fused_causal_conv_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """The kernel against `GatedDeltaNet._causal_conv` itself, on the real conv weights.

    The reference side is the reference method, called on an unpatched module, so what is
    compared is this kernel against the exact expression it replaces — including the
    `cat`, the cuDNN convolution and the `silu` — rather than against a re-derivation of
    it that could be wrong in the same way the kernel is.
    """
    from ..harness.correctness import check_kernel  # noqa: PLC0415
    from ..reference import GatedDeltaNet  # noqa: PLC0415
    from .quantised_linear import _decode_shapes  # noqa: PLC0415

    nets = [module for module in model.modules() if isinstance(module, GatedDeltaNet)]
    if not nets:
        return ()
    net = nets[0]
    weight = net.conv1d.weight
    channels = int(weight.shape[0])
    generator = torch.Generator(device=device).manual_seed(seed)

    checks = []
    for shape, note in _decode_shapes(channels):
        batch = shape[0]
        x = torch.randn((batch, channels, 1), device=device, dtype=weight.dtype, generator=generator)
        history = torch.randn((batch, channels, 3), device=device, dtype=weight.dtype, generator=generator)

        def reference(a, h, net=net):
            from ..reference import _LinearLayerCache  # noqa: PLC0415

            cache = _LinearLayerCache(conv=h.clone(), recurrent=torch.zeros(1, device=h.device))
            out = GatedDeltaNet._causal_conv(net, a, cache)
            return (out, cache.conv)

        def candidate(a, h):
            cache = h.clone()
            out = fused_causal_conv_step(a, cache, weight)
            return (out, cache)

        checks.append(
            check_kernel(
                f"fused_causal_conv.step[C={channels} batch={batch}]",
                reference,
                candidate,
                args=(x, history),
                replaces="causal_conv",
                note=f"{note}; output and the advanced history are both compared",
            )
        )
    return tuple(checks)
