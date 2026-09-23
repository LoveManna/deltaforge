"""045 — the same four taps, written in operations inductor is allowed to fuse across.

`025-fused-causal-conv` won at 1.0144 on rental 40 and the pair
`039-int4-head-and-conv` lost at **0.8111** on rental 43. The `TORCH_LOGS=output_code`
dump of that exact pair named the mechanism, and it is not the kernel:

    torch.ops.deltaforge.fused_causal_conv_step is an opaque custom op, so it is a
    fusion barrier. Inductor must materialise its inputs and outputs into real
    buffers — 59 allocations became 190 — and it splits a producer chain it had been
    fusing, then recomputes the shared prologue rather than reading the buffer it just
    wrote: the linear-attention state reduction over (1, 32, 128, 128) runs twice per
    layer, 24 times per token.

**A custom op costs its own kernel plus everything inductor can no longer fuse across
it, and that second term is invisible at the call site.** That is the law
`docs/HYPOTHESES.md` entry 8 now carries, and this module is the experiment that decides
whether it is the whole story.

## The mechanism, which is a deletion rather than a kernel

At ``seq_len == 1`` the four-tap depthwise causal convolution is one fused multiply-add
per channel::

    out[b, c] = silu( h[b, c, 0]*w[c, 0] + h[b, c, 1]*w[c, 1] + h[b, c, 2]*w[c, 2] + x[b, c]*w[c, 3] )

which is **pointwise**. Inductor generates pointwise kernels for a living and fuses them
into their neighbours by default. What stopped it here was never the arithmetic: it was
that the reference expresses this as `F.conv1d`, which falls out to
`extern_kernels.convolution` — the one `extern` call left in the whole decode graph —
with a `cat` stranded in front of it and a `copy_` stranded behind.

So this candidate writes the same four taps as torch operations and hands them to
inductor. **There is no Triton in this file and no custom op**, which is the entire
point: the barrier is what we are trying to remove, and a hand-written kernel wrapped in
`torch.library.custom_op` re-erects it one call site later.

## What it is worth, and what separates it from 025

Against the reference it removes, per linear-attention layer, one cuDNN dispatch and the
`cat` that feeds it — and unlike `025` it leaves inductor free to fuse the remainder into
the `qkv` projection's consumer chain, which is where the 24 duplicated reductions came
from. 24 layers of that is **0.08-0.24 ms of the baseline's 7.30, a ratio of 1.011 to
1.033** — the same arithmetic `025` was registered against, because it is the same
saving. The difference between the two slots is not their ceiling. It is their bill.

The pair is what makes either readable:

* `044` (the Triton custom op) and `045` (this) **measured alone, in the same process**,
  say what the barrier costs when nothing else is installed — rental 40's `025` never had
  a control and rental 43 never re-measured it at all.
* `050` and `051` compose each with the int4 head. If `051` recovers what `050` loses,
  the barrier *was* the 19%, and the law above is quantified rather than merely named.
  If both lose, the conv's problem is not its opacity and this file says so.

## Rounding, which is a shared reference rather than a better one

`F.conv1d` accumulates in fp32 and returns the activation dtype, and `F.silu` is applied
to that bf16 tensor. This accumulates in fp32, rounds to the activation dtype, and *then*
gates — the same order `fused_causal_conv` chose and for the same reason: a shared
correctness reference is only shared if the implementations share their rounding
(`AGENT.md` §8, which batch 003 paid for).

Because there is no Triton here, **the CPU suite can check the numerics rather than only
the installation**, which is true of nothing else in this directory. A Triton body is a
string a GPU compiles; this is four multiplies a laptop can run.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "InlineCausalConvDeltaNet",
    "conv_taps",
    "inline_causal_conv_correctness_checks",
    "inline_causal_conv_step",
    "install_inline_causal_conv",
]

#: The kernel width this decomposition is written for. `_causal_conv` keeps `kernel_size`
#: general; four taps is what Qwen3.5 has, and unrolling them is what makes the expression
#: pointwise. A checkpoint with a different width raises rather than silently taking the
#: reference path inside a measured region.
TAPS = 4


def conv_taps(weight: Tensor) -> Tensor:
    """``(C, 1, TAPS)`` conv weight as ``(C, TAPS)``, checked."""
    channels = int(weight.shape[0])
    w = weight.reshape(channels, -1)
    if w.shape[1] != TAPS:
        raise ValueError(f"this decomposition is the {TAPS}-tap case; got {w.shape[1]} taps")
    return w


def inline_causal_conv_step(x: Tensor, conv_cache: Tensor, weight: Tensor) -> Tensor:
    """``x`` is ``(B, C, 1)``, ``conv_cache`` ``(B, C, 3)``, ``weight`` ``(C, 1, 4)``.

    Returns ``(B, C, 1)`` and advances the history, exactly as
    `fused_causal_conv_step` does — in torch operations rather than in one opaque call,
    so that inductor can schedule them with the kernels on either side.
    """
    batch, channels, steps = x.shape
    if steps != 1:
        raise ValueError(f"inline_causal_conv_step is the decode step; got seq_len {steps}")
    if conv_cache.shape != (batch, channels, TAPS - 1):
        raise ValueError(f"expected a (B, C, {TAPS - 1}) history, got {tuple(conv_cache.shape)}")

    w = conv_taps(weight).float()
    h = conv_cache
    xv = x[..., 0]

    acc = (
        h[..., 0].float() * w[:, 0]
        + h[..., 1].float() * w[:, 1]
        + h[..., 2].float() * w[:, 2]
        + xv.float() * w[:, 3]
    )
    # fp32 accumulate, round to the activation dtype, *then* gate — `F.conv1d` returns
    # bf16 and the reference applies `F.silu` to that, so gating the fp32 accumulator
    # would be a different function computed more accurately, which is not the claim.
    out = F.silu(acc.to(x.dtype)).unsqueeze(-1)

    # [h0, h1, h2] <- [h1, h2, x]. `torch.cat` materialises before `copy_` reads back
    # into the same storage, and functionalisation preserves that ordering in the graph.
    conv_cache.copy_(torch.cat((h[..., 1:], x.to(conv_cache.dtype)), dim=-1))
    return out


class InlineCausalConvDeltaNet:
    """Mixin body for the patched `GatedDeltaNet`. Never instantiated directly."""

    def _causal_conv(self, x: Tensor, cache):  # type: ignore[override]
        if cache is None or x.shape[-1] != 1:
            # The prefill, which `run_interleaved` excludes from every timed region by
            # design — the same boundary `fused_causal_conv` draws, and a *sequence
            # length* rather than a device or a dtype. Nothing here ever runs the
            # reference inside a measured region while the harness records it as the
            # candidate.
            return super()._causal_conv(x, cache)
        return inline_causal_conv_step(x, cache.conv, self.conv1d.weight)


_PATCHED: dict[str, type] = {}


def _patched_delta_net_class() -> type:
    if "conv" not in _PATCHED:
        from ..reference import GatedDeltaNet  # noqa: PLC0415

        class InlineConvGatedDeltaNet(InlineCausalConvDeltaNet, GatedDeltaNet):
            pass

        _PATCHED["conv"] = InlineConvGatedDeltaNet
    return _PATCHED["conv"]


def _delta_nets(model):
    from ..reference import GatedDeltaNet  # noqa: PLC0415

    return [module for module in model.modules() if isinstance(module, GatedDeltaNet)]


def install_inline_causal_conv(model, entry=None) -> None:
    if getattr(model, "_deltaforge_inline_causal_conv", False):
        return
    model._deltaforge_inline_causal_conv = True
    patched = _patched_delta_net_class()
    for module in _delta_nets(model):
        module.__class__ = patched


def inline_causal_conv_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """This decomposition against `GatedDeltaNet._causal_conv` itself, on the real weights.

    Identical in structure to `fused_causal_conv_correctness_checks`, deliberately: the
    two slots are read against each other, so they must be checked against the same
    reference expression — including the `cat`, the cuDNN convolution and the `silu` —
    rather than against two re-derivations that could be wrong in different ways.
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
            out = inline_causal_conv_step(a, cache, weight)
            return (out, cache)

        checks.append(
            check_kernel(
                f"inline_causal_conv.step[C={channels} batch={batch}]",
                reference,
                candidate,
                args=(x, history),
                replaces="causal_conv",
                note=f"{note}; output and the advanced history are both compared",
            )
        )
    return tuple(checks)
