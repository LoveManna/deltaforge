"""055 and 056 — the champion's own kernel, stripped of the thing that cost the conv 21%.

Rental 45 measured the price of opacity on the causal convolution and it was enormous:
`044-fused-causal-conv` and `045-inline-causal-conv` compute the same function — both
bit-identical to the reference — and returned **0.7854 and 1.0765**. The only difference
was whether the arithmetic reached inductor as a `torch.library.custom_op` it must fence
against, or as operations it could schedule.

**`tiled_gemv_int4` — the champion of `decode_step` — is a `torch.library.custom_op`.**
It appears in rental 45's candidate graph as four dispatches inductor cannot see into, and
the dump shows what sits next to it: the reference fuses the final RMSNorm *into* the
lm_head matmul (`triton_red_fused__to_copy__unsafe_view_add_mean_mm_mul_pow_rsqrt_slice_t_view_43`),
and the candidate cannot. So the head has real fusion to lose, and nobody has measured what
losing it costs.

This module holds the two ways of not paying it, and the batch runs both beside the
unchanged custom op so the comparison is one variable at a time.

## `tiled_gemv_int4_visible` — the same kernel, made visible

`torch.library.triton_op` puts the Triton launch in the graph as a structured node rather
than an opaque call: inductor knows the kernel's inputs, outputs and mutation semantics,
so it can plan buffers around it instead of materialising defensively, and it need not
assume the call clobbers state it was tracking. **The kernel body is byte-for-byte the one
the champion runs** — same tile, same arithmetic, same summation order — so a difference
between `054` and `055` is the registration and nothing else.

What it does *not* do is let inductor fuse a pointwise producer *into* a user-written
Triton kernel; that is not a thing either registration supports. So the expected effect is
smaller than the conv's, and the mechanism says why: the conv sat inside 24 layers between
a fused producer and a fused consumer, and the head sits at the end of the model with one
reduction next to it and one call site.

## `DequantInt4LMHead` — no kernel at all, which is the harder question

The alternative to making our kernel visible is not writing one. This expresses the same
program as torch operations — unpack the nibbles, apply the group scales, round to bf16,
matmul in fp32 — and hands the whole thing to `max-autotune`. **Exactly the expression
`tiled_int4_correctness_checks` already uses as the kernel's reference**, which is why its
layer-2 numbers must come back as `054`'s to the digit: same arithmetic, different author.

`docs/HYPOTHESES.md` entry 1 has one measurement of this shape — `010-int8-dequant-torch`,
rental 37, **0.9893** — and it is not the same experiment. That was int8 on all 248 layer
projections, which batch 005 later showed cannot distinguish a kernel from a site. **This
is int4 on the one site with parallelism to spare**, and its outcome decides something
the project has never established: whether the hand-written kernel is *necessary* here, or
merely sufficient.

A bandwidth budget bounds the losing branch. The head's bf16 weight is 1271.40 MB/token.
If inductor materialises it, the candidate moves that *plus* the 317.85 MB of packed
nibbles it read to build it, and the slot cannot beat 1.0 however good the matmul is. If
it fuses the unpack into the matmul's prologue, the slot collects the same 1.1249x ceiling
the champion collects. **There is no middle outcome that is hard to read**, which is what
makes it worth three minutes.

## Rental 46 answered it, and batch 011 spends the answer

It fuses. `056` returned **1.0171** against the hand-written kernel's **0.9851** in the same
process, and the dump holds one reduction kernel carrying the grouped unpack, the `mm`, the
final RMSNorm and the residual add together, with no weight-sized buffer in the graph. So
this module stopped being a two-slot experiment about registrations and became the way this
project quantises: `install_int4_mlp_torch_dequant` takes the same construction to **52.75%
of per-token bytes** and `install_int4_wide_torch_dequant` to **97.85%**, neither of them
containing a kernel of ours.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ._triton import require_cuda
from .tiled_gemv import (
    GEMV_MAX_ROWS,
    _flatten,
    _launch_shape,
    _quantised_head_model_class,
    _reduce,
    clear_launch_shape,
    dequantise_int4_k_major,
    head_shape_key,
    quantise_int4_k_major,
)

__all__ = [
    "DequantInt4LMHead",
    "DequantInt4LMHeadBf16",
    "VisibleInt4LMHead",
    "DequantInt4Linear",
    "DequantInt4LinearBf16",
    "install_int4_head_torch_dequant",
    "install_int4_head_torch_dequant_bf16",
    "install_int4_head_triton_op",
    "install_int4_mlp_torch_dequant",
    "install_int4_mlp_torch_dequant_bf16",
    "install_int4_wide_torch_dequant",
    "install_int4_wide_torch_dequant_bf16",
    "int4_head_torch_dequant_bf16_correctness_checks",
    "int4_head_torch_dequant_correctness_checks",
    "int4_head_triton_op_correctness_checks",
    "int4_mlp_torch_dequant_bf16_correctness_checks",
    "int4_mlp_torch_dequant_correctness_checks",
    "int4_wide_torch_dequant_bf16_correctness_checks",
    "int4_wide_torch_dequant_correctness_checks",
    "tiled_gemv_int4_visible",
    "torch_dequant_gemv_int4",
    "torch_dequant_gemv_int4_bf16",
]


_HAS_TRITON_OP = hasattr(torch.library, "triton_op") and hasattr(torch.library, "wrap_triton")


def _int4_visible_body(x: Tensor, packed: Tensor, scale: Tensor, group_size: int) -> Tensor:
    """`tiled_gemv.tiled_gemv_int4`'s body, launching through `wrap_triton`.

    Kept as a separate function so the registration decorator is the only difference
    between this and the champion: a divergence in the body would turn the slot pair into
    two unrelated measurements, which is exactly what `044` against `045` was built to
    avoid.
    """
    require_cuda(x, "tiled_gemv_int4_visible")
    half, n = packed.shape
    k = half * 2
    flat = _flatten(x, k)
    rows = flat.shape[0]
    if rows > GEMV_MAX_ROWS:
        dense = dequantise_int4_k_major(packed, scale, group_size=group_size)
        return (flat.float() @ dense).to(flat.dtype).reshape(*x.shape[:-1], n)

    block_n, block_k, split_k, num_warps, num_stages = _launch_shape(n, k, "int4")
    block_k = min(block_k, group_size)
    split_k = max(1, min(split_k, half // block_k))
    partials = torch.empty((split_k, rows, n), device=flat.device, dtype=torch.float32)
    # Fetched here rather than imported: the kernel only exists when Triton does, and
    # this module must import on the CPU box that runs the test suite.
    from . import tiled_gemv as _tg  # noqa: PLC0415

    launcher = torch.library.wrap_triton(_tg._tiled_gemv_int4_kernel)
    launcher[((n + block_n - 1) // block_n, split_k, rows)](
        flat,
        packed,
        scale,
        partials,
        rows,
        n,
        half,
        flat.stride(0),
        packed.stride(0),
        scale.stride(0),
        partials.stride(0),
        partials.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        GROUP=group_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return _reduce(partials, None, flat.dtype).reshape(*x.shape[:-1], n)


if _HAS_TRITON_OP:
    # Registered whether or not Triton is installed: the decorator only builds the op, and
    # `wrap_triton` is not reached until a call. That is what lets the CPU suite assert the
    # registration exists, which is the entire content of this slot.
    tiled_gemv_int4_visible = torch.library.triton_op("deltaforge::tiled_gemv_int4_visible", mutates_args=())(
        _int4_visible_body
    )
else:  # pragma: no cover - every torch this project runs has had it since 2.6

    def tiled_gemv_int4_visible(*args, **kwargs):  # type: ignore[misc]
        """Refuse loudly rather than run the body unregistered.

        This slot's whole claim is the *registration*: the body is the champion's, byte for
        byte. Falling back to calling it directly would install a candidate the manifest
        does not describe and report a plausible ratio for it, which is the silent-fallback
        failure `AGENT.md` §8 keeps paying for.
        """
        raise RuntimeError(
            "torch.library.triton_op is unavailable on this torch build, and this "
            "candidate is defined by it. Refusing to run the kernel unregistered: the "
            "ratio would describe a different candidate than the manifest names."
        )


def torch_dequant_gemv_int4(x: Tensor, packed: Tensor, scale: Tensor, group_size: int) -> Tensor:
    """The same program in torch operations, with no op of ours in the graph at all.

    The expression is `tiled_int4_correctness_checks`'s reference verbatim — dequantise,
    **round the scaled weight to bf16** because the kernel does, then accumulate in fp32 —
    so this and the champion compute the same function and their layer-2 numbers must
    agree. A shared reference is only shared if the implementations share their rounding;
    batch 003 lost a slot to getting that backwards.
    """
    n = packed.shape[1]
    flat = _flatten(x, packed.shape[0] * 2)
    dense = dequantise_int4_k_major(packed, scale, group_size=group_size).to(x.dtype)
    return (flat.float() @ dense.float()).to(x.dtype).reshape(*x.shape[:-1], n)


def torch_dequant_gemv_int4_bf16(x: Tensor, packed: Tensor, scale: Tensor, group_size: int) -> Tensor:
    """`torch_dequant_gemv_int4` with one operator changed: the matmul stays in bf16.

    The expression above asks for `flat.float() @ dense.float()`, and rental 56's dump says
    that line is what decides the MLP slot. At the 96 MLP projections inductor emits a
    pointwise kernel that writes two complete `(2560, 9216)` **fp32** weights and a
    *separate* reduction for the `mm` -- 94.4 MB a site, **18.12 GB/token** against a
    reference that moves 8587.80 -- where at the 248320-wide head the same source fuses the
    whole grouped unpack into the matmul prologue with no weight-sized buffer in the graph.
    `AGENT.md` §1.2 has the dump.

    Two things follow from the dtype, and the slot pair cannot separate them -- which is
    fine, because either one would be the answer. An fp32 operand is **twice the bytes** of
    a bf16 one, so materialising it costs twice as much when inductor chooses to; and an
    fp32 `mm` cannot use the tensor cores a bf16 `mm` does, so the dtype plausibly selects
    the `mm` template over a reduction that could have carried the prologue.

    **bf16 in with fp32 accumulation is also what `tiled_gemv_int4` does**, so this, not the
    fp32 expression, is the torch spelling of the kernel rental 46 retired. The fp32 variant
    was written as the champion's *correctness reference*, where fp32 accumulation is a
    virtue, and it became the shipped candidate without anyone asking whether the reference's
    dtype belonged in a benchmark.

    One operator apart, so a difference between the two slots is the matmul dtype and
    nothing else.
    """
    n = packed.shape[1]
    flat = _flatten(x, packed.shape[0] * 2)
    dense = dequantise_int4_k_major(packed, scale, group_size=group_size).to(x.dtype)
    return (flat @ dense).reshape(*x.shape[:-1], n)


class VisibleInt4LMHead(nn.Module):
    """`TiledLMHead` at int4, launching through `triton_op` instead of `custom_op`."""

    def __init__(self, packed: Tensor, scale: Tensor, group_size: int) -> None:
        super().__init__()
        self.register_buffer("w_k_major", packed, persistent=False)
        self.register_buffer("qscale", scale, persistent=False)
        self.kind = "int4"
        self.qgroup = group_size

    def forward(self, hidden_states: Tensor) -> Tensor:
        return tiled_gemv_int4_visible(hidden_states, self.w_k_major, self.qscale, self.qgroup)


class DequantInt4LMHead(nn.Module):
    """The same weights and the same arithmetic, with the code left to `max-autotune`."""

    def __init__(self, packed: Tensor, scale: Tensor, group_size: int) -> None:
        super().__init__()
        self.register_buffer("w_k_major", packed, persistent=False)
        self.register_buffer("qscale", scale, persistent=False)
        self.kind = "int4"
        self.qgroup = group_size

    def forward(self, hidden_states: Tensor) -> Tensor:
        return torch_dequant_gemv_int4(hidden_states, self.w_k_major, self.qscale, self.qgroup)


class DequantInt4LMHeadBf16(nn.Module):
    """`DequantInt4LMHead` with the bf16 matmul. The `forward` is the only difference."""

    def __init__(self, packed: Tensor, scale: Tensor, group_size: int) -> None:
        super().__init__()
        self.register_buffer("w_k_major", packed, persistent=False)
        self.register_buffer("qscale", scale, persistent=False)
        self.kind = "int4"
        self.qgroup = group_size

    def forward(self, hidden_states: Tensor) -> Tensor:
        return torch_dequant_gemv_int4_bf16(hidden_states, self.w_k_major, self.qscale, self.qgroup)


class DequantInt4Linear(nn.Linear):
    """`torch_dequant_gemv_int4` behind an `nn.Linear`, for the 96 MLP projections.

    ``w_k_major`` and ``qscale`` are **buffers**, not parameters, so
    `cli._assert_parameters_are_shared` still sees only the reference's own weights.
    """

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = torch_dequant_gemv_int4(x, self.w_k_major, self.qscale, self.qgroup)
        return out if self.bias is None else out + self.bias


class DequantInt4LinearBf16(nn.Linear):
    """`DequantInt4Linear` with the bf16 matmul, for the 96 MLP projections.

    The site this pair exists to measure: the buffers, the quantisation and the sites are
    `DequantInt4Linear`'s, and the matmul dtype is the variable.
    """

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = torch_dequant_gemv_int4_bf16(x, self.w_k_major, self.qscale, self.qgroup)
        return out if self.bias is None else out + self.bias


def _install_dequant_linears(linears, cls: type) -> None:
    """Quantise ``linears`` to group-128 k-major int4 and swap in ``cls``.

    `install_int4_mlp_torch_dequant`'s loop, factored out when the bf16 variants would have
    made a third and fourth copy of it. Callers keep their own idempotence guard, because
    the wide installer has a second step the guard also has to cover.
    """
    for linear in linears:
        quantised = quantise_int4_k_major(linear.weight.detach())
        linear.register_buffer("w_k_major", quantised.qweight, persistent=False)
        linear.register_buffer("qscale", quantised.scale, persistent=False)
        linear.qgroup = quantised.group_size
        linear.__class__ = cls


def install_int4_mlp_torch_dequant(model, entry=None) -> None:
    """The MLP's 96 projections at int4, dequantised in torch. **52.75% of per-token bytes.**

    Gated in the manifest on the head slot winning, because it is the same question asked
    of the largest homogeneous block in the model and it is only worth asking once the
    cheap version has said the compiler can fuse a grouped dequantisation at all.
    """
    from .tiled_gemv import _mlp_linears  # noqa: PLC0415

    if getattr(model, "_deltaforge_int4_mlp_torch_dequant", False):
        return
    model._deltaforge_int4_mlp_torch_dequant = True
    for linear in _mlp_linears(model):
        quantised = quantise_int4_k_major(linear.weight.detach())
        linear.register_buffer("w_k_major", quantised.qweight, persistent=False)
        linear.register_buffer("qscale", quantised.scale, persistent=False)
        linear.qgroup = quantised.group_size
        linear.__class__ = DequantInt4Linear


def install_int4_wide_torch_dequant(model, entry=None) -> None:
    """Every layer projection except the 32-channel gates, **and the tied head**, in torch.

    `tiled_gemv.install_tiled_int4_wide`'s sites with `tiled_gemv.install_tiled_int4_wide`'s
    quantisation, and none of its Triton: 200 projections plus the head, **97.85% of what
    the compiled column moves at a 3.7578x ceiling**. The head is inside this installer
    rather than composed beside it for the reason the Triton version gives — both would
    claim `decode_step` in the registry, and one champion per operation is the invariant
    that makes "what does `model.py` assemble?" answerable.

    The gates are excluded deliberately. `in_proj_a` and `in_proj_b` are 7.86 MB/token
    between them — **0.09% of the bytes and the whole of the numerical risk**, because they
    feed an exponential through `A_log`, so quantising them buys nothing measurable and
    puts the slot's correctness gate at the mercy of the one site that cannot repay it.
    """
    from .tiled_gemv import _wide_linears  # noqa: PLC0415

    if getattr(model, "_deltaforge_int4_wide_torch_dequant", False):
        return
    model._deltaforge_int4_wide_torch_dequant = True
    for linear in _wide_linears(model):
        quantised = quantise_int4_k_major(linear.weight.detach())
        linear.register_buffer("w_k_major", quantised.qweight, persistent=False)
        linear.register_buffer("qscale", quantised.scale, persistent=False)
        linear.qgroup = quantised.group_size
        linear.__class__ = DequantInt4Linear
    _install_alternative_head(model, DequantInt4LMHead)


def install_int4_mlp_torch_dequant_bf16(model, entry=None) -> None:
    """`install_int4_mlp_torch_dequant`'s sites and quantisation with a bf16 matmul.

    **52.75% of per-token bytes**, one operator from the slot beside it. Ungated: rental
    56's dump is a prediction about generated code, and a dump the CPU backend cannot
    reproduce is not a reason to decline the largest prize in the backlog for a second
    rental -- see `AGENT.md` §1.2 and `docs/BATCHES.md` on what rental 46's floor cost.
    """
    from .tiled_gemv import _mlp_linears  # noqa: PLC0415

    if getattr(model, "_deltaforge_int4_mlp_torch_dequant_bf16", False):
        return
    model._deltaforge_int4_mlp_torch_dequant_bf16 = True
    _install_dequant_linears(_mlp_linears(model), DequantInt4LinearBf16)


def install_int4_wide_torch_dequant_bf16(model, entry=None) -> None:
    """`install_int4_wide_torch_dequant`'s 200 sites and tied head with a bf16 matmul.

    **97.85% of what the compiled column moves.** The gates stay in bf16 for the reason the
    fp32 installer gives: 0.09% of the bytes and the whole of the numerical risk.
    """
    from .tiled_gemv import _wide_linears  # noqa: PLC0415

    if getattr(model, "_deltaforge_int4_wide_torch_dequant_bf16", False):
        return
    model._deltaforge_int4_wide_torch_dequant_bf16 = True
    _install_dequant_linears(_wide_linears(model), DequantInt4LinearBf16)
    _install_alternative_head(model, DequantInt4LMHeadBf16)


def _dequant_reference_fp32(a: Tensor, p: Tensor, s: Tensor, g: int) -> Tensor:
    """`torch_dequant_gemv_int4`'s arithmetic, spelled out: fp32 matmul, bf16 weight.

    Named rather than inlined because it is now one of **two** reference expressions and
    each candidate must be checked against its own. Batch 003 reported a relative error of
    0.45 for correct code by checking an implementation against a reference whose rounding
    differed, and a second rounding now exists in this module.
    """
    return (
        (a.reshape(-1, a.shape[-1]).float() @ dequantise_int4_k_major(p, s, group_size=g).to(a.dtype).float())
        .to(a.dtype)
        .reshape(*a.shape[:-1], p.shape[1])
    )


def _dequant_reference_bf16(a: Tensor, p: Tensor, s: Tensor, g: int) -> Tensor:
    """`torch_dequant_gemv_int4_bf16`'s arithmetic: bf16 matmul, fp32 accumulation.

    **Not interchangeable with `_dequant_reference_fp32`.** The two differ by the matmul's
    accumulation, which over K=2560 is real and is the whole content of the slot pair, so
    checking the bf16 candidate against the fp32 reference would score the hypothesis as a
    correctness failure. Layer 1 therefore establishes only that the installed module reads
    the weights it was given; the arithmetic is scored by layer 2 against the reference
    *model*, which is where a dtype that genuinely changed the function would show up.
    """
    return (a.reshape(-1, a.shape[-1]) @ dequantise_int4_k_major(p, s, group_size=g).to(a.dtype)).reshape(
        *a.shape[:-1], p.shape[1]
    )


def _torch_dequant_checks(probes, *, device, seed, label="torch_dequant_gemv_int4", fn=None, reference=None):
    """One `check_kernel` per (shape, decode shape) over ``probes``.

    Shared by the MLP and the wide installers, in both dtypes, so all four report the same
    check names and each reports against the reference expression that matches its own
    rounding. ``fn`` and ``reference`` must be the pair for one dtype -- see
    `_dequant_reference_bf16` for what crossing them would score.
    """
    from ..harness.correctness import check_kernel  # noqa: PLC0415
    from .quantised_linear import _decode_shapes  # noqa: PLC0415

    fn = torch_dequant_gemv_int4 if fn is None else fn
    reference = _dequant_reference_fp32 if reference is None else reference

    checks = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for weight, probe_label in probes:
        quantised = quantise_int4_k_major(weight.detach())
        packed, scale, group = quantised.qweight, quantised.scale, quantised.group_size
        k = packed.shape[0] * 2
        for shape, note in _decode_shapes(k):
            x = torch.randn(shape, device=device, dtype=weight.dtype, generator=generator)
            checks.append(
                check_kernel(
                    f"visible_int4_head.{label}[{probe_label}]",
                    lambda a, p, s, g=group, ref=reference: ref(a, p, s, g),
                    lambda a, p, s, g=group, impl=fn: impl(a, p, s, g),
                    args=(x, packed, scale),
                    replaces="decode_step",
                    note=f"{note}; N={packed.shape[1]} K={k}",
                )
            )
    return tuple(checks)


def int4_mlp_torch_dequant_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """Every distinct MLP shape: 9216x2560 and 2560x9216."""
    from .tiled_gemv import _mlp_linears, _shape_probes  # noqa: PLC0415

    return _torch_dequant_checks(_shape_probes(_mlp_linears(model)), device=device, seed=seed)


def int4_mlp_torch_dequant_bf16_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """The same MLP shapes, against the bf16 expression this candidate computes."""
    from .tiled_gemv import _mlp_linears, _shape_probes  # noqa: PLC0415

    return _torch_dequant_checks(
        _shape_probes(_mlp_linears(model)),
        device=device,
        seed=seed,
        label="torch_dequant_gemv_int4_bf16",
        fn=torch_dequant_gemv_int4_bf16,
        reference=_dequant_reference_bf16,
    )


def int4_wide_torch_dequant_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """Every distinct shape among the 200 wide projections, plus the tied head."""
    from .tiled_gemv import _shape_probes, _wide_linears  # noqa: PLC0415

    probes = _shape_probes(_wide_linears(model), head=model.lm_head_weight)
    return _torch_dequant_checks(probes, device=device, seed=seed)


def int4_wide_torch_dequant_bf16_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """The same 200 wide shapes and the tied head, against the bf16 expression."""
    from .tiled_gemv import _shape_probes, _wide_linears  # noqa: PLC0415

    probes = _shape_probes(_wide_linears(model), head=model.lm_head_weight)
    return _torch_dequant_checks(
        probes,
        device=device,
        seed=seed,
        label="torch_dequant_gemv_int4_bf16",
        fn=torch_dequant_gemv_int4_bf16,
        reference=_dequant_reference_bf16,
    )


def _install_alternative_head(model, head_cls: type) -> None:
    """Quantise the tied head and install ``head_cls``, exactly as `_install_head` does.

    The class swap, the attribute and the quantised weights are `tiled_gemv`'s own, so the
    only thing that differs between the champion's slot and these is which `forward` runs.
    Anything else would make the comparison a comparison of two installations.
    """
    quantised = quantise_int4_k_major(model.lm_head_weight)
    head = head_cls(quantised.qweight, quantised.scale, quantised.group_size)
    model.tiled_lm_head = head.to(model.lm_head_weight.device)
    # `_TUNED` is process-global and a batch runs every slot in one process, so a tile a
    # previous slot pinned would silently follow this candidate. Clear it, as
    # `_install_head` does for the same reason.
    clear_launch_shape(*head_shape_key(model.tiled_lm_head))
    model.__class__ = _quantised_head_model_class(type(model))


def install_int4_head_triton_op(model, entry=None) -> None:
    if getattr(model, "_deltaforge_int4_head_triton_op", False):
        return
    model._deltaforge_int4_head_triton_op = True
    _install_alternative_head(model, VisibleInt4LMHead)


def install_int4_head_torch_dequant(model, entry=None) -> None:
    if getattr(model, "_deltaforge_int4_head_torch_dequant", False):
        return
    model._deltaforge_int4_head_torch_dequant = True
    _install_alternative_head(model, DequantInt4LMHead)


def install_int4_head_torch_dequant_bf16(model, entry=None) -> None:
    """The champion's site and weights with a bf16 matmul. One operator from `056`.

    The head is where the fp32 expression already *wins* (1.0171, rental 46), so this is
    the control the MLP pair needs: if bf16 is neutral here and decisive there, the effect
    is the materialisation and not the matmul.
    """
    if getattr(model, "_deltaforge_int4_head_torch_dequant_bf16", False):
        return
    model._deltaforge_int4_head_torch_dequant_bf16 = True
    _install_alternative_head(model, DequantInt4LMHeadBf16)


def _head_checks(model, *, device, seed, label, fn, reference=None):
    from ..harness.correctness import check_kernel  # noqa: PLC0415
    from .quantised_linear import _decode_shapes  # noqa: PLC0415

    reference = _dequant_reference_fp32 if reference is None else reference
    weight = model.lm_head_weight
    quantised = quantise_int4_k_major(weight.detach())
    packed, scale, group = quantised.qweight, quantised.scale, quantised.group_size
    k = packed.shape[0] * 2
    generator = torch.Generator(device=device).manual_seed(seed)

    checks = []
    for shape, note in _decode_shapes(k):
        x = torch.randn(shape, device=device, dtype=weight.dtype, generator=generator)
        checks.append(
            check_kernel(
                f"{label}[N={packed.shape[1]} K={k} (tied lm head)]",
                lambda a, p, s, g=group, ref=reference: ref(a, p, s, g),
                lambda a, p, s, g=group: fn(a, p, s, g),
                args=(x, packed, scale),
                replaces="decode_step",
                note=f"{note}; N={packed.shape[1]} K={k}",
            )
        )
    return tuple(checks)


def int4_head_triton_op_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """The champion's probe against the champion's reference. Only the registration moved."""
    return _head_checks(
        model,
        device=device,
        seed=seed,
        label="visible_int4_head.tiled_gemv_int4_visible",
        fn=tiled_gemv_int4_visible,
    )


def int4_head_torch_dequant_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """This one is checked against *its own expression*, which is the point of it.

    The candidate and the reference here are the same torch program, so layer 1 measures
    only that the installed module reads the weights it was given. The claim under test is
    about speed, and layer 2 — which must reproduce `054`'s 0.9318 and 0.01674 nats — is
    what says the arithmetic survived.
    """
    return _head_checks(
        model,
        device=device,
        seed=seed,
        label="visible_int4_head.torch_dequant_gemv_int4",
        fn=torch_dequant_gemv_int4,
    )


def int4_head_torch_dequant_bf16_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """The champion's site against the bf16 expression, which is a different function.

    `054` and `056` returned layer-2 numbers identical to the digit because they were two
    authors of one function. This slot is **not** quite that experiment -- the matmuls
    accumulate in a different order -- but it is closer than it looks: on CPU the two
    expressions agree to **0.0077 relative at every real site shape** (9216x2560 and
    2560x9216, batch 1 and 32), inside the harness's own 1e-2 rtol, which is about one bf16
    ULP at these magnitudes. So the manifest keeps the fp32 slots' bars rather than
    loosening them, and a large move at layer 2 is the registration bug it would have been.
    CUDA accumulates in fp32 too but blocks the reduction differently, so that figure
    bounds the expression and not the kernel.
    """
    return _head_checks(
        model,
        device=device,
        seed=seed,
        label="visible_int4_head.torch_dequant_gemv_int4_bf16",
        fn=torch_dequant_gemv_int4_bf16,
        reference=_dequant_reference_bf16,
    )
