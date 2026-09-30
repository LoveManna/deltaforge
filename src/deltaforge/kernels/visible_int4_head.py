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
    "VisibleInt4LMHead",
    "DequantInt4Linear",
    "install_int4_head_torch_dequant",
    "install_int4_head_triton_op",
    "install_int4_mlp_torch_dequant",
    "install_int4_wide_torch_dequant",
    "int4_head_torch_dequant_correctness_checks",
    "int4_head_triton_op_correctness_checks",
    "int4_mlp_torch_dequant_correctness_checks",
    "int4_wide_torch_dequant_correctness_checks",
    "tiled_gemv_int4_visible",
    "torch_dequant_gemv_int4",
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


class DequantInt4Linear(nn.Linear):
    """`torch_dequant_gemv_int4` behind an `nn.Linear`, for the 96 MLP projections.

    ``w_k_major`` and ``qscale`` are **buffers**, not parameters, so
    `cli._assert_parameters_are_shared` still sees only the reference's own weights.
    """

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = torch_dequant_gemv_int4(x, self.w_k_major, self.qscale, self.qgroup)
        return out if self.bias is None else out + self.bias


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


def _torch_dequant_checks(probes, *, device, seed):
    """One `check_kernel` per (shape, decode shape) over ``probes``.

    Shared by the MLP and the wide installers so the two report the same check names and
    the same reference expression. A second copy of this loop is how batch 003 ended up
    with a layer-1 reference whose rounding differed from the implementation it was
    checking, and reported a relative error of 0.45 for correct code.
    """
    from ..harness.correctness import check_kernel  # noqa: PLC0415
    from .quantised_linear import _decode_shapes  # noqa: PLC0415

    checks = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for weight, label in probes:
        quantised = quantise_int4_k_major(weight.detach())
        packed, scale, group = quantised.qweight, quantised.scale, quantised.group_size
        k = packed.shape[0] * 2
        for shape, note in _decode_shapes(k):
            x = torch.randn(shape, device=device, dtype=weight.dtype, generator=generator)
            checks.append(
                check_kernel(
                    f"visible_int4_head.torch_dequant_gemv_int4[{label}]",
                    lambda a, p, s, g=group: (
                        (
                            a.reshape(-1, a.shape[-1]).float()
                            @ dequantise_int4_k_major(p, s, group_size=g).to(a.dtype).float()
                        )
                        .to(a.dtype)
                        .reshape(*a.shape[:-1], p.shape[1])
                    ),
                    lambda a, p, s, g=group: torch_dequant_gemv_int4(a, p, s, g),
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


def int4_wide_torch_dequant_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """Every distinct shape among the 200 wide projections, plus the tied head."""
    from .tiled_gemv import _shape_probes, _wide_linears  # noqa: PLC0415

    probes = _shape_probes(_wide_linears(model), head=model.lm_head_weight)
    return _torch_dequant_checks(probes, device=device, seed=seed)


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


def _head_checks(model, *, device, seed, label, fn):
    from ..harness.correctness import check_kernel  # noqa: PLC0415
    from .quantised_linear import _decode_shapes  # noqa: PLC0415

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
                lambda a, p, s, g=group: (
                    (
                        a.reshape(-1, a.shape[-1]).float()
                        @ dequantise_int4_k_major(p, s, group_size=g).to(a.dtype).float()
                    )
                    .to(a.dtype)
                    .reshape(*a.shape[:-1], p.shape[1])
                ),
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
