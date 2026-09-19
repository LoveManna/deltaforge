"""Batch 004 — a decode GEMV whose inner loop has no cross-lane reduction in it.

Batch 003 established that the ceiling arithmetic is sound and that the kernel could not
reach it. Its GEMV achieved **332 GB/s at bf16, 141 at int8 and 65 at int4** against a
compiled baseline of ~1200 GB/s: as it removed bytes it got *slower*, which is the
signature of a kernel bound by instruction issue rather than by bandwidth. Every ratio in
that batch follows from it.

The cause was one line::

    acc += tl.sum(w * x[None, :], axis=1)     # quantised_linear.py

``tl.sum`` across ``axis=1`` is a **cross-lane reduction**, and it ran once per K-iteration
— 20 times for K=2560 and 72 for K=9216 — where the standard form accumulates a tile and
reduces once. `quantised_linear.py` is kept exactly as it is: its numbers are the control
this module's numbers are read against, and deleting a measured loss would throw away the
comparison that makes a new one mean something.

Three changes, each with a reason batch 003 measured.

**K-major weight storage.** ``W`` is stored ``[K, N]``, transposed once at install time. A
program owns output columns ``[n0, n0 + BLOCK_N)``; in ``[K, N]`` the slice it needs at
each ``k`` is contiguous, so consecutive threads read consecutive addresses. It is also
the layout ``tl.dot`` wants, with no transpose in the loop.

**``tl.dot`` with M padded to 16.** The MMA accumulator carries the partial sums across
K-iterations, so there is no cross-lane reduction in the loop at all. ``x`` goes in row 0
and rows 1-15 are zero; ``tl.sum(acc, axis=0)`` extracts the answer with a single
reduction after the loop. The 16x flop waste is free: arithmetic intensity here is ~2
flop/byte against a machine balance near 150, so this kernel has flops to burn and
bandwidth to save.

**Split-K.** ``in_proj_a`` and ``in_proj_b`` are 32 output channels wide and cannot fill a
170-SM card by output channel at any tile size — and ``tl.dot`` needs ``BLOCK_N >= 16``, so
unlike batch 003's kernel the tile cannot be narrowed to buy parallelism. Partials go to a
``[SPLIT_K, M, N]`` fp32 buffer and a second kernel reduces them, which is deterministic
where ``atomic_add`` is not: a benchmark whose candidate returns different bits on
different runs cannot be gated on tokens at all.

## The one place this does not run its own kernel

`GEMV_MAX_ROWS`, exactly as in `quantised_linear`. A GEMV re-reads the whole weight for
every row of ``x``, so at the 2048-row prefill it would read gigabytes per layer. Above
the threshold the op does a dense ``x @ W_k_major`` — the same arithmetic, a different
implementation, and never inside a timed region: `run_interleaved` excludes every ``setup``
by design and the prefill is setup. That is deliberately not the silent fallback
`_triton.require_cuda` forbids; the forbidden thing is running *the reference* while the
harness records it as the candidate.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ._triton import HAS_TRITON, require_cuda, tl, triton
from .quantised_linear import INT4_MAX, QuantisedWeight, group_size_for

__all__ = [
    "FP8_MAX",
    "GEMV_MAX_ROWS",
    "ROW_CHUNK_ELEMENTS",
    "TiledBf16Linear",
    "TiledFp8Linear",
    "TiledInt4Linear",
    "TiledInt8Linear",
    "dequantise_fp8",
    "dequantise_int4_k_major",
    "install_tiled_bf16",
    "install_tiled_fp8_all_linear",
    "install_tiled_fp8_full",
    "install_tiled_fp8_head",
    "install_tiled_fp8_mlp",
    "install_tiled_int4_full",
    "install_tiled_int4_head",
    "install_tiled_int8_head",
    "install_tiled_int8_all_linear",
    "quantise_fp8_per_channel",
    "quantise_int4_k_major",
    "tiled_bf16_correctness_checks",
    "tiled_gemv_bf16",
    "tiled_gemv_fp8",
    "tiled_gemv_int4",
    "tiled_gemv_int8",
    "to_k_major",
]

#: The largest finite value e4m3 represents. Symmetric per-channel scaling divides by it,
#: exactly as int8 divides by 127.
FP8_MAX = 448.0

#: Above this many rows of ``x`` the op stops being a GEMV and does a dense matmul. Only
#: ever crossed by the untimed prefill; see the module docstring.
GEMV_MAX_ROWS = 64

#: How many programs are worth launching before splitting K stops paying. An RTX 5090 has
#: 170 SMs; 256 is one full wave with room for the scheduler to hide a tail.
TARGET_PROGRAMS = 256


def to_k_major(weight: Tensor) -> Tensor:
    """``[N, K]`` -> a contiguous ``[K, N]``. Done once at install; free at decode time."""
    return weight.detach().t().contiguous()


#: Elements of a weight quantised at once. The fp32 intermediate is 4x the bf16 weight and
#: the tied head is 248320 x 2560 -- 2.5 GB per temporary, on a card already holding the
#: reference and its CUDA graphs. Mirrors `quantised_linear.ROW_CHUNK_ELEMENTS`.
ROW_CHUNK_ELEMENTS = 1 << 24


def _row_chunk(rows: int, cols: int) -> int:
    return max(1, min(rows, ROW_CHUNK_ELEMENTS // max(cols, 1)))


def quantise_fp8_per_channel(weight: Tensor) -> QuantisedWeight:
    """Symmetric e4m3, one scale per output channel, ``[N, K]`` in and ``[N, K]`` out.

    **Why fp8 before int8 this time.** Batch 003 measured the conversion tax precisely:
    int8 cost 1.438x the time of bf16 in the same kernel on the same sites, because
    ``int8 -> fp32`` is an ALU instruction on the critical path. On sm_120 e4m3 converts in
    the MMA pipeline instead, and every finite e4m3 value is exactly representable in
    bf16 — so the conversion this kernel performs is lossless and the comparison between
    the two slots isolates the tax and nothing else.

    Per-channel rather than per-tensor for the reason `quantise_int8_per_channel` gives:
    the output channels of these projections differ in magnitude by more than an order of
    magnitude, and one tensor-wide scale would spend the range on the largest row.
    """
    w = weight.detach()
    n, k = w.shape
    qweight = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty((n,), dtype=torch.float32, device=w.device)
    step = _row_chunk(n, k)
    for start in range(0, n, step):
        block = w[start : start + step].float()
        block_scale = block.abs().amax(dim=1) / FP8_MAX
        # A row of exact zeros would divide by zero and produce NaN weights that still
        # look plausible downstream. It cannot happen in a trained checkpoint, which is
        # precisely why nothing would catch it.
        block_scale = torch.where(block_scale > 0, block_scale, torch.ones_like(block_scale))
        qweight[start : start + step] = (
            (block / block_scale.unsqueeze(1)).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        )
        scale[start : start + step] = block_scale
    return QuantisedWeight(qweight=qweight, scale=scale, bits=8, group_size=k)


def dequantise_fp8(qweight: Tensor, scale: Tensor) -> Tensor:
    """``W`` in fp32, materialised. The thing these kernels exist not to do."""
    return qweight.to(torch.float32) * scale.unsqueeze(1)


def quantise_int4_k_major(weight: Tensor) -> QuantisedWeight:
    """Symmetric int4 with per-group scales along K, packed and transposed for this kernel.

    `quantised_linear.quantise_int4_grouped` packs along a row of the ``[N, K]`` weight.
    That layout is wrong here for the reason the whole module is K-major: the slice a
    program needs at each ``k`` must be contiguous. So byte ``[j, n]`` holds element
    ``[j, n]`` in its low nibble and ``[j + K/2, n]`` in its high nibble — one contiguous
    byte load serving two contiguous slices of ``x``, each inside exactly one scale group,
    which is the same trick transposed.

    ``scale`` is ``[K // group, N]``: unlike int8 and fp8 the scale varies along K, so it
    cannot be pulled out of the loop. It is applied to the **weight tile** before the dot
    rather than to a partial sum, which keeps the accumulator's job unchanged and adds no
    cross-lane reduction — the property this module exists to protect.
    """
    w = weight.detach()
    n, k = w.shape
    group = group_size_for(k)
    half = k // 2
    packed = torch.empty((half, n), dtype=torch.uint8, device=w.device)
    scale = torch.empty((k // group, n), dtype=torch.float32, device=w.device)
    step = _row_chunk(n, k)
    for start in range(0, n, step):
        block = w[start : start + step].float().reshape(-1, k // group, group)
        block_scale = block.abs().amax(dim=2) / INT4_MAX
        block_scale = torch.where(block_scale > 0, block_scale, torch.ones_like(block_scale))
        q = (
            torch.round(block / block_scale.unsqueeze(2))
            .clamp_(-INT4_MAX, INT4_MAX)
            .to(torch.int16)
            .reshape(-1, k)
        )
        low = (q[:, :half] + 8).to(torch.uint8)
        high = (q[:, half:] + 8).to(torch.uint8)
        packed[:, start : start + step] = (low | (high << 4)).t()
        scale[:, start : start + step] = block_scale.t()
    return QuantisedWeight(qweight=packed, scale=scale, bits=4, group_size=group)


def dequantise_int4_k_major(packed: Tensor, scale: Tensor, *, group_size: int) -> Tensor:
    """Unpack and rescale into a ``[K, N]`` fp32 weight. Mirrors `_tiled_gemv_int4_kernel`."""
    half, n = packed.shape
    k = half * 2
    low = (packed & 0x0F).to(torch.int16) - 8
    high = (packed >> 4).to(torch.int16) - 8
    q = torch.cat((low, high), dim=0).float().reshape(k // group_size, group_size, n)
    return (q * scale.unsqueeze(1)).reshape(k, n)


def _launch_shape(n: int, k: int) -> tuple[int, int, int, int]:
    """``(BLOCK_N, BLOCK_K, SPLIT_K, num_warps)``.

    ``tl.dot`` needs ``BLOCK_N >= 16`` and ``BLOCK_K >= 16``, so unlike batch 003's kernel
    the tile cannot be narrowed to buy parallelism. Split-K buys it instead, which is the
    right instrument anyway: ``in_proj_a`` is 32 channels wide and no tile choice reaches a
    full card.
    """
    block_n = 32 if n <= 4096 else 64
    block_k = 64
    programs = -(-n // block_n)
    # Never split further than there are K-blocks to split: a program with no work still
    # costs a partial buffer row and a pass over it in the reduction kernel.
    split_k = max(1, min(8, -(-TARGET_PROGRAMS // max(programs, 1)), k // block_k))
    return block_n, block_k, split_k, 4


# ======================================================================================
# Triton kernels
# ======================================================================================


if HAS_TRITON:

    @triton.jit
    def _tiled_gemv_bf16_kernel(
        X,
        W,
        PARTIALS,
        M,
        N,
        K,
        stride_xm,
        stride_wk,
        stride_pk,
        stride_pm,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        """``partials[s, m, n] = sum_{k in chunk s} x[m, k] * w_k_major[k, n]``, in fp32.

        The whole point of this kernel is what is *not* in the loop. `tl.dot` writes into
        ``acc``, so the partial sums live in the MMA accumulator across every K-iteration
        and the only cross-lane reduction is the ``tl.sum`` after the loop. Batch 003's
        GEMV ran one such reduction per iteration -- 20 times for K=2560, 72 for K=9216 --
        and reached 28% of the baseline's byte rate.
        """
        pid_n, pid_k, pid_m = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        rows = tl.arange(0, 16)
        acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)

        chunk = tl.cdiv(K, SPLIT_K)
        k_lo = pid_k * chunk
        k_hi = tl.minimum(k_lo + chunk, K)
        for k0 in range(k_lo, k_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_hi
            xv = tl.load(X + pid_m * stride_xm + offs_k, mask=mask_k, other=0.0)
            # x in row 0, zeros elsewhere. The 15 wasted rows cost flops this kernel has
            # in abundance and save the reduction it does not.
            xt = tl.where(rows[:, None] == 0, xv[None, :], 0.0).to(X.dtype.element_ty)
            w = tl.load(
                W + offs_k[:, None] * stride_wk + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc = tl.dot(xt, w, acc)

        tl.store(
            PARTIALS + pid_k * stride_pk + pid_m * stride_pm + offs_n,
            tl.sum(acc, axis=0),
            mask=mask_n,
        )

    @triton.jit
    def _tiled_gemv_scaled_kernel(
        X,
        W,
        PARTIALS,
        M,
        N,
        K,
        stride_xm,
        stride_wk,
        stride_pk,
        stride_pm,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        """`_tiled_gemv_bf16_kernel` over an int8 or fp8 weight. One line differs.

        ``tl.load(...).to(tl.bfloat16)`` is the whole change. Both conversions are exact:
        |q| <= 127 fits bf16's 8 significand bits, and every finite e4m3 value is a bf16
        value. So the partial sums here are the same numbers the bf16 kernel produces over
        the dequantised weight, and the per-channel scale is applied once at the end by
        `_reduce_partials_kernel` -- it is constant along K for a given output channel, so
        pulling it out of the loop is exact rather than an approximation.
        """
        pid_n, pid_k, pid_m = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        rows = tl.arange(0, 16)
        acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)

        chunk = tl.cdiv(K, SPLIT_K)
        k_lo = pid_k * chunk
        k_hi = tl.minimum(k_lo + chunk, K)
        for k0 in range(k_lo, k_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_hi
            xv = tl.load(X + pid_m * stride_xm + offs_k, mask=mask_k, other=0.0)
            xt = tl.where(rows[:, None] == 0, xv[None, :], 0.0).to(tl.bfloat16)
            w = tl.load(
                W + offs_k[:, None] * stride_wk + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            ).to(tl.bfloat16)
            acc = tl.dot(xt, w, acc)

        tl.store(
            PARTIALS + pid_k * stride_pk + pid_m * stride_pm + offs_n,
            tl.sum(acc, axis=0),
            mask=mask_n,
        )

    @triton.jit
    def _tiled_gemv_int4_kernel(
        X,
        PACKED,
        SCALE,
        PARTIALS,
        M,
        N,
        HALF,
        stride_xm,
        stride_wk,
        stride_sg,
        stride_pk,
        stride_pm,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        """Group-wise int4, unpacked in registers, K-major.

        One byte block ``PACKED[j0:j0+BLOCK_K, n0:n0+BLOCK_N]`` holds elements
        ``[j0, j0+BLOCK_K)`` in its low nibbles and ``[j0+HALF, j0+HALF+BLOCK_K)`` in its
        high nibbles — two contiguous slices of ``x``, each inside exactly one scale group
        because ``GROUP`` divides both ``K`` and ``HALF`` and ``BLOCK_K`` divides ``GROUP``.

        The scale varies along K, so unlike int8 and fp8 it cannot leave the loop. It is
        applied to the **weight tile** rather than to a partial sum, so the accumulator's
        job is unchanged and no cross-lane reduction enters the loop — which is the one
        property this module exists to protect. The bf16 rounding of the scaled weight is
        what the layer-1 reference rounds to as well; see `tiled_int4_correctness_checks`.
        """
        pid_n, pid_k, pid_m = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        rows = tl.arange(0, 16)
        acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)

        chunk = tl.cdiv(HALF, SPLIT_K)
        j_lo = pid_k * chunk
        j_hi = tl.minimum(j_lo + chunk, HALF)
        for j0 in range(j_lo, j_hi, BLOCK_K):
            offs_j = j0 + tl.arange(0, BLOCK_K)
            mask_j = offs_j < j_hi
            packed = tl.load(
                PACKED + offs_j[:, None] * stride_wk + offs_n[None, :],
                mask=mask_j[:, None] & mask_n[None, :],
                other=0,
            )
            low = (packed & 0x0F).to(tl.float32) - 8.0
            high = ((packed >> 4) & 0x0F).to(tl.float32) - 8.0

            s_low = tl.load(SCALE + (j0 // GROUP) * stride_sg + offs_n, mask=mask_n, other=0.0)
            s_high = tl.load(SCALE + ((HALF + j0) // GROUP) * stride_sg + offs_n, mask=mask_n, other=0.0)

            x_low = tl.load(X + pid_m * stride_xm + offs_j, mask=mask_j, other=0.0)
            x_high = tl.load(X + pid_m * stride_xm + HALF + offs_j, mask=mask_j, other=0.0)
            xt_low = tl.where(rows[:, None] == 0, x_low[None, :], 0.0).to(tl.bfloat16)
            xt_high = tl.where(rows[:, None] == 0, x_high[None, :], 0.0).to(tl.bfloat16)

            acc = tl.dot(xt_low, (low * s_low[None, :]).to(tl.bfloat16), acc)
            acc = tl.dot(xt_high, (high * s_high[None, :]).to(tl.bfloat16), acc)

        tl.store(
            PARTIALS + pid_k * stride_pk + pid_m * stride_pm + offs_n,
            tl.sum(acc, axis=0),
            mask=mask_n,
        )

    @triton.jit
    def _reduce_partials_kernel(
        PARTIALS,
        SCALE,
        OUT,
        N,
        SPLIT_K,
        stride_pk,
        stride_pm,
        stride_om,
        HAS_SCALE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``[SPLIT_K, M, N]`` fp32 -> ``[M, N]`` in the output dtype.

        A separate kernel rather than ``atomic_add`` into the output: atomics would make
        the summation order depend on scheduling, so the candidate would return different
        bits on different runs and could not be gated on tokens at all.
        """
        pid_n, pid_m = tl.program_id(0), tl.program_id(1)
        offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for s in range(SPLIT_K):
            acc += tl.load(PARTIALS + s * stride_pk + pid_m * stride_pm + offs_n, mask=mask_n, other=0.0)
        tl.store(OUT + pid_m * stride_om + offs_n, acc.to(OUT.dtype.element_ty), mask=mask_n)


# ======================================================================================
# Custom ops. Opaque to dynamo on purpose: inductor may CUDA-graph around them but may not
# decompose them back into something it would rather generate.
# ======================================================================================


def _flatten(x: Tensor, k: int) -> Tensor:
    if x.shape[-1] != k:
        raise ValueError(f"x has {x.shape[-1]} columns, weight expects {k}")
    return x.reshape(-1, k).contiguous()


@torch.library.custom_op("deltaforge::tiled_gemv_bf16", mutates_args=())
def tiled_gemv_bf16(x: Tensor, w_k_major: Tensor) -> Tensor:
    """``x @ w_k_major`` where ``w_k_major`` is ``[K, N]``: the reference's weight, transposed."""
    require_cuda(x, "tiled_gemv_bf16")
    k, n = w_k_major.shape
    flat = _flatten(x, k)
    rows = flat.shape[0]
    if rows > GEMV_MAX_ROWS:
        # The untimed prefill. Same arithmetic, a dense implementation, and nothing
        # materialised: `w_k_major` is already the operand `matmul` wants.
        return (flat @ w_k_major).reshape(*x.shape[:-1], n)

    block_n, block_k, split_k, num_warps = _launch_shape(n, k)
    partials = torch.empty((split_k, rows, n), device=flat.device, dtype=torch.float32)
    _tiled_gemv_bf16_kernel[((n + block_n - 1) // block_n, split_k, rows)](
        flat,
        w_k_major,
        partials,
        rows,
        n,
        k,
        flat.stride(0),
        w_k_major.stride(0),
        partials.stride(0),
        partials.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        num_warps=num_warps,
    )
    return _reduce(partials, None, flat.dtype).reshape(*x.shape[:-1], n)


@tiled_gemv_bf16.register_fake
def _tiled_gemv_bf16_fake(x: Tensor, w_k_major: Tensor) -> Tensor:
    return x.new_empty((*x.shape[:-1], w_k_major.shape[1]))


def _reduce(partials: Tensor, scale: Tensor | None, dtype) -> Tensor:
    """``[SPLIT_K, M, N]`` fp32 -> ``[M, N]``, applying a per-channel scale if there is one."""
    split_k, rows, n = partials.shape
    out = torch.empty((rows, n), device=partials.device, dtype=dtype)
    block = 256
    _reduce_partials_kernel[((n + block - 1) // block, rows)](
        partials,
        scale if scale is not None else partials,
        out,
        n,
        split_k,
        partials.stride(0),
        partials.stride(1),
        out.stride(0),
        HAS_SCALE=scale is not None,
        BLOCK=block,
        num_warps=4,
    )
    return out


@torch.library.custom_op("deltaforge::tiled_gemv_fp8", mutates_args=())
def tiled_gemv_fp8(x: Tensor, w_k_major: Tensor, scale: Tensor) -> Tensor:
    """``x @ (w_k_major * scale)`` with ``w_k_major`` an ``[K, N]`` e4m3 weight."""
    require_cuda(x, "tiled_gemv_fp8")
    return _scaled_gemv(x, w_k_major, scale, "fp8")


@tiled_gemv_fp8.register_fake
def _tiled_gemv_fp8_fake(x: Tensor, w_k_major: Tensor, scale: Tensor) -> Tensor:
    return x.new_empty((*x.shape[:-1], w_k_major.shape[1]))


@torch.library.custom_op("deltaforge::tiled_gemv_int8", mutates_args=())
def tiled_gemv_int8(x: Tensor, w_k_major: Tensor, scale: Tensor) -> Tensor:
    """``x @ (w_k_major * scale)`` with ``w_k_major`` an ``[K, N]`` int8 weight."""
    require_cuda(x, "tiled_gemv_int8")
    return _scaled_gemv(x, w_k_major, scale, "int8")


@tiled_gemv_int8.register_fake
def _tiled_gemv_int8_fake(x: Tensor, w_k_major: Tensor, scale: Tensor) -> Tensor:
    return x.new_empty((*x.shape[:-1], w_k_major.shape[1]))


def _scaled_gemv(x: Tensor, w_k_major: Tensor, scale: Tensor, kind: str) -> Tensor:
    k, n = w_k_major.shape
    flat = _flatten(x, k)
    rows = flat.shape[0]
    if rows > GEMV_MAX_ROWS:
        # The untimed prefill, chunked over N: the tied head dequantises to 2.5 GB of fp32
        # in one piece, and a gate that OOMs reports nothing.
        return _dense_scaled(flat, w_k_major, scale, x, n)

    block_n, block_k, split_k, num_warps = _launch_shape(n, k)
    partials = torch.empty((split_k, rows, n), device=flat.device, dtype=torch.float32)
    _tiled_gemv_scaled_kernel[((n + block_n - 1) // block_n, split_k, rows)](
        flat,
        w_k_major,
        partials,
        rows,
        n,
        k,
        flat.stride(0),
        w_k_major.stride(0),
        partials.stride(0),
        partials.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        num_warps=num_warps,
    )
    return _reduce(partials, scale, flat.dtype).reshape(*x.shape[:-1], n)


def _dense_scaled(flat: Tensor, w_k_major: Tensor, scale: Tensor, x: Tensor, n: int) -> Tensor:
    """The prefill path: the same arithmetic, materialised a column block at a time."""
    out = torch.empty((flat.shape[0], n), device=flat.device, dtype=torch.float32)
    step = _row_chunk(n, w_k_major.shape[0])
    x32 = flat.float()
    for start in range(0, n, step):
        stop = min(start + step, n)
        out[:, start:stop] = x32 @ w_k_major[:, start:stop].to(torch.float32)
    return (out * scale.unsqueeze(0)).to(flat.dtype).reshape(*x.shape[:-1], n)


@torch.library.custom_op("deltaforge::tiled_gemv_int4", mutates_args=())
def tiled_gemv_int4(x: Tensor, packed: Tensor, scale: Tensor, group_size: int) -> Tensor:
    """Group-wise int4 GEMV. ``packed`` is ``[K // 2, N]``; see `quantise_int4_k_major`."""
    require_cuda(x, "tiled_gemv_int4")
    half, n = packed.shape
    k = half * 2
    flat = _flatten(x, k)
    rows = flat.shape[0]
    if rows > GEMV_MAX_ROWS:
        dense = dequantise_int4_k_major(packed, scale, group_size=group_size)
        return (flat.float() @ dense).to(flat.dtype).reshape(*x.shape[:-1], n)

    block_n, block_k, split_k, num_warps = _launch_shape(n, k)
    # Every K-block must lie inside one scale group, and every split-K chunk must start on
    # a block boundary. Both follow from BLOCK_K dividing GROUP, so clamp it.
    block_k = min(block_k, group_size)
    split_k = max(1, min(split_k, half // block_k))
    partials = torch.empty((split_k, rows, n), device=flat.device, dtype=torch.float32)
    _tiled_gemv_int4_kernel[((n + block_n - 1) // block_n, split_k, rows)](
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
    )
    return _reduce(partials, None, flat.dtype).reshape(*x.shape[:-1], n)


@tiled_gemv_int4.register_fake
def _tiled_gemv_int4_fake(x: Tensor, packed: Tensor, scale: Tensor, group_size: int) -> Tensor:
    return x.new_empty((*x.shape[:-1], packed.shape[1]))


# ======================================================================================
# Module classes and installation
# ======================================================================================


class TiledBf16Linear(nn.Linear):
    """The tiled GEMV on a K-major copy of the reference's own bf16 weight.

    ``w_k_major`` is a **buffer**, not a parameter: `cli._assert_parameters_are_shared`
    walks `named_parameters` and rejects anything without a counterpart in the reference.
    ``self.weight`` stays registered and shared, and is read for nothing at decode time.
    """

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = tiled_gemv_bf16(x, self.w_k_major)
        return out if self.bias is None else out + self.bias


class TiledFp8Linear(nn.Linear):
    """e4m3 weight, converted to bf16 inside the K-loop and rescaled once at the end."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = tiled_gemv_fp8(x, self.w_k_major, self.qscale)
        return out if self.bias is None else out + self.bias


class TiledInt8Linear(nn.Linear):
    """int8 weight. Identical to `TiledFp8Linear` but for the stored dtype, which is the
    point: batch 003 measured int8 at 1.438x the time of bf16 in the same kernel, and on
    sm_120 e4m3 converts in the MMA pipeline where int8 converts on the ALU. The two slots
    are adjacent in the batch so the difference between them is that tax and nothing else."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = tiled_gemv_int8(x, self.w_k_major, self.qscale)
        return out if self.bias is None else out + self.bias


class TiledInt4Linear(nn.Linear):
    """Group-wise int4, two values per byte, unpacked in registers."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = tiled_gemv_int4(x, self.w_k_major, self.qscale, self.qgroup)
        return out if self.bias is None else out + self.bias


class TiledLMHead(nn.Module):
    """The tied LM head as a tiled GEMV.

    The head is 15.1% of weight bytes and has no `nn.Linear` to swap: `tie_word_embeddings`
    makes it ``F.linear(h, embed_tokens.weight)`` inside `ReferenceModel.project_logits`.
    Installing here replaces that method's callable, not the embedding lookup, which stays
    bf16 — a table lookup reads one row and is not on the bandwidth path.
    """

    def __init__(self, w_k_major: Tensor, scale: Tensor | None, kind: str, group_size: int = 0) -> None:
        super().__init__()
        self.register_buffer("w_k_major", w_k_major, persistent=False)
        if scale is not None:
            self.register_buffer("qscale", scale, persistent=False)
        self.kind = kind
        self.qgroup = group_size

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.kind == "fp8":
            return tiled_gemv_fp8(hidden_states, self.w_k_major, self.qscale)
        if self.kind == "int8":
            return tiled_gemv_int8(hidden_states, self.w_k_major, self.qscale)
        return tiled_gemv_int4(hidden_states, self.w_k_major, self.qscale, self.qgroup)


def _mlp_linears(model) -> list[nn.Linear]:
    out: list[nn.Linear] = []
    for layer in model.layers:
        out.extend((layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj))
    return out


def _layer_linears(model) -> list[nn.Linear]:
    """Every projection inside a decoder layer. See `quantised_linear._layer_linears`."""
    return [module for module in model.layers.modules() if isinstance(module, nn.Linear)]


def _install_k_major(linears: list[nn.Linear], patched: type) -> None:
    for linear in linears:
        linear.register_buffer("w_k_major", to_k_major(linear.weight), persistent=False)
        linear.__class__ = patched


def _quantise_k_major(linears: list[nn.Linear], patched: type, *, kind: str) -> None:
    """Replace each projection's decode path with a K-major quantised copy.

    The quantised weight is a **buffer**, never a parameter: `_assert_parameters_are_shared`
    walks `named_parameters` and would reject one with no counterpart in the reference.
    """
    from .quantised_linear import quantise_int8_per_channel

    for linear in linears:
        if kind == "int4":
            quantised = quantise_int4_k_major(linear.weight)
            linear.register_buffer("w_k_major", quantised.qweight, persistent=False)
            linear.register_buffer("qscale", quantised.scale, persistent=False)
            linear.qgroup = quantised.group_size
        else:
            quantised = (
                quantise_fp8_per_channel(linear.weight)
                if kind == "fp8"
                else quantise_int8_per_channel(linear.weight)
            )
            linear.register_buffer("w_k_major", to_k_major(quantised.qweight), persistent=False)
            linear.register_buffer("qscale", quantised.scale, persistent=False)
        linear.__class__ = patched


_PATCHED: dict[str, type] = {}


def _quantised_head_model_class(base: type | None = None) -> type:
    """A subclass of ``base`` whose `project_logits` calls the tiled head.

    A class swap rather than an instance attribute holding a bound method. Both work in
    eager; only one is reliably traceable, and a graph break inside the candidate would
    partially decompile it and hand back a ratio comparing two different amounts of
    compilation — which is blocker 16 wearing a different hat.

    **Subclassed from whatever the model already is, not from `ReferenceModel`.** Batch 005
    composes this with `static_cache`, which patches the root class too, and a factory
    anchored at `ReferenceModel` would have the second install silently discard the first.
    The candidate would still pass `_build_candidate`'s "did any module class change"
    check, run, and return a plausible ratio for a model missing one of the two kernels it
    claims to hold. Keyed by base, so the cache holds one class per distinct composition.
    """
    if base is None:
        from ..reference import ReferenceModel

        base = ReferenceModel
    key = f"head:{base.__module__}.{base.__qualname__}"
    if key not in _PATCHED:

        class TiledHeadModel(base):  # type: ignore[misc, valid-type]
            def project_logits(self, hidden_states):
                return self.tiled_lm_head(hidden_states)

        _PATCHED[key] = TiledHeadModel
    return _PATCHED[key]


def _install_head(model, *, kind: str) -> None:
    from .quantised_linear import quantise_int8_per_channel

    weight = model.lm_head_weight
    if kind == "int4":
        quantised = quantise_int4_k_major(weight)
        head = TiledLMHead(quantised.qweight, quantised.scale, "int4", quantised.group_size)
    else:
        quantised = quantise_fp8_per_channel(weight) if kind == "fp8" else quantise_int8_per_channel(weight)
        head = TiledLMHead(to_k_major(quantised.qweight), quantised.scale, kind)
    model.tiled_lm_head = head.to(weight.device)
    model.__class__ = _quantised_head_model_class(type(model))


def _guard(model, flag: str) -> bool:
    """True if this install has already run on this model. Installs are idempotent."""
    if getattr(model, flag, False):
        return True
    setattr(model, flag, True)
    return False


def install_tiled_bf16(model, entry=None) -> None:
    if _guard(model, "_deltaforge_tiled_bf16"):
        return
    _install_k_major(_layer_linears(model), TiledBf16Linear)


def install_tiled_fp8_mlp(model, entry=None) -> None:
    if _guard(model, "_deltaforge_tiled_fp8_mlp"):
        return
    _quantise_k_major(_mlp_linears(model), TiledFp8Linear, kind="fp8")


def install_tiled_fp8_all_linear(model, entry=None) -> None:
    if _guard(model, "_deltaforge_tiled_fp8_all_linear"):
        return
    _quantise_k_major(_layer_linears(model), TiledFp8Linear, kind="fp8")


def install_tiled_fp8_full(model, entry=None) -> None:
    if _guard(model, "_deltaforge_tiled_fp8_full"):
        return
    _quantise_k_major(_layer_linears(model), TiledFp8Linear, kind="fp8")
    _install_head(model, kind="fp8")


def install_tiled_int8_all_linear(model, entry=None) -> None:
    if _guard(model, "_deltaforge_tiled_int8_all_linear"):
        return
    _quantise_k_major(_layer_linears(model), TiledInt8Linear, kind="int8")


def install_tiled_int4_full(model, entry=None) -> None:
    if _guard(model, "_deltaforge_tiled_int4_full"):
        return
    _quantise_k_major(_layer_linears(model), TiledInt4Linear, kind="int4")
    _install_head(model, kind="int4")


# --------------------------------------------------------------------------------------
# Batch 005 — the head on its own
# --------------------------------------------------------------------------------------
#
# **Nobody has ever measured a hand-written GEMV on one site.** Batches 003 and 004 both
# installed on all 248 layer projections at once, where the widest is 9216 and the
# narrowest 32, and reported one aggregate byte rate — 319 GB/s, then 228, against a
# compiled baseline at 1177. That number cannot say whether the kernel is slow everywhere
# or slow where there is no parallelism to have: `in_proj_a` is 32 channels wide and gets
# four programs at any tile size.
#
# The tied LM head is the opposite extreme and the largest single weight in the model:
# **248320 x 2560, 1271.40 MB/token, 14.80% of everything the compiled column moves.** At
# BLOCK_N=64 it launches 3880 programs on a 170-SM card, so it is the one site in this
# model where a hand-written GEMV is not grid-starved by construction, and quantising it
# replaces one kernel launch with two rather than 248 with 496.
#
# What it takes to break even is arithmetic rather than hope. The baseline spends
# 1271.40 MB / 1177 GB/s = 1.08 ms/token in that matmul. So:
#
#   int4 (327.7 MB with its group scales) ties at 303 GB/s and wins outright above it;
#   int8 / fp8 (635.7 MB)                 tie at 588 GB/s.
#
# The int4 bar is 1.33x the aggregate rate two rentals have already measured; the 8-bit bar
# is 2.6x. That is why all three run: if they rank int4 > int8 the kernel is bandwidth-bound
# at this site, and if int8 > int4 it is still issue-bound and the nibble unpack is on the
# critical path — which is what batch 003 measured (int4 cost 1.046x int8 *while moving half
# the bytes*). One site, three encodings, and the ordering is the finding either way.


def install_tiled_int8_head(model, entry=None) -> None:
    if _guard(model, "_deltaforge_tiled_int8_head"):
        return
    _install_head(model, kind="int8")


def install_tiled_fp8_head(model, entry=None) -> None:
    if _guard(model, "_deltaforge_tiled_fp8_head"):
        return
    _install_head(model, kind="fp8")


def install_tiled_int4_head(model, entry=None) -> None:
    if _guard(model, "_deltaforge_tiled_int4_head"):
        return
    _install_head(model, kind="int4")


# ======================================================================================
# Layer-1 correctness checks
# ======================================================================================


def _branch_of(n: int) -> tuple[int]:
    """This module's ``BLOCK_N`` for ``N``, as `_probe_weights` wants it.

    ``_launch_shape`` needs a ``K`` it does not use for the tile width, and the probes only
    ever read element 0. Passing `quantised_linear`'s function instead is what made batch
    004's probe labels name tiles this kernel never launched.
    """
    return (_launch_shape(n, 4096)[0],)


def tiled_bf16_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """One check per distinct launch-shape branch, against an fp32 dense product.

    The probes come from `quantised_linear._probe_weights` so every branch this kernel
    launches is covered; the name is prefixed because `quantised_linear` already exports
    `bf16_correctness_checks` for a *different* kernel, and `CHECK_BUILDERS` keys by kernel
    name for exactly this reason.
    """
    from ..harness.correctness import check_kernel
    from .quantised_linear import _decode_shapes, _linear_reference, _probe_weights

    checks = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for weight, label in _probe_weights(model, sites="layers", launch_shape=_branch_of):
        w = weight.detach()
        n, k = w.shape
        for shape, note in _decode_shapes(k):
            x = torch.randn(shape, device=device, dtype=w.dtype, generator=generator)
            checks.append(
                check_kernel(
                    f"tiled_gemv.tiled_gemv_bf16[{label}]",
                    lambda a, b, n=n, k=k: _linear_reference(a, lambda i, j: b[i:j].float(), n, k),
                    lambda a, b: tiled_gemv_bf16(a, to_k_major(b)),
                    args=(x, w),
                    replaces="decode_step",
                    note=note,
                )
            )
    return tuple(checks)


def _scaled_checks(model, *, device, seed, sites, kind, op):
    """Layer 1 for one quantised variant: the kernel against the identical arithmetic.

    Deliberately *not* against the bf16 reference. That difference is the quantisation
    error, which is the hypothesis rather than a defect, and layer 2 measures it properly.
    What layer 1 must catch is a kernel that computes the quantised product wrongly, so the
    reference dequantises and matmuls in fp32 — the same numbers in the same order.
    """
    from ..harness.correctness import check_kernel
    from .quantised_linear import _decode_shapes, _linear_reference, _probe_weights, dequantise_int8

    quantise = quantise_fp8_per_channel if kind == "fp8" else _int8_quantiser()
    dequantise = dequantise_fp8 if kind == "fp8" else dequantise_int8

    checks = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for weight, label in _probe_weights(model, sites=sites, launch_shape=_branch_of):
        quantised = quantise(weight.detach())
        qw, scale = quantised.qweight, quantised.scale
        n, k = qw.shape
        for shape, note in _decode_shapes(k):
            x = torch.randn(shape, device=device, dtype=weight.dtype, generator=generator)
            checks.append(
                check_kernel(
                    f"tiled_gemv.{op}[{label}]",
                    lambda a, q, s, n=n, k=k: _linear_reference(
                        a, lambda i, j: dequantise(q[i:j], s[i:j]), n, k
                    ),
                    # The transpose is free here and never inside a timed region: the
                    # installer does it once, at install time, in the real candidate.
                    (
                        (lambda a, q, s: tiled_gemv_fp8(a, to_k_major(q), s))
                        if kind == "fp8"
                        else (lambda a, q, s: tiled_gemv_int8(a, to_k_major(q), s))
                    ),
                    args=(x, qw, scale),
                    replaces="decode_step",
                    note=note,
                )
            )
    return tuple(checks)


def _int8_quantiser():
    from .quantised_linear import quantise_int8_per_channel

    return quantise_int8_per_channel


def tiled_fp8_mlp_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    return _scaled_checks(model, device=device, seed=seed, sites="mlp", kind="fp8", op="tiled_gemv_fp8")


def tiled_fp8_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    return _scaled_checks(model, device=device, seed=seed, sites="layers", kind="fp8", op="tiled_gemv_fp8")


def tiled_fp8_full_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    return _scaled_checks(model, device=device, seed=seed, sites="full", kind="fp8", op="tiled_gemv_fp8")


def tiled_int8_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    return _scaled_checks(model, device=device, seed=seed, sites="layers", kind="int8", op="tiled_gemv_int8")


def tiled_int4_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0, sites: str = "full"):
    """int4's reference rounds the scaled weight to bf16, because the kernel does.

    A shared reference is only shared if the implementations share their rounding. The
    group scale varies along K, so it is applied to the weight tile inside the loop and the
    product is rounded to bf16 before the dot; an fp32 reference would report a large error
    for a kernel doing exactly what it should, which is how batch 003 failed `010`.
    """
    from ..harness.correctness import check_kernel
    from .quantised_linear import _decode_shapes, _probe_weights

    checks = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for weight, label in _probe_weights(model, sites=sites, launch_shape=_branch_of):
        quantised = quantise_int4_k_major(weight.detach())
        packed, scale, group = quantised.qweight, quantised.scale, quantised.group_size
        n = packed.shape[1]
        k = packed.shape[0] * 2
        for shape, note in _decode_shapes(k):
            x = torch.randn(shape, device=device, dtype=weight.dtype, generator=generator)
            checks.append(
                check_kernel(
                    f"tiled_gemv.tiled_gemv_int4[{label}]",
                    lambda a, p, s, g=group: (
                        (
                            a.reshape(-1, a.shape[-1]).float()
                            @ dequantise_int4_k_major(p, s, group_size=g).to(a.dtype).float()
                        )
                        .to(a.dtype)
                        .reshape(*a.shape[:-1], p.shape[1])
                    ),
                    lambda a, p, s, g=group: tiled_gemv_int4(a, p, s, g),
                    args=(x, packed, scale),
                    replaces="decode_step",
                    note=f"{note}; N={n} K={k}",
                )
            )
    return tuple(checks)


def tiled_int8_head_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    return _scaled_checks(model, device=device, seed=seed, sites="head", kind="int8", op="tiled_gemv_int8")


def tiled_fp8_head_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    return _scaled_checks(model, device=device, seed=seed, sites="head", kind="fp8", op="tiled_gemv_fp8")


def tiled_int4_head_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """int4 on the head alone, against the same bf16-rounded reference `020` used.

    One probe rather than four, and it is the probe that matters: the head is the only
    site whose error reaches the argmax with no further layer to attenuate it.
    """
    return tiled_int4_correctness_checks(model, device=device, dtype=dtype, seed=seed, sites="head")
