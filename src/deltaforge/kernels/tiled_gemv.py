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

__all__ = [
    "GEMV_MAX_ROWS",
    "TiledBf16Linear",
    "install_tiled_bf16",
    "tiled_bf16_correctness_checks",
    "tiled_gemv_bf16",
    "to_k_major",
]

#: Above this many rows of ``x`` the op stops being a GEMV and does a dense matmul. Only
#: ever crossed by the untimed prefill; see the module docstring.
GEMV_MAX_ROWS = 64

#: How many programs are worth launching before splitting K stops paying. An RTX 5090 has
#: 170 SMs; 256 is one full wave with room for the scheduler to hide a tail.
TARGET_PROGRAMS = 256


def to_k_major(weight: Tensor) -> Tensor:
    """``[N, K]`` -> a contiguous ``[K, N]``. Done once at install; free at decode time."""
    return weight.detach().t().contiguous()


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
    def _reduce_partials_kernel(
        PARTIALS, OUT, N, SPLIT_K, stride_pk, stride_pm, stride_om, BLOCK: tl.constexpr
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
    grid = ((n + block_n - 1) // block_n, split_k, rows)
    _tiled_gemv_bf16_kernel[grid](
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

    out = torch.empty((rows, n), device=flat.device, dtype=flat.dtype)
    reduce_block = 256
    _reduce_partials_kernel[((n + reduce_block - 1) // reduce_block, rows)](
        partials,
        out,
        n,
        split_k,
        partials.stride(0),
        partials.stride(1),
        out.stride(0),
        BLOCK=reduce_block,
        num_warps=4,
    )
    return out.reshape(*x.shape[:-1], n)


@tiled_gemv_bf16.register_fake
def _tiled_gemv_bf16_fake(x: Tensor, w_k_major: Tensor) -> Tensor:
    return x.new_empty((*x.shape[:-1], w_k_major.shape[1]))


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


def _layer_linears(model) -> list[nn.Linear]:
    """Every projection inside a decoder layer. See `quantised_linear._layer_linears`."""
    return [module for module in model.layers.modules() if isinstance(module, nn.Linear)]


def _install_k_major(linears: list[nn.Linear], patched: type) -> None:
    for linear in linears:
        linear.register_buffer("w_k_major", to_k_major(linear.weight), persistent=False)
        linear.__class__ = patched


def install_tiled_bf16(model, entry=None) -> None:
    if getattr(model, "_deltaforge_tiled_bf16", False):
        return
    model._deltaforge_tiled_bf16 = True
    _install_k_major(_layer_linears(model), TiledBf16Linear)


# ======================================================================================
# Layer-1 correctness checks
# ======================================================================================


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
    for weight, label in _probe_weights(model, sites="layers"):
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
