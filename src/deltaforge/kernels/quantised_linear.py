"""Batch 003 — weight-only quantisation with a fused dequantise-GEMV.

**The one hypothesis in the backlog whose ceiling is above 1.0.** At batch-1 decode every
weight is read once per token and almost no arithmetic is done with it, so the model runs
at the bandwidth roofline. You do not beat a roofline with a better kernel; you beat it by
moving fewer bytes. `docs/roofline.py` puts weights at **91.85%** of per-token traffic and
prints the ceiling directly: **1.85x at int8, 3.21x at int4.**

**Why the compiler structurally cannot take this.** Given `dequant(W_q, s) @ x`, inductor
has no way to keep the dequantised weight off the memory bus: it materialises a full bf16
`W` into global memory and then calls cuBLAS. That *adds* an 8.4 GB write on top of the
8.4 GB read, so the "quantised" version moves more bytes than the bf16 baseline it was
meant to beat. `010-int8-dequant-torch` is that program, written in PyTorch and handed to
`max-autotune`, and it is in the batch as the control that tests this paragraph rather
than asserting it. The kernels below never materialise anything: the int8 or int4 weight
is loaded, converted in registers, and consumed inside the K-loop.

---

## What is in this module

Four module classes, all installed by swapping the `__class__` of existing `nn.Linear`
instances so the reference's bf16 `weight` **Parameter stays registered and shared** —
`cli._assert_parameters_are_shared` walks `named_parameters`, and the quantised copies are
registered as *buffers*, which it does not walk and which do not need a counterpart.

| Class | Weights read | Purpose |
|---|---|---|
| `Bf16GemvLinear` | bf16, unquantised | **Control.** The same GEMV, moving the bytes cuBLAS moves. |
| `Int8DequantLinear` | int8, dequantised by torch | **Control.** What the compiler can express. |
| `Int8GemvLinear` | int8, dequantised in registers | The hypothesis. |
| `Int4GemvLinear` | int4, group-128, in registers | The hypothesis, pushed. |

## Numerics, and why the correctness claim had to change

A quantised candidate is **not bit-comparable to a bf16 reference**, so the layer-2
exact-token gate fails by construction rather than by defect. `docs/HYPOTHESES.md` says
what to do instead, and `harness/correctness.check_distribution` implements it: teacher-force
both models over the reference's own greedy continuation and gate on top-1 agreement and
mean KL, against thresholds registered in `batches.py` *before* the rental.

Layer 1 stays exact, and is the sharper of the two gates for these kernels: each Triton op
is compared against a PyTorch expression computing the identical quantised arithmetic in
the identical order, so anything but a near-ULP agreement is a kernel bug and not a
quantisation error.

## The one place this does not run its own kernel

`GEMV_MAX_ROWS`. A GEMV re-reads the whole weight for every row of `x`, so at the 2048-row
prefill it would read 7 GB per layer and the rental would end inside slot 0. Above the
threshold the op dequantises and calls `F.linear` — **the same quantised numerics, a
different implementation**, and never inside a timed region: `run_interleaved` excludes
every `setup` by design and the prefill is setup. Both `DEFAULT_WORKLOADS` batch sizes (1
and 32) are below the threshold, which `quantised_linear_test.py` pins so it stays true.

That is deliberately not the silent fallback `_triton.require_cuda` forbids. The forbidden
thing is running *the reference* while the harness records it as the candidate; this runs
the candidate's own arithmetic, outside the measurement, and says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._triton import HAS_TRITON, require_cuda, tl, triton

__all__ = [
    "GEMV_MAX_ROWS",
    "TILE_ELEMENTS",
    "INT4_MAX",
    "INT8_MAX",
    "Bf16GemvLinear",
    "Int4GemvLinear",
    "Int8DequantLinear",
    "Int8GemvLinear",
    "QuantisedWeight",
    "bf16_gemv",
    "group_size_for",
    "int4_gemv",
    "int8_gemv",
    "install_bf16_gemv",
    "install_int4_full",
    "install_int8_all_linear",
    "install_int8_dequant",
    "install_int8_full",
    "install_int8_mlp",
    "quantise_int4_grouped",
    "quantise_int8_per_channel",
]


#: Above this many rows of ``x`` the op stops being a GEMV and dequantises instead. See
#: the module docstring: this bound is only ever crossed by the untimed prefill.
GEMV_MAX_ROWS = 64

INT8_MAX = 127
INT4_MAX = 7

#: Elements of the weight each program handles per K-loop iteration. Held roughly constant
#: while the rows-per-program varies, so the tile stays register-resident at every shape.
TILE_ELEMENTS = 4096


def _launch_shape(n: int) -> tuple[int, int, int]:
    """``(BLOCK_N, BLOCK_K, num_warps)`` for an ``N``-wide projection.

    **Rows per program is a parallelism decision, not a tiling one.** A GEMV has no M to
    spread across the card at batch 1, so the only parallelism available is over output
    channels: the grid is ``ceil(N / BLOCK_N)`` programs and nothing else. At 64 rows per
    program `down_proj` — N=2560 — would launch **40 programs onto a 170-SM card**, leaving
    three quarters of the GPU idle on 24% of the model's weight bytes. The kernel would be
    perfectly correct and the hypothesis would lose for a reason that has nothing to do
    with quantisation.

    So: 8 rows per program below N=2048, 16 below N=4096 and 32 above, with ``BLOCK_K``
    widened to keep the tile constant. That puts every projection in this model at 128
    programs or more — **except `in_proj_a` and `in_proj_b`**, which have 32 output channels
    and cannot be spread past 4 programs by any choice of tile. Fixing those needs split-K,
    not a different block size, and they are 0.03% of per-token bytes, so they stay as they
    are and `012`'s rationale says so rather than the number arriving unexplained.

    Fixed rather than autotuned. `triton.autotune` runs its trials on the first call, which
    under CUDA-graph capture is exactly the wrong moment, and every other kernel in this
    package sizes its launch the same deterministic way.

    **This is the one number in the batch nothing on a CPU can validate**, which is why
    `009-gemv-bf16-control` exists: it runs this same geometry over unquantised bf16
    weights, so if the launch shape is wrong the control says so directly instead of the
    error hiding inside a quantisation result.
    """
    block_n = 8 if n <= 2048 else 16 if n <= 4096 else 32
    return block_n, TILE_ELEMENTS // block_n, 4


# ======================================================================================
# Quantisers. Plain torch, so they run in the CPU test suite against `tiny_config`.
# ======================================================================================


@dataclass(frozen=True)
class QuantisedWeight:
    """A quantised weight and its scales, plus the arithmetic that defines them."""

    qweight: Tensor
    scale: Tensor
    bits: int
    group_size: int


#: Rows of a weight quantised at once. The fp32 intermediate is 4x the bf16 weight, and
#: the tied LM head is 248320 x 2560 — 2.5 GB of fp32 per temporary, twice over, on a card
#: that is already holding the reference, its CUDA graphs and the quantised copies. Every
#: quantiser and every dequantise fallback here works a row block at a time for that
#: reason: the arithmetic is per-row, so chunking changes nothing but the peak.
ROW_CHUNK_ELEMENTS = 1 << 24


def _row_chunk(rows: int, cols: int) -> int:
    return max(1, min(rows, ROW_CHUNK_ELEMENTS // max(cols, 1)))


def quantise_int8_per_channel(weight: Tensor) -> QuantisedWeight:
    """Symmetric int8, one scale per output channel.

    Per-channel rather than per-tensor because the output channels of these projections
    differ in magnitude by more than an order of magnitude, and a single tensor-wide scale
    would spend most of the 8-bit range on the largest row. Per-channel costs ``N`` fp32
    scales — 0.01% of the weight — and is what every published weight-only kernel uses.
    """
    w = weight.detach()
    n, k = w.shape
    qweight = torch.empty((n, k), dtype=torch.int8, device=w.device)
    scale = torch.empty((n,), dtype=torch.float32, device=w.device)
    step = _row_chunk(n, k)
    for start in range(0, n, step):
        block = w[start : start + step].float()
        block_scale = block.abs().amax(dim=1) / INT8_MAX
        # A row of exact zeros would divide by zero and produce NaN weights that still
        # look plausible downstream. It cannot happen in a trained checkpoint, which is
        # precisely why nothing would catch it.
        block_scale = torch.where(block_scale > 0, block_scale, torch.ones_like(block_scale))
        qweight[start : start + step] = (
            torch.round(block / block_scale.unsqueeze(1)).clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)
        )
        scale[start : start + step] = block_scale
    return QuantisedWeight(qweight=qweight, scale=scale, bits=8, group_size=k)


def group_size_for(in_features: int) -> int:
    """The int4 group width for a given ``K``.

    128 wherever it fits. The constraint is not only ``group | K``: the packing below
    pairs element ``j`` with element ``j + K/2`` in one byte, so a K-block must lie inside
    one group *in both halves*, which needs ``group | K/2`` as well — hence ``K % 256``.
    Below that the weight is too narrow for grouping to mean anything and the whole half
    becomes one group, which is what `tiny_config` takes.
    """
    if in_features % 2:
        raise ValueError(f"int4 packing needs an even K, got {in_features}")
    group = 128 if in_features % 256 == 0 else in_features // 2
    if group & (group - 1):
        raise ValueError(f"int4 group size {group} for K={in_features} is not a power of two")
    return group


def quantise_int4_grouped(weight: Tensor) -> QuantisedWeight:
    """Symmetric int4 with per-group scales, two values packed per byte.

    **Packing.** Byte ``j`` of row ``n`` holds element ``j`` in its low nibble and element
    ``j + K/2`` in its high nibble, each stored offset by 8 so a nibble is an unsigned
    0-15. Pairing ``j`` with ``j + K/2`` rather than with ``j + 1`` is what lets the kernel
    read one contiguous byte block and consume it against two contiguous slices of ``x``,
    with exactly one scale per half — see `_int4_gemv_kernel`.
    """
    w = weight.detach()
    n, k = w.shape
    group = group_size_for(k)
    half = k // 2
    packed = torch.empty((n, half), dtype=torch.uint8, device=w.device)
    scale = torch.empty((n, k // group), dtype=torch.float32, device=w.device)
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
        packed[start : start + step] = low | (high << 4)
        scale[start : start + step] = block_scale
    return QuantisedWeight(qweight=packed, scale=scale, bits=4, group_size=group)


def dequantise_int8(qweight: Tensor, scale: Tensor) -> Tensor:
    """``W`` in fp32, materialised. The thing the Triton kernels exist not to do."""
    return qweight.float() * scale.unsqueeze(1)


def dequantise_int4(qweight: Tensor, scale: Tensor, *, group_size: int) -> Tensor:
    """Unpack and rescale, in fp32. Mirrors `_int4_gemv_kernel` exactly."""
    n, half = qweight.shape
    k = half * 2
    low = (qweight & 0x0F).to(torch.int16) - 8
    high = (qweight >> 4).to(torch.int16) - 8
    q = torch.cat((low, high), dim=1).float().reshape(n, k // group_size, group_size)
    return (q * scale.unsqueeze(2)).reshape(n, k)


def _dequant_linear(x: Tensor, dequant, n: int, k: int) -> Tensor:
    """``F.linear`` against a dequantised weight, materialised a row block at a time.

    Only ever reached above `GEMV_MAX_ROWS` — the untimed prefill, and the distribution
    gate, which asks for logits at every position. Chunked over ``N`` because the tied LM
    head dequantises to 2.5 GB of fp32 in one piece otherwise, and a correctness gate that
    OOMs is a gate that reports nothing.
    """
    rows = x.shape[0]
    out = torch.empty((rows, n), device=x.device, dtype=torch.float32)
    step = _row_chunk(n, k)
    x32 = x.float()
    for start in range(0, n, step):
        stop = min(start + step, n)
        out[:, start:stop] = F.linear(x32, dequant(start, stop))
    return out


# ======================================================================================
# Triton kernels
# ======================================================================================


if HAS_TRITON:

    @triton.jit
    def _bf16_gemv_kernel(
        X, W, OUT, M, N, K, stride_xm, stride_wn, stride_om, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        """``out[m, n] = sum_k x[m, k] * w[n, k]``, accumulated in fp32.

        The control kernel: it moves exactly the bytes cuBLAS moves, so its ratio isolates
        whether a hand-written GEMV is competitive at all from whether quantisation helps.
        """
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(X + pid_m * stride_xm + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            w = tl.load(
                W + offs_n[:, None] * stride_wn + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        tl.store(OUT + pid_m * stride_om + offs_n, acc.to(OUT.dtype.element_ty), mask=mask_n)

    @triton.jit
    def _int8_gemv_kernel(
        X,
        WQ,
        SCALE,
        OUT,
        M,
        N,
        K,
        stride_xm,
        stride_wn,
        stride_om,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """The same GEMV over int8 weights, rescaled once per output channel at the end.

        The scale is applied *after* the K-reduction rather than per element. It is
        constant along K for a given output channel, so pulling it out of the loop is
        exact, not an approximation — and it is the reason per-channel int8 costs nothing
        in the inner loop.
        """
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(X + pid_m * stride_xm + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            w = tl.load(
                WQ + offs_n[:, None] * stride_wn + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0,
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        scale = tl.load(SCALE + offs_n, mask=mask_n, other=0.0)
        tl.store(OUT + pid_m * stride_om + offs_n, (acc * scale).to(OUT.dtype.element_ty), mask=mask_n)

    @triton.jit
    def _int4_gemv_kernel(
        X,
        PACKED,
        SCALE,
        OUT,
        M,
        N,
        K,
        HALF,
        stride_xm,
        stride_wn,
        stride_sn,
        stride_om,
        BLOCK_N: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        """Group-wise int4, unpacked in registers.

        One iteration reads a ``BLOCK_N x GROUP`` block of packed bytes. Its low nibbles
        are elements ``[j0, j0+GROUP)`` and its high nibbles are ``[j0+HALF, j0+HALF+GROUP)``
        — two contiguous slices of ``x``, and because ``GROUP`` divides both ``K`` and
        ``HALF`` (see `group_size_for`), each slice lies inside exactly one scale group.
        So the whole block costs two scale loads and no per-element scaling.
        """
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for j0 in range(0, HALF, GROUP):
            offs_j = j0 + tl.arange(0, GROUP)
            mask_j = offs_j < HALF
            packed = tl.load(
                PACKED + offs_n[:, None] * stride_wn + offs_j[None, :],
                mask=mask_n[:, None] & mask_j[None, :],
                other=0,
            )
            low = (packed & 0x0F).to(tl.float32) - 8.0
            high = ((packed >> 4) & 0x0F).to(tl.float32) - 8.0

            x_low = tl.load(X + pid_m * stride_xm + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            x_high = tl.load(X + pid_m * stride_xm + HALF + offs_j, mask=mask_j, other=0.0).to(tl.float32)

            g_low = j0 // GROUP
            g_high = (HALF + j0) // GROUP
            s_low = tl.load(SCALE + offs_n * stride_sn + g_low, mask=mask_n, other=0.0)
            s_high = tl.load(SCALE + offs_n * stride_sn + g_high, mask=mask_n, other=0.0)

            acc += s_low * tl.sum(low * x_low[None, :], axis=1)
            acc += s_high * tl.sum(high * x_high[None, :], axis=1)

        tl.store(OUT + pid_m * stride_om + offs_n, acc.to(OUT.dtype.element_ty), mask=mask_n)


# ======================================================================================
# Custom ops. Opaque to dynamo on purpose: inductor may CUDA-graph around them but may not
# decompose them back into the materialise-then-matmul this hypothesis exists to avoid.
# ======================================================================================


def _flatten(x: Tensor, k: int) -> Tensor:
    if x.shape[-1] != k:
        raise ValueError(f"x has {x.shape[-1]} columns, weight expects {k}")
    return x.reshape(-1, k).contiguous()


@torch.library.custom_op("deltaforge::bf16_gemv", mutates_args=())
def bf16_gemv(x: Tensor, weight: Tensor) -> Tensor:
    """``x @ weight.T`` by hand, in the same shape as the quantised kernels."""
    require_cuda(x, "bf16_gemv")
    n, k = weight.shape
    flat = _flatten(x, k)
    rows = flat.shape[0]
    if rows > GEMV_MAX_ROWS:
        out = _dequant_linear(flat, lambda a, b: weight[a:b].float(), n, k)
        return out.to(x.dtype).reshape(*x.shape[:-1], n)
    out = torch.empty((rows, n), device=flat.device, dtype=flat.dtype)
    block_n, block_k, num_warps = _launch_shape(n)
    grid = ((n + block_n - 1) // block_n, rows)
    _bf16_gemv_kernel[grid](
        flat,
        weight,
        out,
        rows,
        n,
        k,
        flat.stride(0),
        weight.stride(0),
        out.stride(0),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out.reshape(*x.shape[:-1], n)


@bf16_gemv.register_fake
def _bf16_gemv_fake(x: Tensor, weight: Tensor) -> Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@torch.library.custom_op("deltaforge::int8_gemv", mutates_args=())
def int8_gemv(x: Tensor, qweight: Tensor, scale: Tensor) -> Tensor:
    """``x @ (qweight * scale[:, None]).T``, never materialising the dequantised weight."""
    require_cuda(x, "int8_gemv")
    n, k = qweight.shape
    flat = _flatten(x, k)
    rows = flat.shape[0]
    if rows > GEMV_MAX_ROWS:
        out = _dequant_linear(flat, lambda a, b: dequantise_int8(qweight[a:b], scale[a:b]), n, k)
        return out.to(x.dtype).reshape(*x.shape[:-1], n)
    out = torch.empty((rows, n), device=flat.device, dtype=flat.dtype)
    block_n, block_k, num_warps = _launch_shape(n)
    grid = ((n + block_n - 1) // block_n, rows)
    _int8_gemv_kernel[grid](
        flat,
        qweight,
        scale,
        out,
        rows,
        n,
        k,
        flat.stride(0),
        qweight.stride(0),
        out.stride(0),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out.reshape(*x.shape[:-1], n)


@int8_gemv.register_fake
def _int8_gemv_fake(x: Tensor, qweight: Tensor, scale: Tensor) -> Tensor:
    return x.new_empty((*x.shape[:-1], qweight.shape[0]))


@torch.library.custom_op("deltaforge::int4_gemv", mutates_args=())
def int4_gemv(x: Tensor, packed: Tensor, scale: Tensor, group_size: int) -> Tensor:
    """Group-wise int4 GEMV. ``packed`` is ``(N, K // 2)``; see `quantise_int4_grouped`."""
    require_cuda(x, "int4_gemv")
    n, half = packed.shape
    k = half * 2
    flat = _flatten(x, k)
    rows = flat.shape[0]
    if rows > GEMV_MAX_ROWS:
        out = _dequant_linear(
            flat,
            lambda a, b: dequantise_int4(packed[a:b], scale[a:b], group_size=group_size),
            n,
            k,
        )
        return out.to(x.dtype).reshape(*x.shape[:-1], n)
    out = torch.empty((rows, n), device=flat.device, dtype=flat.dtype)
    block_n, _, num_warps = _launch_shape(n)
    grid = ((n + block_n - 1) // block_n, rows)
    _int4_gemv_kernel[grid](
        flat,
        packed,
        scale,
        out,
        rows,
        n,
        k,
        half,
        flat.stride(0),
        packed.stride(0),
        scale.stride(0),
        out.stride(0),
        BLOCK_N=block_n,
        GROUP=group_size,
        num_warps=num_warps,
    )
    return out.reshape(*x.shape[:-1], n)


@int4_gemv.register_fake
def _int4_gemv_fake(x: Tensor, packed: Tensor, scale: Tensor, group_size: int) -> Tensor:
    return x.new_empty((*x.shape[:-1], packed.shape[0]))


# ======================================================================================
# Module classes. Installed by `__class__` swap, so the bf16 Parameter stays shared.
# ======================================================================================


class Bf16GemvLinear(nn.Linear):
    """Control: the hand-written GEMV on the reference's own bf16 weight."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = bf16_gemv(x, self.weight)
        return out if self.bias is None else out + self.bias


class Int8GemvLinear(nn.Linear):
    """The hypothesis: int8 weight, dequantised inside the K-loop."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = int8_gemv(x, self.qweight, self.qscale)
        return out if self.bias is None else out + self.bias


class Int8DequantLinear(nn.Linear):
    """Control: the same quantised weight, dequantised the only way PyTorch can say it.

    This is the program `docs/HYPOTHESES.md` claims inductor cannot fuse — it materialises
    a full bf16 ``W`` into global memory and then calls cuBLAS, adding a write the bf16
    baseline never pays. Predicted `loss`, and the claim is the point of measuring it.
    """

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        weight = self.qweight.to(x.dtype) * self.qscale.to(x.dtype).unsqueeze(1)
        return F.linear(x, weight, self.bias)


class Int4GemvLinear(nn.Linear):
    """The hypothesis, pushed: group-wise int4 unpacked in registers."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = int4_gemv(x, self.qweight, self.qscale, self.qgroup)
        return out if self.bias is None else out + self.bias


class Int8GemvLMHead(nn.Module):
    """The tied LM head, as an int8 GEMV.

    The head is 15.1% of weight bytes and has no `nn.Linear` to swap: `tie_word_embeddings`
    makes it `F.linear(h, embed_tokens.weight)` inside `ReferenceModel.project_logits`.
    Installing here replaces that method's callable, not the embedding lookup, which stays
    bf16 — a table lookup reads one row and is not on the bandwidth path.
    """

    def __init__(self, qweight: Tensor, scale: Tensor) -> None:
        super().__init__()
        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("qscale", scale, persistent=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return int8_gemv(hidden_states, self.qweight, self.qscale)


class Int4GemvLMHead(nn.Module):
    """The tied LM head as a group-wise int4 GEMV. See `Int8GemvLMHead`."""

    def __init__(self, packed: Tensor, scale: Tensor, group_size: int) -> None:
        super().__init__()
        self.register_buffer("qweight", packed, persistent=False)
        self.register_buffer("qscale", scale, persistent=False)
        self.group_size = group_size

    def forward(self, hidden_states: Tensor) -> Tensor:
        return int4_gemv(hidden_states, self.qweight, self.qscale, self.group_size)


# ======================================================================================
# Site selection and installation
# ======================================================================================


def _mlp_linears(model) -> list[nn.Linear]:
    out: list[nn.Linear] = []
    for layer in model.layers:
        out.extend((layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj))
    return out


def _layer_linears(model) -> list[nn.Linear]:
    """Every projection inside a decoder layer: MLP, attention and linear-attention.

    `nn.Conv1d` is deliberately absent — the depthwise causal conv is 8192 x 4 weights per
    layer, four parts in a million of the model, and quantising it would add risk to
    nothing.
    """
    return [module for module in model.layers.modules() if isinstance(module, nn.Linear)]


def _quantise_in_place(linears: list[nn.Linear], patched: type, *, bits: int) -> None:
    for linear in linears:
        if bits == 8:
            quantised = quantise_int8_per_channel(linear.weight)
            linear.register_buffer("qweight", quantised.qweight, persistent=False)
            linear.register_buffer("qscale", quantised.scale, persistent=False)
        else:
            quantised = quantise_int4_grouped(linear.weight)
            linear.register_buffer("qweight", quantised.qweight, persistent=False)
            linear.register_buffer("qscale", quantised.scale, persistent=False)
            linear.qgroup = quantised.group_size
        linear.__class__ = patched


_PATCHED: dict[str, type] = {}


def _quantised_head_model_class() -> type:
    """A `ReferenceModel` subclass whose `project_logits` calls the quantised head.

    A class swap rather than an instance attribute holding a bound method. Both work in
    eager; only one is reliably traceable, and a graph break inside the candidate would
    partially decompile it and hand back a ratio that compares two different amounts of
    compilation — which is blocker 16 wearing a different hat.
    """
    if "head" not in _PATCHED:
        from ..reference import ReferenceModel

        class QuantisedHeadModel(ReferenceModel):
            def project_logits(self, hidden_states):
                return self.quantised_lm_head(hidden_states)

        _PATCHED["head"] = QuantisedHeadModel
    return _PATCHED["head"]


def _install_head(model, *, bits: int) -> None:
    """Quantise the tied LM head and route `project_logits` through it."""
    weight = model.lm_head_weight
    if bits == 8:
        quantised = quantise_int8_per_channel(weight)
        head = Int8GemvLMHead(quantised.qweight, quantised.scale)
    else:
        quantised = quantise_int4_grouped(weight)
        head = Int4GemvLMHead(quantised.qweight, quantised.scale, quantised.group_size)
    model.quantised_lm_head = head.to(weight.device)
    model.__class__ = _quantised_head_model_class()


def _guard(model, flag: str) -> bool:
    """True if this install has already run on this model. Installs are idempotent."""
    if getattr(model, flag, False):
        return True
    setattr(model, flag, True)
    return False


def install_bf16_gemv(model, entry=None) -> None:
    if _guard(model, "_deltaforge_bf16_gemv"):
        return
    for linear in _layer_linears(model):
        linear.__class__ = Bf16GemvLinear


def install_int8_dequant(model, entry=None) -> None:
    if _guard(model, "_deltaforge_int8_dequant"):
        return
    _quantise_in_place(_layer_linears(model), Int8DequantLinear, bits=8)


def install_int8_mlp(model, entry=None) -> None:
    if _guard(model, "_deltaforge_int8_mlp"):
        return
    _quantise_in_place(_mlp_linears(model), Int8GemvLinear, bits=8)


def install_int8_all_linear(model, entry=None) -> None:
    if _guard(model, "_deltaforge_int8_all_linear"):
        return
    _quantise_in_place(_layer_linears(model), Int8GemvLinear, bits=8)


def install_int8_full(model, entry=None) -> None:
    if _guard(model, "_deltaforge_int8_full"):
        return
    _quantise_in_place(_layer_linears(model), Int8GemvLinear, bits=8)
    _install_head(model, bits=8)


def install_int4_full(model, entry=None) -> None:
    if _guard(model, "_deltaforge_int4_full"):
        return
    _quantise_in_place(_layer_linears(model), Int4GemvLinear, bits=4)
    _install_head(model, bits=4)


# ======================================================================================
# Layer-1 correctness checks
# ======================================================================================


def _decode_shapes(k: int) -> tuple[tuple[tuple[int, int, int], str], ...]:
    return (
        ((1, 1, k), "headline decode step: batch 1, one token"),
        ((32, 1, k), "secondary decode step: batch 32, one token"),
    )


def _probe_weights(model) -> tuple[tuple[Tensor, str], ...]:
    """Two real weights spanning the K extremes the model actually contains.

    `gate_proj` is the narrow-K, wide-N shape (2560 -> 9216) and `down_proj` the reverse
    (9216 -> 2560). A kernel that is right on one and wrong on the other is the usual
    shape bug, and one probe would miss it.
    """
    mlp = model.layers[0].mlp
    return ((mlp.gate_proj.weight, "gate_proj"), (mlp.down_proj.weight, "down_proj"))


def bf16_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    from ..harness.correctness import check_kernel

    checks = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for weight, label in _probe_weights(model):
        w = weight.detach()
        for shape, note in _decode_shapes(w.shape[1]):
            x = torch.randn(shape, device=device, dtype=w.dtype, generator=generator)
            checks.append(
                check_kernel(
                    f"quantised_linear.bf16_gemv[{label}]",
                    lambda a, b: F.linear(a.float(), b.float()).to(a.dtype),
                    lambda a, b: bf16_gemv(a, b),
                    args=(x, w),
                    replaces="decode_step",
                    note=note,
                )
            )
    return tuple(checks)


def int8_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """The Triton kernel against the identical quantised arithmetic, written in torch.

    The comparison is deliberately *not* against the bf16 reference. That difference is
    the quantisation error, which is the hypothesis rather than a defect, and layer 2
    measures it properly. What layer 1 must catch is a kernel that computes the quantised
    product wrongly, so the reference here dequantises in fp32 and matmuls in fp32 — the
    same numbers in the same order as the kernel.
    """
    from ..harness.correctness import check_kernel

    checks = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for weight, label in _probe_weights(model):
        quantised = quantise_int8_per_channel(weight.detach())
        qw = quantised.qweight.to(device)
        scale = quantised.scale.to(device)
        for shape, note in _decode_shapes(qw.shape[1]):
            x = torch.randn(shape, device=device, dtype=weight.dtype, generator=generator)
            checks.append(
                check_kernel(
                    f"quantised_linear.int8_gemv[{label}]",
                    lambda a, q, s: F.linear(a.float(), dequantise_int8(q, s)).to(a.dtype),
                    lambda a, q, s: int8_gemv(a, q, s),
                    args=(x, qw, scale),
                    replaces="decode_step",
                    note=note,
                )
            )
    return tuple(checks)


def int8_dequant_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """The torch control against the same fp32 dequantised matmul.

    It is not a Triton kernel and cannot be wrong in the way one can, but recording an
    error magnitude for it makes the two int8 slots comparable: if `010` and `012` report
    the same layer-1 error, they really are the same arithmetic run two ways, which is
    what makes their *timings* a clean statement about materialisation.
    """
    from ..harness.correctness import check_kernel

    checks = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for weight, label in _probe_weights(model):
        quantised = quantise_int8_per_channel(weight.detach())
        qw = quantised.qweight.to(device)
        scale = quantised.scale.to(device)
        shape, note = _decode_shapes(qw.shape[1])[0]
        x = torch.randn(shape, device=device, dtype=weight.dtype, generator=generator)
        checks.append(
            check_kernel(
                f"quantised_linear.int8_dequant[{label}]",
                lambda a, q, s: F.linear(a.float(), dequantise_int8(q, s)).to(a.dtype),
                lambda a, q, s: F.linear(a, q.to(a.dtype) * s.to(a.dtype).unsqueeze(1)),
                args=(x, qw, scale),
                replaces="decode_step",
                note=note,
            )
        )
    return tuple(checks)


def int4_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    from ..harness.correctness import check_kernel

    checks = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for weight, label in _probe_weights(model):
        quantised = quantise_int4_grouped(weight.detach())
        packed = quantised.qweight.to(device)
        scale = quantised.scale.to(device)
        group = quantised.group_size
        for shape, note in _decode_shapes(packed.shape[1] * 2):
            x = torch.randn(shape, device=device, dtype=weight.dtype, generator=generator)
            checks.append(
                check_kernel(
                    f"quantised_linear.int4_gemv[{label}]",
                    lambda a, p, s, g=group: F.linear(a.float(), dequantise_int4(p, s, group_size=g)).to(
                        a.dtype
                    ),
                    lambda a, p, s, g=group: int4_gemv(a, p, s, g),
                    args=(x, packed, scale),
                    replaces="decode_step",
                    note=note,
                )
            )
    return tuple(checks)
