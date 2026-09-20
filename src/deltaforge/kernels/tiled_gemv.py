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

import os
import time

import torch
from torch import Tensor, nn

from ._triton import HAS_TRITON, require_cuda, tl, triton
from .quantised_linear import INT4_MAX, QuantisedWeight, group_size_for

__all__ = [
    "FP8_MAX",
    "GEMV_MAX_ROWS",
    "ROW_CHUNK_ELEMENTS",
    "TUNE_BUDGET_SECONDS",
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
    "install_tiled_int4_head_deep",
    "install_tiled_int4_head_tuned",
    "install_tiled_int4_head_wide",
    "install_tiled_int4_mlp",
    "install_tiled_int4_wide",
    "install_tiled_int8_head",
    "install_tiled_int8_all_linear",
    "quantise_fp8_per_channel",
    "quantise_int4_k_major",
    "candidate_launch_shapes",
    "clear_launch_shape",
    "head_shape_key",
    "pin_launch_shape",
    "tiled_bf16_correctness_checks",
    "tiled_gemv_bf16",
    "tiled_gemv_fp8",
    "tiled_gemv_int4",
    "tiled_gemv_int8",
    "to_k_major",
    "tuned_launch_shapes",
]

#: The largest finite value e4m3 represents. Symmetric per-channel scaling divides by it,
#: exactly as int8 divides by 127.
FP8_MAX = 448.0

#: Above this many rows of ``x`` the op stops being a GEMV and does a dense matmul. Only
#: ever crossed by the untimed prefill; see the module docstring.
GEMV_MAX_ROWS = 64

#: How many programs are worth launching before splitting K stops paying. An RTX 5090 has
#: 170 SMs; 256 is one full wave with room for the scheduler to hide a tail.
#:
#: **Batch 005 says one wave is not the right target.** The tied LM head launches 3880
#: programs -- 23 waves -- and reached 656 GB/s at int4, where the same kernel family
#: averaged 228-319 over layer projections that land on 40-320 program instances. One
#: block per SM cannot keep enough loads in flight to cover DRAM latency, whatever the
#: occupancy calculator says. This constant now only seeds `_heuristic_shape`, which is
#: the tuner's first candidate rather than its answer.
TARGET_PROGRAMS = 256

#: Ceiling on ``ceil(N / BLOCK_N) * SPLIT_K`` in the tuner's search space. 8192 is 48
#: waves on a 170-SM card: past that the split-K partial buffer and its reduction pass
#: cost more than the extra parallelism can return.
MAX_PROGRAMS = 8192

#: Triton's own default, made explicit so a tuned config and an untuned one are the same
#: kind of object and `_time_launch` compares like with like.
DEFAULT_NUM_STAGES = 3

#: Wall-clock ceiling on tuning **one** ``(kind, N, K)``. A tuner that can hang a slot is
#: worse than a heuristic; whatever is best when this expires is what gets installed, and
#: the log says the search was cut short. ``DF_TILE_TUNE_BUDGET`` overrides it.
TUNE_BUDGET_SECONDS = float(os.environ.get("DF_TILE_TUNE_BUDGET", "90"))

#: **Off, because rental 42 measured it choosing a tile 2.3x slower than the heuristic.**
#:
#: `tune_launch_shape` times one site at a time on an otherwise idle card. On the head it
#: reported 0.2 ms for 327 MB -- **1639 GB/s, faster than the whole compiled model
#: achieves** -- and picked BLOCK_N=256, which ran at **282 GB/s in the decode step**
#: against the heuristic tile's 656 on rental 40. A 5.6x gap between a micro-benchmark and
#: the same code in place is not noise: the card boosts through a tight loop of one kernel,
#: and in the real step that kernel is one of 508 arriving with a cache and a clock shaped
#: by the 507 around it. Fewer, fatter programs win the first measurement and lose the
#: second.
#:
#: The search space, the rounds and `_implausible` are kept because the machinery is not
#: what was wrong -- the *measurement it ranks on* is. A tuner that times the decode step
#: rather than the site can use all of it. Until one exists this defaults off, so no batch
#: silently inherits a selector a rental has discredited.
#: See `results/batches/006-tile-and-sites/README.md` section 1.
TILE_TUNING_ENABLED = os.environ.get("DF_TILE_TUNE", "0") != "0"

#: Bytes/second above which a tuning measurement is refused as impossible. An HBM-class
#: card tops out near 1.8 TB/s and no GEMV reading from DRAM can exceed it; a number above
#: this means the timer measured something other than the kernel, and acting on it is
#: worse than not tuning. Rental 42's head measurement was 1639 GB/s -- under this bar and
#: still wrong, which is why the bar is a floor on scepticism and not a substitute for it.
IMPLAUSIBLE_GBPS = 2500.0

#: ``(BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)``.
LaunchShape = tuple[int, int, int, int, int]

#: The tile in force for each ``(kind, N, K)``, overriding `_heuristic_shape`.
#:
#: Written by `tune_launch_shape` -- and, since batch 007, by `pin_launch_shape`, because
#: the only two tiles this project has ever measured **in the decode step** disagree by
#: 2.3x and neither was chosen by timing that step. A pin is a tile registered in the
#: manifest before the rental and read back out of the slot record afterwards, which is
#: the one selection method rental 42 did not discredit: it ranks tiles by the ratio they
#: produce in place rather than by a micro-benchmark on an idle card.
#:
#: **It is process-global, so a slot that pins a tile would otherwise change every slot
#: behind it.** `_install_head` clears this key before every head install and the pinning
#: installers set it immediately after, so a slot always runs the tile its manifest names.
#: Without that, `035-int4-head` would re-measure the champion at whatever tile ran last.
_TUNED: dict[tuple[str, int, int], LaunchShape] = {}


def pin_launch_shape(kind: str, n: int, k: int, shape: LaunchShape) -> None:
    """Force ``shape`` for this site, in place of the heuristic or a measured tile."""
    _TUNED[(kind, n, k)] = shape


def clear_launch_shape(kind: str, n: int, k: int) -> None:
    """Drop any tile in force for this site, so `_heuristic_shape` decides again."""
    _TUNED.pop((kind, n, k), None)


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


def _heuristic_shape(n: int, k: int) -> LaunchShape:
    """Batch 004's tile, kept as the default and as the first config the tuner tries.

    ``tl.dot`` needs ``BLOCK_N >= 16`` and ``BLOCK_K >= 16``, so unlike batch 003's kernel
    the tile cannot be narrowed to buy parallelism. Split-K buys it instead, which is the
    right instrument anyway: ``in_proj_a`` is 32 channels wide and no tile choice reaches a
    full card.

    **It was never measured, and batch 005 showed what that cost.** The head won at
    BLOCK_N=64 with 3880 programs; every layer projection lands on 40-320 program instances
    and achieved a third of the head's byte rate. `tune_launch_shape` is the answer to that
    and this function is its baseline: the tuner cannot return anything slower than what
    this chooses, because this is always the first candidate it times.
    """
    block_n = 32 if n <= 4096 else 64
    block_k = 64
    programs = -(-n // block_n)
    # Never split further than there are K-blocks to split: a program with no work still
    # costs a partial buffer row and a pass over it in the reduction kernel.
    split_k = max(1, min(8, -(-TARGET_PROGRAMS // max(programs, 1)), k // block_k))
    return block_n, block_k, split_k, 4, DEFAULT_NUM_STAGES


def _launch_shape(n: int, k: int, kind: str = "bf16") -> LaunchShape:
    """The tile this kernel launches for ``N x K``: measured if it has been, guessed if not.

    Keyed by ``(kind, n, k)`` rather than by module instance, because the 32 ``up_proj``
    sites are one shape and one measurement. `tune_launch_shape` writes the table; nothing
    else does, so a process that never tunes behaves exactly as batch 004's did.
    """
    tuned = _TUNED.get((kind, n, k))
    return tuned if tuned is not None else _heuristic_shape(n, k)


def candidate_launch_shapes(n: int, k: int) -> list[LaunchShape]:
    """Round 1: a **coarse** sweep of the two axes batch 004 never varied by measurement.

    * **Programs.** ``ceil(N / BLOCK_N) * SPLIT_K``. The head launches 3880 of them and
      reached 656 GB/s; the layer projections land on 40-320 and reached ~200-280. The
      heuristic caps ``SPLIT_K`` at 8 and targets 256 programs -- *one wave* on a 170-SM
      card, which is one block per SM and nowhere near enough loads in flight to cover
      DRAM latency.
    * **Transaction width.** In ``[K, N]`` a program's row slice is ``BLOCK_N`` contiguous
      elements. At BLOCK_N=32 that is **64 bytes, half a cache line**, and the heuristic
      picks 32 for every site with ``N <= 4096``: ``down_proj``, ``out_proj``, ``o_proj``,
      ``in_proj_z`` and the k/v projections, about 40% of the layer-projection bytes. It
      narrows the tile to buy programs and pays in DRAM efficiency -- the opposite of the
      trade the head won on.

    Coarse on purpose. Every configuration is a Triton compile before it is a measurement,
    and `tune_launch_shape` has a wall clock; a full cross product would spend the budget
    on the first axis and never reach the second. `refine_split_k` and `refine_pipeline`
    narrow around whatever this finds.
    """
    return _dedupe(
        [_heuristic_shape(n, k)]
        + [
            (block_n, 64, split_k, 4, DEFAULT_NUM_STAGES)
            for block_n in (32, 64, 128, 256)
            for split_k in (1, 4, 16, 64)
            if _tile_is_legal(n, k, block_n, 64, split_k)
        ]
    )


def refine_split_k(best: LaunchShape, n: int, k: int) -> list[LaunchShape]:
    """Round 2: the neighbours of the coarse winner on the same two axes.

    The coarse grid steps ``SPLIT_K`` by 4x, so the winner is only known to within a
    factor of two either side. This is the half-step, and it is where the answer to "how
    many programs does this kernel actually want" gets its significant figures.
    """
    block_n, block_k, split_k, warps, stages = best
    neighbours = [(block_n, block_k, sk, warps, stages) for sk in (split_k // 2, split_k * 2) if sk >= 1]
    neighbours += [(bn, block_k, split_k, warps, stages) for bn in (block_n // 2, block_n * 2) if bn >= 16]
    return [c for c in _dedupe(neighbours) if _tile_is_legal(n, k, c[0], c[1], c[2])]


def refine_pipeline(best: LaunchShape, n: int, k: int) -> list[LaunchShape]:
    """Round 3: warps, ``BLOCK_K`` and pipeline depth, one coordinate at a time.

    These do not change how much parallelism the launch has, only how well each program
    keeps memory in flight, so they are refined last and independently rather than crossed.
    """
    block_n, block_k, split_k, warps, stages = best
    out = [(block_n, block_k, split_k, w, stages) for w in (2, 8) if w != warps]
    out += [(block_n, bk, split_k, warps, stages) for bk in (32, 128) if bk != block_k]
    out += [(block_n, block_k, split_k, warps, st) for st in (2, 5) if st != stages]
    return [c for c in _dedupe(out) if _tile_is_legal(n, k, c[0], c[1], c[2])]


def _tile_is_legal(n: int, k: int, block_n: int, block_k: int, split_k: int) -> bool:
    """Whether a tile is worth compiling at all.

    ``tl.dot`` refuses either dimension below 16. Past `MAX_PROGRAMS` the split-K partial
    buffer and its reduction pass cost more than the parallelism returns. And a split that
    leaves a program fewer than one ``BLOCK_K`` of work has it launch, allocate a partial
    row, store a zero and be read back by the reduction — which is pure overhead however
    the arithmetic is written.
    """
    if block_n < 16 or block_k < 16:
        return False
    if -(-n // block_n) * split_k > MAX_PROGRAMS:
        return False
    return split_k <= -(-k // block_k)


def _dedupe(shapes: list[LaunchShape]) -> list[LaunchShape]:
    seen: set[LaunchShape] = set()
    out: list[LaunchShape] = []
    for shape in shapes:
        if shape not in seen:
            seen.add(shape)
            out.append(shape)
    return out


def _implausible(elapsed_s: float, weight_bytes: int | None) -> bool:
    """Whether a measurement implies a bandwidth no card can deliver.

    Rental 42 is why this exists and also why it is not enough on its own: the head's
    measurement implied **1639 GB/s**, under this bar, and the tile it endorsed ran at 282
    GB/s in the decode step. A guard can refuse an impossible number; it cannot make a
    possible one representative.
    """
    if weight_bytes is None or elapsed_s <= 0:
        return False
    return weight_bytes / elapsed_s / 1e9 > IMPLAUSIBLE_GBPS


def tune_launch_shape(
    kind: str,
    n: int,
    k: int,
    run,
    *,
    budget: float | None = None,
    log=None,
    weight_bytes: int | None = None,
) -> LaunchShape:
    """Time every candidate tile for ``N x K`` on the card in hand and keep the fastest.

    **The point of the whole batch.** We are trying to beat ``max-autotune``, and
    ``max-autotune`` is called that because it measures its tile instead of deriving it
    from a comment about SM counts. Two rentals of hand-written GEMV lost on sites whose
    tile no one had ever timed.

    ``run`` is a zero-argument callable that performs one decode-shaped launch against the
    real quantised weight. It is called with `_TUNED` already set to the config under test,
    so the thing being timed is exactly the code path the benchmark will run -- not a
    reconstruction of it.

    Three guarantees, each of which is a failure this repository has already paid for:

    * **It cannot lose to the heuristic.** `candidate_launch_shapes` puts
      `_heuristic_shape` first, so the worst outcome is batch 004's tile and a few seconds.
    * **It cannot hang a slot.** ``budget`` bounds the wall clock; whatever is best when it
      expires is kept and the log says the search was truncated.
    * **A config that will not compile costs that config.** Triton raises at launch for
      tiles a shape cannot carry, and those are skipped rather than propagated.

    Timed through `triton.testing.do_bench` where it exists, because it flushes L2 between
    iterations. Without that an 11.8 MB int4 ``up_proj`` sits entirely in a 5090's 96 MB L2
    and the tuner would rank tiles on a cache the real decode step has already evicted.

    Three rounds, coarse to fine, so a truncated search still ends somewhere sensible:
    `candidate_launch_shapes` steps the two parallelism axes by 4x, `refine_split_k` takes
    the half-step around whatever won, and `refine_pipeline` varies warps, ``BLOCK_K`` and
    pipeline depth one coordinate at a time. A single flat cross product would spend the
    whole budget inside the first axis.
    """
    key = (kind, n, k)
    cached = _TUNED.get(key)
    if cached is not None:
        return cached
    budget = TUNE_BUDGET_SECONDS if budget is None else budget
    deadline = time.monotonic() + budget

    def measure(shape: LaunchShape) -> float | None:
        _TUNED[key] = shape
        try:
            elapsed = _time_launch(run)
        except Exception:  # noqa: BLE001 - an unlaunchable tile is a skipped candidate
            return None
        return None if _implausible(elapsed, weight_bytes) else elapsed

    best: LaunchShape | None = None
    best_time = float("inf")
    truncated = False
    rounds = 0
    for build in (
        lambda _best: candidate_launch_shapes(n, k),
        lambda current: refine_split_k(current, n, k),
        lambda current: refine_pipeline(current, n, k),
    ):
        if best is None and rounds:
            break  # nothing in the coarse round ran; there is no winner to refine around
        for shape in build(best):
            if time.monotonic() > deadline:
                truncated = True
                break
            elapsed = measure(shape)
            if elapsed is not None and elapsed < best_time:
                best, best_time = shape, elapsed
        rounds += 1
        if truncated:
            break

    chosen = best if best is not None else _heuristic_shape(n, k)
    _TUNED[key] = chosen
    if log is not None:
        log(
            f"[tile] {kind} N={n} K={k}: {chosen} after {rounds} round(s)"
            f" at {best_time * 1e3:.3f} ms"
            f"{' -- SEARCH TRUNCATED BY BUDGET' if truncated else ''}"
        )
    return chosen


def _time_launch(run) -> float:
    """Seconds per call. `do_bench` when Triton is present, CUDA events otherwise."""
    if triton is not None and hasattr(triton, "testing"):
        try:
            # `warmup` and `rep` are **milliseconds** in Triton, not iterations, and the
            # return is a median in milliseconds. Small on purpose: the tuner runs this
            # once per candidate and the compile dominates either way.
            return float(triton.testing.do_bench(run, warmup=5, rep=20)) * 1e-3
        except TypeError:  # pragma: no cover - a do_bench whose signature has moved
            pass
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        run()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1e-3 / 20


def tuned_launch_shapes() -> dict[tuple[str, int, int], LaunchShape]:
    """What the tuner chose, for the slot record. Empty means nothing was tuned."""
    return dict(_TUNED)


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

        chunk = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K) * BLOCK_K
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

        chunk = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K) * BLOCK_K
        k_lo = pid_k * chunk
        k_hi = tl.minimum(k_lo + chunk, K)
        for k0 in range(k_lo, k_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_hi
            xv = tl.load(X + pid_m * stride_xm + offs_k, mask=mask_k, other=0.0)
            xt = tl.where(rows[:, None] == 0, xv[None, :], 0.0).to(tl.bfloat16)
            # `other=0` is an int32 literal. Triton casts it to int8 happily and refuses
            # it for e4m3 -- `cannot cast int32[64, 64] to fp8e4nv` -- so the fp8 slot did
            # not compile at all on rental 40 while the int8 slot did. A float literal is
            # castable to both.
            w = tl.load(
                W + offs_k[:, None] * stride_wk + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
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
        because ``GROUP`` divides both ``K`` and ``HALF``, ``BLOCK_K`` divides ``GROUP``,
        **and every split-K chunk starts on a ``BLOCK_K`` boundary.**

        That last clause is load-bearing and was not true until batch 006. ``chunk`` was
        ``cdiv(HALF, SPLIT_K)``, which splits *elements*: at ``in_proj_a`` -- HALF 1280,
        SPLIT_K 8 -- that is 160, and 160 is not a multiple of 64, so program 1 started at
        ``j0 = 160`` and read **one scale for a block straddling two groups.** Splitting
        *blocks* instead makes the boundary exact by construction:
        ``cdiv(cdiv(HALF, BLOCK_K), SPLIT_K) * BLOCK_K``. A program whose chunk starts past
        ``HALF`` simply runs no iterations and stores its zero, which is what the reduction
        expects.

        It had never fired, because it needs ``SPLIT_K`` large enough that the element
        split lands off a block boundary and batch 004's heuristic never went above 8 on a
        site whose HALF it did not divide. `020-int4-full` was the slot that would have hit
        it, and its precondition declined it in batch 004 and it was not in batch 005 --
        **a declined slot is an unexecuted code path**, for the third time. The tuner makes
        large ``SPLIT_K`` ordinary, so the same bug would have reached every site.

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

        chunk = tl.cdiv(tl.cdiv(HALF, BLOCK_K), SPLIT_K) * BLOCK_K
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
        # The per-channel scale, which this kernel took in its signature and then did not
        # apply. Rental 40 measured the consequence: `023-int8-head` returned layer-1
        # relative error **4511** and 9473 nats at layer 2, which is what an int8 dot
        # product looks like when nobody divides it back down -- |q| <= 127 against weights
        # near 0.05. `quantised_linear`'s epilogue had always done this (line 355); the
        # batch-004 rewrite moved the epilogue here and dropped the multiply, and then the
        # preconditions declined all five slots that would have run it.
        if HAS_SCALE:
            acc = acc * tl.load(SCALE + offs_n, mask=mask_n, other=0.0)
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

    block_n, block_k, split_k, num_warps, num_stages = _launch_shape(n, k, "bf16")
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
        num_stages=num_stages,
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

    block_n, block_k, split_k, num_warps, num_stages = _launch_shape(n, k, kind)
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
        num_stages=num_stages,
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

    block_n, block_k, split_k, num_warps, num_stages = _launch_shape(n, k, "int4")
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
        num_stages=num_stages,
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


def head_shape_key(head: TiledLMHead) -> tuple[str, int, int]:
    """``(kind, N, K)`` for a built head, as `_launch_shape` keys it.

    One function rather than the expression repeated, because `_tune_for` and the pinning
    installers have to agree with the op about which entry they are writing: a key that is
    off by the int4 packing factor would silently pin nothing.
    """
    packed_k = int(head.w_k_major.shape[0]) * (2 if head.kind == "int4" else 1)
    return head.kind, int(head.w_k_major.shape[1]), packed_k


def _install_head(model, *, kind: str, shape: LaunchShape | None = None) -> None:
    """Replace `project_logits` with the tiled head, at ``shape`` or at the heuristic.

    ``shape=None`` **clears** any tile an earlier slot pinned for this site rather than
    leaving it in force. `_TUNED` lives for the life of the process and a batch runs every
    slot in one process, so an uncleared pin would make `035-int4-head` a re-measurement of
    whichever tile ran before it — a candidate that is not the one the manifest names, with
    a plausible ratio beside it. That is blocker 16's shape, one layer down.
    """
    from .quantised_linear import quantise_int8_per_channel

    weight = model.lm_head_weight
    if kind == "int4":
        quantised = quantise_int4_k_major(weight)
        head = TiledLMHead(quantised.qweight, quantised.scale, "int4", quantised.group_size)
    else:
        quantised = quantise_fp8_per_channel(weight) if kind == "fp8" else quantise_int8_per_channel(weight)
        head = TiledLMHead(to_k_major(quantised.qweight), quantised.scale, kind)
    model.tiled_lm_head = head.to(weight.device)
    key = head_shape_key(model.tiled_lm_head)
    if shape is None:
        clear_launch_shape(*key)
    else:
        pin_launch_shape(*key, shape)
    model.__class__ = _quantised_head_model_class(type(model))


#: The two projections per linear-attention layer that no tile can give parallelism to.
#:
#: ``in_proj_a`` and ``in_proj_b`` are **32 output channels wide**. ``tl.dot`` needs
#: ``BLOCK_N >= 16``, so the widest grid available is two programs times whatever split-K
#: can add -- well under one wave on a 170-SM card at any configuration in the search
#: space. They are also worth **7.86 MB/token of 8587.80, 0.09%**, across 48 of the 248
#: layer projections. Excluding them costs nothing measurable and removes the only sites
#: whose slowness cannot be a property of the tile, which is what batches 003 and 004
#: folded into a single aggregate byte rate.
GATE_PROJECTIONS = ("in_proj_a", "in_proj_b")

#: The ``N`` below which a site is a gate on this checkpoint. Not the selection criterion
#: -- `_wide_linears` names the modules structurally, so it behaves on any config -- but
#: the arithmetic behind it, asserted against the real model by
#: `test_the_structural_gate_set_is_the_narrow_one`.
WIDE_MIN_N = 1024


def _gate_linears(model) -> list[nn.Linear]:
    """The `GATE_PROJECTIONS` of every linear-attention layer."""
    return [
        module
        for parent in model.modules()
        for name in GATE_PROJECTIONS
        if isinstance(module := getattr(parent, name, None), nn.Linear)
    ]


def _wide_linears(model) -> list[nn.Linear]:
    """Every layer projection except the gates. See `GATE_PROJECTIONS`."""
    gates = {id(linear) for linear in _gate_linears(model)}
    return [linear for linear in _layer_linears(model) if id(linear) not in gates]


def _tune_for(linears: list[nn.Linear], kind: str, *, head=None, log=None) -> None:
    """Measure a tile for each distinct ``(N, K)`` these sites present.

    Once per shape, not once per site: the 32 ``up_proj`` modules are one measurement and
    one entry in `_TUNED`. Runs at **install** time, which is inside `_build_candidate` and
    therefore outside every timed region -- `run_interleaved` excludes setup by design.

    Silent no-op without CUDA, so the CPU suite exercises the installers without ever
    needing a card, and `DF_TILE_TUNE=0` restores batch 004's untuned behaviour exactly.
    """
    if not TILE_TUNING_ENABLED or not torch.cuda.is_available():
        return
    sites: list[tuple[int, int, object]] = [
        (int(linear.out_features), int(linear.in_features), linear) for linear in linears
    ]
    if head is not None:
        _, head_n, head_k = head_shape_key(head)
        sites.append((head_n, head_k, head))
    seen: set[tuple[int, int]] = set()
    for n, k, module in sites:
        if (n, k) in seen:
            continue
        seen.add((n, k))
        x = torch.randn((1, k), device=module.w_k_major.device, dtype=torch.bfloat16)
        tune_launch_shape(
            kind,
            n,
            k,
            _runner(kind, x, module),
            log=log,
            weight_bytes=module.w_k_major.numel() * module.w_k_major.element_size(),
        )


def _runner(kind: str, x: Tensor, module):
    """One decode-shaped launch of the op this module will actually call."""
    if kind == "int4":
        return lambda: tiled_gemv_int4(x, module.w_k_major, module.qscale, module.qgroup)
    if kind == "fp8":
        return lambda: tiled_gemv_fp8(x, module.w_k_major, module.qscale)
    if kind == "int8":
        return lambda: tiled_gemv_int8(x, module.w_k_major, module.qscale)
    return lambda: tiled_gemv_bf16(x, module.w_k_major)


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


# --------------------------------------------------------------------------------------
# Batch 006 — the same kernel, on the sites whose tile was never measured
# --------------------------------------------------------------------------------------
#
# Batch 005's finding was about the *site*, and it left a second question standing that
# the site-vs-kernel split had hidden: **what was ever wrong with the tile?** `_launch_shape`
# is a heuristic nobody timed. It targets 256 programs -- one wave on a 170-SM card -- and
# the one site this project has won on launches 3880. It also *narrows* BLOCK_N to 32 for
# every site with N <= 4096, which buys programs by halving the contiguous run each program
# reads from 128 bytes to 64: the opposite of the trade the head won on.
#
# So these installers change nothing about the kernel and everything about how it is
# launched. `tune_launch_shape` times every candidate tile on the card in hand, exactly as
# `max-autotune` does for the code we are trying to beat, and the heuristic is its first
# candidate rather than its answer.
#
# What breaks even, in bytes this repository has measured. At the baseline's 1282 GB/s the
# MLP's 4529.85 MB/token costs 3.53 ms of 6.70; group-128 int4 moves 0.2578 of that, so
# the MLP slot **ties at 331 GB/s** and collects the full 1.643x ceiling at 1282. The wide
# set plus the head is 97.83% of per-token bytes and ties at the same 331, for a 3.65x
# ceiling. Batch 004's untuned tile reached ~196-280 GB/s on these sites and batch 005's
# head reached 656 at int4, so the tie point sits squarely between the two numbers this
# project has measured -- which is what makes it worth a rental rather than an argument.


def install_tiled_int4_head_tuned(model, entry=None) -> None:
    """`install_tiled_int4_head` with the head's tile measured instead of guessed.

    The champion collected **70% of its 1.123x ceiling** at BLOCK_N=64, SPLIT_K=1,
    num_warps=4 -- a configuration chosen by a comment about SM counts. The remaining 30%
    is the cheapest slot in this batch and the one whose result is directly comparable to a
    number already on the leaderboard.
    """
    if _guard(model, "_deltaforge_tiled_int4_head_tuned"):
        return
    _install_head(model, kind="int4")
    _tune_for([], "int4", head=model.tiled_lm_head, log=_installer_log())


def install_tiled_int4_mlp(model, entry=None) -> None:
    """Group-128 int4 on the 96 MLP projections: **52.75% of what the compiled column moves.**

    The largest single block of bytes in the model, and every one of its sites is wide:
    ``gate_proj`` and ``up_proj`` are 9216 and ``down_proj`` 2560, against ``in_proj_a``'s
    32. It is the cleanest test of whether the tile was the problem, because it is one
    mechanism on one homogeneous group of sites.
    """
    if _guard(model, "_deltaforge_tiled_int4_mlp"):
        return
    linears = _mlp_linears(model)
    _quantise_k_major(linears, TiledInt4Linear, kind="int4")
    _tune_for(linears, "int4", log=_installer_log())


def install_tiled_int4_wide(model, entry=None) -> None:
    """int4 on every layer projection with ``N >= WIDE_MIN_N``, **and the tied head**.

    200 sites plus the head: **97.83% of what the compiled column moves.** The 48 excluded
    sites are ``in_proj_a`` and ``in_proj_b``, 32 channels wide and worth 0.05% of
    per-token traffic between them. Batches 003 and 004 installed there too and reported
    one aggregate byte rate over all 248, which is the measurement batch 005 showed cannot
    separate a slow kernel from a starved grid.

    **The head is inside this installer rather than composed beside it**, because the
    registry allows one champion per replaceable operation and both would claim
    ``decode_step``. That invariant is the right one -- two kernels claiming the same
    boundary makes "what does `model.py` assemble?" ambiguous -- so the composition
    happens here, where it is one named kernel with one set of layer-1 probes, exactly as
    `tiled_int4_full` already does for all 248 sites.
    """
    if _guard(model, "_deltaforge_tiled_int4_wide"):
        return
    linears = _wide_linears(model)
    _quantise_k_major(linears, TiledInt4Linear, kind="int4")
    _install_head(model, kind="int4")
    _tune_for(linears, "int4", head=model.tiled_lm_head, log=_installer_log())


# --------------------------------------------------------------------------------------
# Batch 007 — the champion's tile, chosen by the decode step instead of by a loop
# --------------------------------------------------------------------------------------
#
# Two tiles have ever been measured **in place** on the tied head, and they disagree by
# 2.3x: BLOCK_N=64 ran at 656 GB/s on rental 40 and BLOCK_N=256 at 282 on rental 42. Every
# other tile in this project's history was ranked by `tune_launch_shape`, which times one
# site in a loop on an idle card and reported 1639 GB/s for the tile that ran at 282.
#
# So the instrument is the slot, not the tuner. A pinned tile is registered in
# `batches.py` before the rental like any other prediction, installed with no measurement
# on the box at all, and scored by the ratio the whole decode step returns. Three minutes
# a point, against 12-15 seconds of micro-benchmark whose ranking inverted.
#
# **What the two in-place points say, and why these two pins.** The kernel materialises a
# ``(BLOCK_K, BLOCK_N)`` fp32 weight tile in registers before each `tl.dot`. At BLOCK_N=256
# and BLOCK_K=64 that is 64 KB of fp32 per program — 128 registers per thread at 4 warps,
# on top of a ``(16, 256)`` accumulator — which is past what an SM can hold and is the
# ranked suspect for 282 GB/s. Doubling the warps with the width keeps per-thread pressure
# where the champion has it:
#
#   `tiled_int4_head_wide`  BLOCK_N 128, num_warps 8 — 1940 programs, 128 contiguous bytes
#                           per row read, per-thread tile identical to the champion's.
#   `tiled_int4_head_deep`  BLOCK_N 64, num_stages 5 — the champion's tile with four loads
#                           in flight instead of two. If 656 GB/s of 1792 is latency rather
#                           than pressure, this is the axis that moves it and nothing in
#                           the search space rental 42 ran varied it independently.
#
# Neither changes an arithmetic operation, so both must reproduce `022`'s correctness
# numbers exactly — 0.9318 agreement, 0.01674 nats. A different number there is a tiling
# bug, not a quantisation effect, and the gate is set to catch it rather than to pass it.

#: BLOCK_N 128 at 8 warps: twice the champion's width, twice its warps, same per-thread
#: register pressure, and a full 128-byte contiguous run per row of the packed weight.
HEAD_WIDE_SHAPE: LaunchShape = (128, 64, 1, 8, DEFAULT_NUM_STAGES)

#: The champion's tile with a deeper software pipeline. `DEFAULT_NUM_STAGES` is 3, which
#: keeps two loads in flight; 5 keeps four, at the cost of shared memory rather than
#: registers.
HEAD_DEEP_SHAPE: LaunchShape = (64, 64, 1, 4, 5)


def install_tiled_int4_head_wide(model, entry=None) -> None:
    """`install_tiled_int4_head` at `HEAD_WIDE_SHAPE`. Same site, same bytes, same math."""
    if _guard(model, "_deltaforge_tiled_int4_head_wide"):
        return
    _install_head(model, kind="int4", shape=HEAD_WIDE_SHAPE)


def install_tiled_int4_head_deep(model, entry=None) -> None:
    """`install_tiled_int4_head` at `HEAD_DEEP_SHAPE`. Same site, same bytes, same math."""
    if _guard(model, "_deltaforge_tiled_int4_head_deep"):
        return
    _install_head(model, kind="int4", shape=HEAD_DEEP_SHAPE)


def _installer_log():
    """Print the tuner's choices. They are the slot's finding as much as its ratio is."""
    return print


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


def _shape_probes(linears, *, head=None) -> tuple[tuple[Tensor, str], ...]:
    """One real weight per distinct ``(N, K)`` among these sites.

    `quantised_linear._probe_weights` groups by **BLOCK_N branch**, which was the right
    key while the tile came from a branch on ``N``. `tune_launch_shape` keys on
    ``(kind, N, K)`` and can hand two sites with the same BLOCK_N completely different
    tiles, so a branch-grouped probe would leave real launch configurations uncovered —
    the exact gap `_probe_weights` was written to close, moved one level down.

    There are six distinct shapes among the wide layer projections, so covering every one
    of them costs a few seconds and needs no argument about which are representative.
    """
    by_shape: dict[tuple[int, int], Tensor] = {}
    for linear in linears:
        by_shape.setdefault((int(linear.out_features), int(linear.in_features)), linear.weight)
    probes = [(weight, f"N={n} K={k}") for (n, k), weight in sorted(by_shape.items())]
    if head is not None:
        probes.append((head, f"N={head.shape[0]} K={head.shape[1]} (tied lm head)"))
    return tuple(probes)


def tiled_int4_correctness_checks(
    model, *, device="cuda", dtype=None, seed: int = 0, sites: str = "full", probes=None
):
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
    if probes is None:
        probes = _probe_weights(model, sites=sites, launch_shape=_branch_of)
    for weight, label in probes:
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


def tiled_int4_head_tuned_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """The same probe as the champion's. The tile changed; the arithmetic did not.

    Registered separately rather than aliased because `CHECK_BUILDERS` keys by kernel name
    and a kernel with no entry of its own is checked by nothing — which is the defect
    `docs/BATCHES.md` records as keying a check table by the operation instead.
    """
    return tiled_int4_correctness_checks(model, device=device, dtype=dtype, seed=seed, sites="head")


def tiled_int4_head_wide_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """The champion's probe. `HEAD_WIDE_SHAPE` moves the launch geometry and nothing else.

    Its own entry rather than an alias of the champion's, for the reason
    `tiled_int4_head_tuned_correctness_checks` gives: `CHECK_BUILDERS` keys by kernel name
    and a kernel with no entry of its own is checked by nothing.
    """
    return tiled_int4_correctness_checks(model, device=device, dtype=dtype, seed=seed, sites="head")


def tiled_int4_head_deep_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """The champion's probe at `HEAD_DEEP_SHAPE`. See `tiled_int4_head_wide_correctness_checks`."""
    return tiled_int4_correctness_checks(model, device=device, dtype=dtype, seed=seed, sites="head")


def tiled_int4_mlp_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """Every distinct shape among the 96 MLP projections: 9216x2560 and 2560x9216."""
    return tiled_int4_correctness_checks(
        model, device=device, dtype=dtype, seed=seed, probes=_shape_probes(_mlp_linears(model))
    )


def tiled_int4_wide_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """Every distinct shape this installer touches: six layer shapes and the tied head.

    Seven probes rather than the three a BLOCK_N grouping would give, and the extra four
    cost a few seconds. `tune_launch_shape` keys on ``(kind, N, K)`` and can hand two sites
    with the same BLOCK_N different tiles, so grouping by branch would leave real launch
    configurations unexercised -- the gap `_probe_weights` exists to close, one level down.

    ``in_proj_a`` and ``in_proj_b`` are absent because the installer leaves them in bf16,
    and a check that reports on a projection its hypothesis left alone is reporting on the
    reference.
    """
    return tiled_int4_correctness_checks(
        model,
        device=device,
        dtype=dtype,
        seed=seed,
        probes=_shape_probes(_wide_linears(model), head=model.lm_head_weight),
    )
