"""Interleaved A/B/A/D timing.

This module times **opaque named callables**. It knows nothing about Qwen, Triton, the
kernel registry, or what a "candidate" is — that separation is what lets the timing
arithmetic be tested exhaustively on a machine with no GPU, by injecting a fake timer.

Why interleaved
---------------
Every session rents a different physical GPU, so absolute milliseconds are provenance,
never the score. Worse, within a single session the card drifts: thermals, clock
throttling and noisy neighbours all move on timescales of seconds to minutes. A
sequential "measure A ten times, then B ten times" design attributes that drift to
whichever column happened to run during it.

So every column is timed **once per round**, in a fixed order, and the score is the
median of the per-round ratio. Drift that affects a whole round divides out.

Rounds 1 and 2 are discarded as warmup: that is where `torch.compile` does its
compilation and autotuning, and where CUDA graphs are captured.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

__all__ = [
    "BenchConfig",
    "BenchResult",
    "CudaEventTimer",
    "Timer",
    "WallClockTimer",
    "median",
    "quantile",
    "run_interleaved",
]


class Timer(Protocol):
    """Times a single call, in milliseconds."""

    def time_ms(self, fn: Callable[[], object]) -> float: ...


class CudaEventTimer:
    """CUDA events with an explicit synchronize on both sides of the measured region.

    The synchronize before ``record`` keeps work queued by the previous column out of
    this column's measurement; the one after is what makes ``elapsed_time`` meaningful.
    """

    def __init__(self, device: object | None = None) -> None:
        self.device = device

    def time_ms(self, fn: Callable[[], object]) -> float:
        import torch  # noqa: PLC0415 - keeps this module importable without torch

        torch.cuda.synchronize(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(self.device)
        return float(start.elapsed_time(end))


class WallClockTimer:
    """`perf_counter` fallback for CPU smoke runs. Never used for a reported result."""

    def time_ms(self, fn: Callable[[], object]) -> float:
        from time import perf_counter  # noqa: PLC0415

        start = perf_counter()
        fn()
        return (perf_counter() - start) * 1000.0


@dataclass(frozen=True)
class BenchConfig:
    """R = 7 with the first two rounds discarded is the spec's default methodology."""

    rounds: int = 7
    warmup_rounds: int = 2
    baseline: str = "compiled"
    #: Untimed calls per column before round 1. Zero by default: the discarded warmup
    #: rounds already absorb compilation, and adding hidden warmup would deviate from
    #: the documented methodology.
    untimed_warmup_calls: int = 0
    #: MB moved per decoded token, per column — from ``harness.bytes_model``. Only the
    #: columns named here report an achieved bandwidth; a column with no byte model
    #: reports nothing, because zero would be a measurement.
    bytes_per_token: dict[str, float] = field(default_factory=dict)
    #: Tokens one timed call decodes. Required whenever ``bytes_per_token`` is supplied,
    #: and deliberately not defaulted: a silent 1 here would report a rate 128x too small
    #: that still looks like a measurement.
    decode_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.rounds <= self.warmup_rounds:
            raise ValueError(
                f"rounds ({self.rounds}) must exceed warmup_rounds ({self.warmup_rounds}); "
                "there would be no scoring rounds left"
            )
        if self.warmup_rounds < 0 or self.untimed_warmup_calls < 0:
            raise ValueError("round counts must be non-negative")
        if self.bytes_per_token and not self.decode_tokens:
            raise ValueError(
                "bytes_per_token needs decode_tokens: a per-token byte model is not a rate "
                "until it is told how many tokens one timed call decodes"
            )

    @property
    def scoring_rounds(self) -> int:
        return self.rounds - self.warmup_rounds


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of an empty sequence")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile, spelled out so the number is reproducible.

    Matches numpy's default ``linear`` method; implemented here so the scoring
    arithmetic has no dependency and can be reasoned about directly.
    """
    if not values:
        raise ValueError("quantile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


@dataclass(frozen=True)
class BenchResult:
    """Everything a results record needs, plus the evidence behind it."""

    labels: tuple[str, ...]
    config: BenchConfig
    #: Per-column timings for every round, warmup included, in round order.
    timings_ms: dict[str, list[float]]
    #: The exact order calls were made in, across the whole run. Provenance for the
    #: claim that the measurement really was interleaved.
    call_order: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def scoring_timings_ms(self) -> dict[str, list[float]]:
        skip = self.config.warmup_rounds
        return {label: times[skip:] for label, times in self.timings_ms.items()}

    @property
    def ratios(self) -> dict[str, list[float]]:
        """Per-round ``t_baseline / t_label``: how many times faster ``label`` is.

        The headline score is ``ratios["candidate"]`` with ``baseline="compiled"``.
        """
        scoring = self.scoring_timings_ms
        base = scoring[self.config.baseline]
        out: dict[str, list[float]] = {}
        for label, times in scoring.items():
            if label == self.config.baseline:
                continue
            per_round = []
            for round_idx, (b, t) in enumerate(zip(base, times)):
                if t <= 0.0:
                    raise ValueError(
                        f"column {label!r} recorded a non-positive time "
                        f"({t}) in scoring round {round_idx}; cannot form a ratio"
                    )
                per_round.append(b / t)
            out[label] = per_round
        return out

    @property
    def median_ratio(self) -> dict[str, float]:
        return {label: median(values) for label, values in self.ratios.items()}

    @property
    def iqr_ratio(self) -> dict[str, float]:
        """Interquartile spread of the per-round ratios.

        Promotion requires beating the incumbent by more than this: a margin inside the
        noise band is recorded as *inconclusive*, not as a win.
        """
        return {
            label: quantile(values, 0.75) - quantile(values, 0.25) for label, values in self.ratios.items()
        }

    @property
    def median_ms(self) -> dict[str, float]:
        return {label: median(times) for label, times in self.scoring_timings_ms.items()}

    @property
    def achieved_gbps(self) -> dict[str, float]:
        """Bytes per second each column actually moved, for the columns that declared bytes.

        This is the number batch 003 lacked. A ratio says a kernel lost; GB/s says whether
        it lost on bandwidth or on instruction issue, and those have different fixes. A
        memory-bound kernel holds the same GB/s as it removes bytes; an issue-bound one
        sheds bandwidth as it sheds bytes, which is exactly what 332 -> 141 -> 65 was.

        SI units, matching the vendor peak every result is scored against.
        """
        per_token = self.config.bytes_per_token
        if not per_token:
            return {}
        tokens = self.config.decode_tokens
        out: dict[str, float] = {}
        for label, megabytes in per_token.items():
            times = self.scoring_timings_ms.get(label)
            if not times:
                continue
            # MB/token x tokens / ms is already GB/s: 1e6 bytes over 1e-3 s is 1e9 B/s.
            out[label] = megabytes * tokens / median(times)
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "rounds": self.config.rounds,
            "warmup_rounds": self.config.warmup_rounds,
            "baseline": self.config.baseline,
            "timings_ms": {k: list(v) for k, v in self.timings_ms.items()},
            "scoring_timings_ms": {k: list(v) for k, v in self.scoring_timings_ms.items()},
            "median_ms": self.median_ms,
            "achieved_gbps": self.achieved_gbps,
            "ratios": {k: list(v) for k, v in self.ratios.items()},
            "median_ratio": self.median_ratio,
            "iqr_ratio": self.iqr_ratio,
            "call_order": list(self.call_order),
            "metadata": dict(self.metadata),
        }


def _inference_context():
    """Autograd off for every model call a benchmark makes.

    The benchmark measures inference, so nothing it calls needs a graph. With autograd
    live, inductor compiles the backward as well as the forward, and the fp32 recurrent
    state's in-place update trips autograd's version counter, so the compile fails
    outright. Batch 001 slot 0 spent 3162s there and died with `BackendCompilerFailed`;
    the same model under the correctness gate's `no_grad` took 17.1s and passed.

    Imported lazily, and degrading to a null context, because this module stays
    importable without torch for the CPU smoke path.
    """
    try:
        import torch  # noqa: PLC0415 - keeps this module importable without torch
    except ModuleNotFoundError:
        from contextlib import nullcontext  # noqa: PLC0415

        return nullcontext()
    return torch.no_grad()


def run_interleaved(
    columns: Mapping[str, Callable[[], object]],
    config: BenchConfig | None = None,
    timer: Timer | None = None,
    metadata: Mapping[str, object] | None = None,
    setups: Mapping[str, Callable[[], object]] | None = None,
    progress: Callable[[int, str, float], None] | None = None,
) -> BenchResult:
    """Time every column once per round, in ``columns`` order, for ``config.rounds`` rounds.

    ``columns`` must be ordered (a plain dict is). The baseline named by the config must
    be one of the columns, otherwise there is nothing to form a ratio against.

    ``setups`` maps a column to work that must happen **before each timed call and be
    excluded from it** — restoring a decode cache to a fixed starting state, for example.
    Putting that inside the timed region would add the same constant to every column,
    which does not cancel in a ratio: it drags every ratio toward 1 and silently shrinks
    whatever win is really there.

    ``progress`` is called ``(round_index, label, elapsed_ms)`` as each column finishes.
    It exists because a benchmark that says nothing until it returns cannot be diagnosed
    when it does not: rental 32's slot 0 spent its entire 6980.9s cap inside this function
    and its results record could only say that the cap was hit. Round 0 is where
    compilation happens, so the first line is the one worth watching.
    """
    if not columns:
        raise ValueError("no columns to time")
    config = config or BenchConfig()
    timer = timer or CudaEventTimer()
    labels = tuple(columns)
    if config.baseline not in columns:
        raise ValueError(f"baseline {config.baseline!r} is not among the columns {list(labels)}")
    setups = dict(setups or {})
    unknown = set(setups) - set(labels)
    if unknown:
        raise ValueError(f"setups given for unknown columns: {sorted(unknown)}")
    unknown = set(config.bytes_per_token) - set(labels)
    if unknown:
        raise ValueError(f"bytes_per_token given for unknown columns: {sorted(unknown)}")

    def _setup(label: str) -> None:
        setup = setups.get(label)
        if setup is not None:
            setup()

    timings: dict[str, list[float]] = {label: [] for label in labels}
    call_order: list[str] = []
    # Every model call the benchmark makes -- warmup, setup and timed alike -- runs with
    # autograd off. A setup that builds a decode cache under autograd makes those tensors
    # graph-tracked, which is how the in-place state update becomes an error later.
    with _inference_context():
        for label in labels:
            for _ in range(config.untimed_warmup_calls):
                _setup(label)
                columns[label]()

        for round_index in range(config.rounds):
            for label in labels:
                _setup(label)
                call_order.append(label)
                elapsed = timer.time_ms(columns[label])
                timings[label].append(elapsed)
                if progress is not None:
                    progress(round_index, label, elapsed)

    return BenchResult(
        labels=labels,
        config=config,
        timings_ms=timings,
        call_order=tuple(call_order),
        metadata=dict(metadata or {}),
    )
