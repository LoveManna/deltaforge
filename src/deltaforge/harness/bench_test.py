"""The bench is the one component that must be tested hard without a GPU.

Everything here uses an injected timer with scripted return values, so the interleaving
order, the warmup discard and the median-ratio arithmetic are checked as exact facts
rather than inferred from noisy wall-clock numbers.
"""

from __future__ import annotations

import pytest

from .bench import BenchConfig, BenchResult, median, quantile, run_interleaved


class ScriptedTimer:
    """Returns pre-programmed durations and records what it was asked to time."""

    def __init__(self, durations: dict[str, list[float]]):
        self.durations = {k: list(v) for k, v in durations.items()}
        self.calls: list[str] = []
        self._labels: dict[int, str] = {}

    def bind(self, columns: dict[str, object]) -> dict[str, object]:
        for label, fn in columns.items():
            self._labels[id(fn)] = label
        return columns

    def time_ms(self, fn) -> float:
        label = self._labels[id(fn)]
        self.calls.append(label)
        fn()  # a real timer runs the work; so does this one
        return self.durations[label].pop(0)


def make_columns(labels):
    """Distinct callables, one per label, that record their own invocation order."""
    log: list[str] = []
    columns = {}
    for label in labels:
        columns[label] = (lambda name: lambda: log.append(name))(label)
    return columns, log


LABELS = ("eager", "compiled", "compiled_nocudagraphs", "candidate")


def build(durations, config=None):
    columns, log = make_columns(LABELS)
    timer = ScriptedTimer(durations)
    timer.bind(columns)
    result = run_interleaved(columns, config or BenchConfig(), timer=timer)
    return result, timer, log


def constant_durations(per_label: dict[str, float], rounds: int = 7):
    return {label: [value] * rounds for label, value in per_label.items()}


# -- interleaving ---------------------------------------------------------------------


def test_every_column_is_timed_once_per_round_in_declaration_order():
    result, timer, log = build(constant_durations(dict.fromkeys(LABELS, 1.0)))

    assert timer.calls == list(LABELS) * 7
    assert log == list(LABELS) * 7
    assert result.call_order == tuple(LABELS) * 7


def test_interleaving_is_not_grouped_by_column():
    """The failure this guards against: 'measure A seven times, then B seven times'."""
    _result, timer, _log = build(constant_durations(dict.fromkeys(LABELS, 1.0)))

    grouped = [label for label in LABELS for _ in range(7)]
    assert timer.calls != grouped
    # Consecutive calls are never the same column.
    assert all(a != b for a, b in zip(timer.calls, timer.calls[1:]))


def test_column_declaration_order_is_preserved():
    reordered = ("candidate", "eager", "compiled", "compiled_nocudagraphs")
    columns, _log = make_columns(reordered)
    timer = ScriptedTimer(constant_durations(dict.fromkeys(reordered, 1.0)))
    timer.bind(columns)

    result = run_interleaved(columns, BenchConfig(), timer=timer)

    assert result.labels == reordered
    assert timer.calls[:4] == list(reordered)


# -- warmup discard -------------------------------------------------------------------


def test_first_two_rounds_are_discarded_from_scoring():
    durations = {
        # Rounds 1-2 are wildly slow, as they are in reality while torch.compile
        # compiles and CUDA graphs are captured.
        "eager": [900.0, 800.0] + [10.0] * 5,
        "compiled": [900.0, 800.0] + [8.0] * 5,
        "compiled_nocudagraphs": [900.0, 800.0] + [9.0] * 5,
        "candidate": [900.0, 800.0] + [4.0] * 5,
    }
    result, _timer, _log = build(durations)

    assert len(result.timings_ms["candidate"]) == 7
    assert result.scoring_timings_ms["candidate"] == [4.0] * 5
    # The 900/800 ms warmup rounds would have dragged the ratio to ~1.0 if counted.
    assert result.median_ratio["candidate"] == pytest.approx(2.0)


def test_warmup_rounds_are_still_recorded_as_evidence():
    durations = {label: [99.0, 88.0] + [1.0] * 5 for label in LABELS}
    result, _timer, _log = build(durations)

    assert result.timings_ms["eager"][:2] == [99.0, 88.0]
    assert result.to_dict()["timings_ms"]["eager"][:2] == [99.0, 88.0]


def test_config_rejects_having_no_scoring_rounds():
    with pytest.raises(ValueError, match="no scoring rounds"):
        BenchConfig(rounds=2, warmup_rounds=2)
    with pytest.raises(ValueError, match="no scoring rounds"):
        BenchConfig(rounds=1, warmup_rounds=3)


def test_scoring_rounds_property():
    assert BenchConfig(rounds=7, warmup_rounds=2).scoring_rounds == 5


# -- the score ------------------------------------------------------------------------


def test_score_is_median_of_per_round_ratios_not_ratio_of_medians():
    """These differ, and the spec says median of ratios. This pins which one we compute."""
    durations = {
        "eager": [1.0] * 7,
        "compiled": [0.0, 0.0, 10.0, 100.0, 10.0, 100.0, 10.0],
        "compiled_nocudagraphs": [1.0] * 7,
        "candidate": [0.0, 0.0, 10.0, 10.0, 5.0, 50.0, 1.0],
    }
    # Warmup rounds are discarded, so the zeros never reach the arithmetic.
    result, _timer, _log = build(durations)

    per_round = [10 / 10, 100 / 10, 10 / 5, 100 / 50, 10 / 1]
    assert result.ratios["candidate"] == pytest.approx(per_round)
    assert result.median_ratio["candidate"] == pytest.approx(median(per_round))
    assert result.median_ratio["candidate"] == pytest.approx(2.0)

    ratio_of_medians = median([10.0, 100.0, 10.0, 100.0, 10.0]) / median([10.0, 10.0, 5.0, 50.0, 1.0])
    assert ratio_of_medians == pytest.approx(1.0)
    assert result.median_ratio["candidate"] != pytest.approx(ratio_of_medians)


def test_ratio_is_baseline_over_column_so_faster_is_greater_than_one():
    durations = constant_durations(
        {
            "eager": 20.0,
            "compiled": 10.0,
            "compiled_nocudagraphs": 12.0,
            "candidate": 5.0,
        }
    )
    result, _timer, _log = build(durations)

    assert result.median_ratio["candidate"] == pytest.approx(2.0)
    assert result.median_ratio["eager"] == pytest.approx(0.5)
    assert result.median_ratio["compiled_nocudagraphs"] == pytest.approx(10 / 12)
    assert "compiled" not in result.median_ratio


def test_baseline_column_is_configurable():
    durations = constant_durations({"eager": 20.0, "compiled": 10.0, "candidate": 5.0})
    columns, _log = make_columns(("eager", "compiled", "candidate"))
    timer = ScriptedTimer(durations)
    timer.bind(columns)

    result = run_interleaved(columns, BenchConfig(baseline="eager"), timer=timer)

    assert result.median_ratio["candidate"] == pytest.approx(4.0)


def test_missing_baseline_column_is_rejected():
    columns, _log = make_columns(("eager", "candidate"))
    with pytest.raises(ValueError, match="baseline 'compiled' is not among"):
        run_interleaved(columns, BenchConfig(), timer=ScriptedTimer({}))


def test_zero_timing_is_an_error_not_an_infinite_speedup():
    durations = constant_durations(
        {"eager": 1.0, "compiled": 1.0, "compiled_nocudagraphs": 1.0, "candidate": 1.0}
    )
    durations["candidate"][4] = 0.0
    result, _timer, _log = build(durations)

    with pytest.raises(ValueError, match="non-positive time"):
        _ = result.median_ratio


def test_iqr_of_ratios_measures_the_noise_band():
    durations = {
        "eager": [1.0] * 7,
        "compiled": [1.0] * 7,
        "compiled_nocudagraphs": [1.0] * 7,
        "candidate": [1.0, 1.0] + [1 / 1.0, 1 / 2.0, 1 / 3.0, 1 / 4.0, 1 / 5.0],
    }
    result, _timer, _log = build(durations)

    assert result.ratios["candidate"] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.median_ratio["candidate"] == pytest.approx(3.0)
    assert result.iqr_ratio["candidate"] == pytest.approx(2.0)


def test_empty_columns_rejected():
    with pytest.raises(ValueError, match="no columns"):
        run_interleaved({}, BenchConfig(), timer=ScriptedTimer({}))


# -- untimed setup --------------------------------------------------------------------


def test_setup_runs_before_each_timed_call_and_is_not_timed():
    order: list[str] = []
    columns = {
        "compiled": lambda: order.append("run:compiled"),
        "candidate": lambda: order.append("run:candidate"),
    }
    setups = {
        "compiled": lambda: order.append("setup:compiled"),
        "candidate": lambda: order.append("setup:candidate"),
    }
    timer = ScriptedTimer({"compiled": [10.0] * 4, "candidate": [5.0] * 4})
    timer.bind(columns)

    result = run_interleaved(columns, BenchConfig(rounds=4, warmup_rounds=1), timer=timer, setups=setups)

    assert order[:4] == [
        "setup:compiled",
        "run:compiled",
        "setup:candidate",
        "run:candidate",
    ]
    # The timer only ever saw the run callables, never the setups.
    assert timer.calls == ["compiled", "candidate"] * 4
    assert result.median_ratio["candidate"] == pytest.approx(2.0)


def test_setup_for_unknown_column_is_rejected():
    columns, _log = make_columns(("compiled", "candidate"))
    with pytest.raises(ValueError, match="unknown columns"):
        run_interleaved(
            columns,
            BenchConfig(),
            timer=ScriptedTimer({}),
            setups={"nonexistent": lambda: None},
        )


def test_untimed_warmup_calls_run_before_any_round():
    calls: list[str] = []
    columns = {"compiled": lambda: calls.append("compiled"), "candidate": lambda: calls.append("candidate")}
    timer = ScriptedTimer({"compiled": [1.0] * 3, "candidate": [1.0] * 3})
    timer.bind(columns)

    run_interleaved(
        columns,
        BenchConfig(rounds=3, warmup_rounds=1, untimed_warmup_calls=2),
        timer=timer,
    )

    # Two untimed warmups per column, then the interleaved rounds.
    assert calls[:4] == ["compiled", "compiled", "candidate", "candidate"]
    assert len(timer.calls) == 6


# -- statistics helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0], 1.0),
        ([1.0, 3.0], 2.0),
        ([3.0, 1.0, 2.0], 2.0),
        ([4.0, 1.0, 3.0, 2.0], 2.5),
    ],
)
def test_median(values, expected):
    assert median(values) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("q", "expected"),
    [(0.0, 1.0), (0.25, 2.0), (0.5, 3.0), (0.75, 4.0), (1.0, 5.0)],
)
def test_quantile_matches_linear_interpolation(q, expected):
    assert quantile([1.0, 2.0, 3.0, 4.0, 5.0], q) == pytest.approx(expected)


def test_quantile_interpolates_between_samples():
    assert quantile([0.0, 10.0], 0.3) == pytest.approx(3.0)


def test_median_and_quantile_reject_empty():
    with pytest.raises(ValueError):
        median([])
    with pytest.raises(ValueError):
        quantile([], 0.5)


def test_quantile_rejects_out_of_range_q():
    with pytest.raises(ValueError, match=r"q must be in \[0, 1\]"):
        quantile([1.0], 1.5)


# -- serialisation --------------------------------------------------------------------


def test_to_dict_carries_raw_per_round_timings_for_provenance():
    durations = constant_durations(
        {"eager": 20.0, "compiled": 10.0, "compiled_nocudagraphs": 12.0, "candidate": 5.0}
    )
    result, _timer, _log = build(durations)
    data = result.to_dict()

    assert data["rounds"] == 7
    assert data["warmup_rounds"] == 2
    assert data["baseline"] == "compiled"
    assert data["timings_ms"]["candidate"] == [5.0] * 7
    assert data["scoring_timings_ms"]["candidate"] == [5.0] * 5
    assert data["median_ratio"]["candidate"] == pytest.approx(2.0)
    assert data["call_order"] == list(LABELS) * 7
    assert isinstance(result, BenchResult)


# -- autograd must be off for every model call ----------------------------------------
#
# Batch 001 slot 0 (000-identity) spent 3162s in the benchmark and then died with
# `BackendCompilerFailed`: autograd was live, so inductor compiled the backward graph as
# well as the forward, and the fp32 recurrent state's in-place update tripped autograd's
# version counter. The same model under the correctness gate's `torch.no_grad()` took
# 17.1s and passed. The benchmark measures inference; nothing it calls needs a graph.


def grad_probe_columns(labels, record):
    """Columns that record whether autograd was enabled when they were called."""
    import torch

    columns = {}
    for label in labels:
        columns[label] = (lambda name: lambda: record.append((name, torch.is_grad_enabled())))(label)
    return columns


def test_every_timed_call_runs_with_autograd_disabled():
    import torch

    record: list[tuple[str, bool]] = []
    columns = grad_probe_columns(LABELS, record)
    timer = ScriptedTimer(constant_durations({label: 1.0 for label in LABELS}))
    timer.bind(columns)

    assert torch.is_grad_enabled(), "this test is meaningless if grad is already off"
    run_interleaved(columns, BenchConfig(), timer=timer)

    assert record, "no column was ever called"
    enabled = [name for name, grad_on in record if grad_on]
    assert enabled == [], f"these timed calls ran with autograd live: {sorted(set(enabled))}"


def test_untimed_warmup_calls_also_run_with_autograd_disabled():
    """The warmup calls are where `torch.compile` does its work, so they are exactly
    where a backward graph would get compiled."""
    import torch

    record: list[tuple[str, bool]] = []
    columns = grad_probe_columns(LABELS, record)
    timer = ScriptedTimer(constant_durations({label: 1.0 for label in LABELS}))
    timer.bind(columns)

    run_interleaved(columns, BenchConfig(untimed_warmup_calls=2), timer=timer)

    assert any(True for _ in record)
    assert [name for name, grad_on in record if grad_on] == []
    assert torch.is_grad_enabled(), "run_interleaved must not leak its grad mode to the caller"


def test_setups_run_with_autograd_disabled():
    """A setup restores a decode cache. Building that under autograd makes the cache
    tensors graph-tracked, which is how the in-place state update becomes an error."""
    import torch

    record: list[tuple[str, bool]] = []
    columns = grad_probe_columns(LABELS, record)
    setup_grad: list[bool] = []
    timer = ScriptedTimer(constant_durations({label: 1.0 for label in LABELS}))
    timer.bind(columns)

    run_interleaved(
        columns,
        BenchConfig(),
        timer=timer,
        setups={"candidate": lambda: setup_grad.append(torch.is_grad_enabled())},
    )

    assert setup_grad, "the setup never ran"
    assert not any(setup_grad), "setups ran with autograd live"


# -- saying where the time went, while it is still going -----------------------------------


def test_progress_reports_each_column_as_it_finishes():
    """A silent benchmark is how four rentals died without saying what they were doing.

    Rental 32's slot 0 spent its entire 6980.9s cap inside one `run_interleaved` call and
    recorded nothing: the `bench` phase is only written once the call returns, so a slot
    killed inside it leaves no evidence at all. Warmup round 1 is where compilation
    happens, so the first line of progress is the one that matters.
    """
    columns, _log = make_columns(["compiled", "candidate_compiled"])
    timer = ScriptedTimer({"compiled": [10.0] * 3, "candidate_compiled": [20.0] * 3})
    seen: list[tuple[int, str, float]] = []

    run_interleaved(
        timer.bind(columns),
        BenchConfig(rounds=3, warmup_rounds=2),
        timer=timer,
        progress=lambda round_index, label, elapsed_ms: seen.append((round_index, label, elapsed_ms)),
    )

    assert [(r, label) for r, label, _ in seen] == [
        (0, "compiled"),
        (0, "candidate_compiled"),
        (1, "compiled"),
        (1, "candidate_compiled"),
        (2, "compiled"),
        (2, "candidate_compiled"),
    ]
    assert seen[0][2] == 10.0


def test_progress_is_optional():
    """The CPU smoke path and every existing caller pass no progress callback."""
    columns, _log = make_columns(["compiled"])
    timer = ScriptedTimer({"compiled": [1.0] * 3})

    result = run_interleaved(timer.bind(columns), BenchConfig(rounds=3, warmup_rounds=2), timer=timer)

    assert result.labels == ("compiled",)
