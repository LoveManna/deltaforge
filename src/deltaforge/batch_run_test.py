"""Tests for the batch loop's control flow, with the GPU work stubbed out.

`run_batch` is where the design's load-bearing promises live: a failing slot must not end
the batch, a slot that will not fit must not be started, and every hypothesis must end up
with an outcome. Those are all decidable without a GPU by driving the loop with a fake
runner, and they are exactly the properties that are expensive to discover are wrong on a
rented box.
"""

from __future__ import annotations

from types import SimpleNamespace

from .batch import Batch, Hypothesis, SlotBudget
from .batch_run import SlotResult, release_compiled_state, run_batch


def hyp(slug: str, prediction: str = "inconclusive", kernels=("k",)) -> Hypothesis:
    return Hypothesis(
        slug=slug,
        kernels=kernels,
        category="A",
        byte_share=0.01,
        mechanism="does a thing",
        prediction=prediction,
        rationale="a rationale long enough to be a claim rather than a label, stated up front",
    )


def batch_of(*hypotheses) -> Batch:
    return Batch(batch_id="test", hypotheses=tuple(hypotheses))


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeRunner:
    """Stands in for `BatchRunner`, returning scripted results per slug."""

    def __init__(self, results: dict, clock: FakeClock | None = None, slot_seconds: float = 100.0):
        self.results = results
        self.clock = clock
        self.slot_seconds = slot_seconds
        self.prepared = False
        self.ran: list[str] = []
        self.caps: list[float] = []

    def prepare_reference(self) -> None:
        self.prepared = True

    def run_slot(self, hypothesis: Hypothesis, cap_s: float = 0.0) -> SlotResult:
        self.ran.append(hypothesis.slug)
        self.caps.append(cap_s)
        if self.clock is not None:
            self.clock.advance(self.slot_seconds)
        result = self.results.get(hypothesis.slug)
        if result is None:
            result = SlotResult(
                hypothesis=hypothesis,
                outcome="inconclusive",
                median_ratio=1.0,
                iqr_ratio=0.01,
                duration_s=self.slot_seconds,
            )
        return result


def silent(*_args, **_kwargs) -> None:
    return None


def test_the_reference_is_prepared_once_before_any_slot():
    runner = FakeRunner({})
    batch = batch_of(hyp("a"), hyp("b"))

    run_batch(runner, batch, budget=SlotBudget(deadline_epoch=1e12), log=silent)

    assert runner.prepared
    assert runner.ran == ["a", "b"]


def test_a_failing_slot_does_not_end_the_batch():
    """The property the whole design rests on: a wrong kernel costs a slot, not the rental."""
    batch = batch_of(hyp("a"), hyp("boom"), hyp("c"))
    runner = FakeRunner(
        {
            "boom": SlotResult(
                hypothesis=batch.get("boom"),
                outcome="error",
                error="Triton exploded",
                duration_s=5.0,
            )
        }
    )

    results, _calibrated, _scores = run_batch(
        runner, batch, budget=SlotBudget(deadline_epoch=1e12), log=silent
    )

    assert runner.ran == ["a", "boom", "c"]
    assert [r.outcome for r in results] == ["inconclusive", "error", "inconclusive"]


def test_every_hypothesis_gets_an_outcome():
    batch = batch_of(hyp("a"), hyp("b"), hyp("c"))
    results, _calibrated, _scores = run_batch(
        FakeRunner({}), batch, budget=SlotBudget(deadline_epoch=1e12), log=silent
    )

    assert [r.hypothesis.slug for r in results] == ["a", "b", "c"]
    assert all(r.outcome for r in results)


def test_slots_that_do_not_fit_are_recorded_as_not_run_rather_than_omitted():
    """A hypothesis missing from a record must be distinguishable from one that ran and
    produced nothing."""
    clock = FakeClock()
    batch = batch_of(hyp("a"), hyp("b"), hyp("c"), hyp("d"))
    # Room for roughly two 100s slots once the 1.2 safety factor is applied.
    budget = SlotBudget(deadline_epoch=260.0, clock=clock, seed_estimate_s=100.0)

    results, _calibrated, _scores = run_batch(
        FakeRunner({}, clock=clock, slot_seconds=100.0), batch, budget=budget, log=silent
    )

    outcomes = {r.hypothesis.slug: r.outcome for r in results}
    assert outcomes["a"] == "inconclusive"
    assert outcomes["b"] == "inconclusive"
    assert outcomes["c"] == "not_run"
    assert outcomes["d"] == "not_run"
    assert len(results) == 4


def test_the_budget_learns_from_the_slots_that_ran():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=10_000.0, clock=clock, seed_estimate_s=240.0)
    batch = batch_of(hyp("a"), hyp("b"))

    run_batch(FakeRunner({}, clock=clock, slot_seconds=30.0), batch, budget=budget, log=silent)

    # Seeded at 240s, but both slots took 30, so the estimate follows the evidence.
    assert budget.estimate_s() == 30.0


def test_a_not_run_slot_does_not_pollute_the_duration_estimate():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=140.0, clock=clock, seed_estimate_s=100.0)
    batch = batch_of(hyp("a"), hyp("b"), hyp("c"))

    run_batch(FakeRunner({}, clock=clock, slot_seconds=100.0), batch, budget=budget, log=silent)

    assert budget.durations_s == [100.0]


# -- calibration -------------------------------------------------------------------------


def identity_batch(*rest) -> Batch:
    return batch_of(hyp("000-identity", prediction="identity", kernels=()), *rest)


def test_an_identity_slot_at_one_calibrates_the_batch():
    batch = identity_batch(hyp("a"))
    runner = FakeRunner(
        {
            "000-identity": SlotResult(
                hypothesis=batch.get("000-identity"),
                outcome="inconclusive",
                median_ratio=1.002,
                iqr_ratio=0.01,
                duration_s=10.0,
            )
        }
    )

    _results, calibrated, _scores = run_batch(
        runner, batch, budget=SlotBudget(deadline_epoch=1e12), log=silent
    )

    assert calibrated is True


def test_an_identity_slot_away_from_one_voids_the_batch():
    batch = identity_batch(hyp("a"))
    runner = FakeRunner(
        {
            "000-identity": SlotResult(
                hypothesis=batch.get("000-identity"),
                outcome="win",
                median_ratio=1.4,
                iqr_ratio=0.01,
                duration_s=10.0,
            )
        }
    )

    _results, calibrated, _scores = run_batch(
        runner, batch, budget=SlotBudget(deadline_epoch=1e12), log=silent
    )

    assert calibrated is False


def test_an_errored_identity_slot_does_not_calibrate():
    batch = identity_batch(hyp("a"))
    runner = FakeRunner(
        {
            "000-identity": SlotResult(
                hypothesis=batch.get("000-identity"), outcome="error", error="boom", duration_s=1.0
            )
        }
    )

    _results, calibrated, _scores = run_batch(
        runner, batch, budget=SlotBudget(deadline_epoch=1e12), log=silent
    )

    assert calibrated is False


def test_a_batch_with_no_identity_slot_reports_calibration_as_unknown():
    """`None` rather than `False`: nothing was claimed and nothing was refuted, and the
    record must not imply the harness was checked when it was not."""
    _results, calibrated, _scores = run_batch(
        FakeRunner({}), batch_of(hyp("a")), budget=SlotBudget(deadline_epoch=1e12), log=silent
    )

    assert calibrated is None


# -- prediction scoring --------------------------------------------------------------------


def test_predictions_are_scored_against_the_measured_outcomes():
    batch = identity_batch(hyp("right", prediction="inconclusive"), hyp("wrong", prediction="win"))
    runner = FakeRunner(
        {
            "000-identity": SlotResult(
                hypothesis=batch.get("000-identity"),
                outcome="inconclusive",
                median_ratio=1.0,
                iqr_ratio=0.01,
                duration_s=1.0,
            )
        }
    )

    _results, _calibrated, scores = run_batch(
        runner, batch, budget=SlotBudget(deadline_epoch=1e12), log=silent
    )

    by_slug = {s.slug: s.correct for s in scores}
    assert by_slug["000-identity"] is True
    assert by_slug["right"] is True
    assert by_slug["wrong"] is False


def test_on_slot_is_called_for_every_slot_that_ran_so_records_hit_disk_early():
    seen = []
    batch = batch_of(hyp("a"), hyp("b"))

    run_batch(
        FakeRunner({}),
        batch,
        budget=SlotBudget(deadline_epoch=1e12),
        on_slot=lambda r: seen.append(r.hypothesis.slug),
        log=silent,
    )

    assert seen == ["a", "b"]


def test_slot_dict_carries_the_prediction_and_its_rationale():
    """The record has to stand alone: a reader must be able to see what was claimed in
    advance without going back to the manifest at the commit the run used."""
    result = SlotResult(hypothesis=hyp("a", prediction="win"), outcome="loss", median_ratio=0.8)
    slot = result.to_slot_dict()

    assert slot["prediction"] == "win"
    assert slot["outcome"] == "loss"
    assert slot["rationale"]
    assert slot["mechanism"]
    assert slot["byte_share"] == 0.01


# -- releasing what a finished slot leaves on the card ------------------------------------


def _fake_torch(calls: list[str], cuda: bool = True):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            empty_cache=lambda: calls.append("empty_cache"),
            synchronize=lambda: calls.append("synchronize"),
        )
    )


def test_releasing_compiled_state_leaves_the_cudagraph_trees_alone():
    """Resetting the trees invalidates the reference columns, which is fatal to the batch.

    This used to call `reset_cudagraph_trees` to give back the graph pool a finished slot
    was holding, on the stated premise that the reference would "re-record on the next
    slot's first warmup call". It does not. The shutdown is permanent for a callable that
    has already been recorded, and inductor's cudagraph trees are per *device*, not per
    model -- so releasing the candidate tore down the reference's graphs too.

    Rental 34 is the proof, and it could only appear once a slot finally completed: slot 0
    calibrated at ratio 1.0009, and then all eight scoring slots died in 23s each on
    `AssertionError: Running CUDAGraph after shutdown`, before any of them timed anything.

    The reclaim it bought is also smaller than it looks. Slot 0 recorded CUDA graphs for
    both columns and measured 8.07 GiB allocated / 8.08 reserved of 31.36 both before and
    after the release — at two columns with autograd off, the reset gave back less than the
    logged number resolves. Losing that to a measured bound beats keeping a guarantee that
    empties the batch.
    """
    calls: list[str] = []
    cudagraphs = SimpleNamespace(reset_cudagraph_trees=lambda: calls.append("reset_cudagraph_trees"))

    steps = release_compiled_state(_fake_torch(calls), cudagraphs, log=silent)

    assert "reset_cudagraph_trees" not in calls
    assert "reset_cudagraph_trees" not in steps
    assert "empty_cache" in calls


def test_releasing_compiled_state_survives_a_torch_without_the_private_api():
    """`reset_cudagraph_trees` is private API. A torch that lacks it must cost us the
    reclaim, not the batch."""
    calls: list[str] = []

    steps = release_compiled_state(_fake_torch(calls), SimpleNamespace(), log=silent)

    assert "empty_cache" in calls
    assert "reset_cudagraph_trees" not in steps


def test_releasing_compiled_state_does_nothing_without_cuda():
    calls: list[str] = []

    assert release_compiled_state(_fake_torch(calls, cuda=False), None, log=silent) == []
    assert calls == []


# -- the cap, and what the clock ending a rental is called --------------------------------


def test_each_slot_is_handed_the_cap_its_index_earns():
    """Slots 0 and 1 get the whole remaining budget; later slots get the ceiling."""
    clock = FakeClock()
    runner = FakeRunner({}, clock=clock, slot_seconds=0.0)
    batch = batch_of(hyp("a"), hyp("b"), hyp("c"))
    budget = SlotBudget(deadline_epoch=1000.0, clock=clock, slot_cap_s=60.0)

    run_batch(runner, batch, budget=budget, log=silent)

    assert runner.caps == [1000.0, 1000.0, 60.0]


def test_a_rental_that_scored_nothing_records_starved_not_merely_not_run():
    """`not_run` means the batch stopped early having already measured something.

    A rental where the clock arrived before any slot scored produced nothing, and the
    session gate is the reason. Flattening the two would erase the evidence for raising
    that gate — which is exactly what the last two rentals needed and did not have.
    """
    clock = FakeClock()
    runner = FakeRunner(
        {"a": SlotResult(hypothesis=hyp("a"), outcome="error", error="boom", duration_s=100.0)},
        clock=clock,
        slot_seconds=100.0,
    )
    batch = batch_of(hyp("a"), hyp("b"), hyp("c"))
    budget = SlotBudget(deadline_epoch=150.0, clock=clock, seed_estimate_s=100.0)

    results, _calibrated, _scores = run_batch(runner, batch, budget=budget, log=silent)

    assert [r.outcome for r in results] == ["error", "starved", "starved"]


def test_a_batch_that_measured_something_before_stopping_records_not_run():
    clock = FakeClock()
    runner = FakeRunner({}, clock=clock, slot_seconds=100.0)
    batch = batch_of(hyp("a"), hyp("b"), hyp("c"))
    budget = SlotBudget(deadline_epoch=150.0, clock=clock, seed_estimate_s=100.0)

    results, _calibrated, _scores = run_batch(runner, batch, budget=budget, log=silent)

    assert results[0].outcome == "inconclusive"
    assert [r.outcome for r in results[1:]] == ["not_run", "not_run"]


# -- what the untimed prefill is allowed to compile ---------------------------------------


class RecordingCompile:
    """Stands in for the object `torch.compile` returns, and says when it was called."""

    def __init__(self, model) -> None:
        self.model = model
        self.calls: list[int] = []

    def __call__(self, input_ids, *args, **kwargs):
        self.calls.append(int(input_ids.shape[1]))
        return self.model(input_ids, *args, **kwargs)


def _column_under_test(monkeypatch, context: int, tokens: int):
    """`BatchRunner._make_column` against the tiny CPU model, with compilation recorded."""
    import torch

    from .batch_run import BatchRunner
    from .config import tiny_config
    from .reference import ReferenceModel

    config = tiny_config()
    model = ReferenceModel(config).eval()
    compiled: list[RecordingCompile] = []

    def fake_compile(target, **_kwargs):
        wrapper = RecordingCompile(target)
        compiled.append(wrapper)
        return wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)

    runner = BatchRunner(
        config=config,
        reference=model,
        prompt=torch.randint(0, config.vocab_size, (1, context)),
        prompt_ids=None,
        workload={"batch_size": 1, "context_length": context, "decode_tokens": tokens},
        weights_dtype=torch.float32,
        bench_config=None,
        max_new_tokens=tokens,
        columns=("compiled",),
        log=silent,
    )
    setup, run = runner._make_column(model, "max-autotune")
    return setup, run, compiled


def test_the_untimed_prefill_does_not_go_through_the_compiled_wrapper(monkeypatch):
    """Compiling the prefill is what stopped this project, and none of it is scored.

    `gated_delta_rule` unrolls its scan over the sequence, so the benchmark's 2048-token
    prefill hands inductor about 22 nodes x 2048 tokens x 24 linear-attention layers --
    roughly a million FX nodes -- under `max-autotune`, once per column. Rental 32's slot 0
    burned its whole 6980.9s cap there without finishing, and rentals 22, 30 and 31 died on
    the same graph.

    `run_interleaved` excludes every `setup` from the timed region by construction, so that
    compilation buys no measurement at all. The decode calls, which *are* timed, run the
    scan once per step and stay small -- those still go through the compiled wrapper, which
    is the whole claim.
    """
    setup, run, compiled = _column_under_test(monkeypatch, context=16, tokens=3)
    assert len(compiled) == 1, "the column should compile exactly one runnable"
    wrapper = compiled[0]

    setup()

    assert wrapper.calls == [], (
        "the prefill reached the compiled wrapper; at the real workload that is a "
        "million-node max-autotune compile of a graph nothing times"
    )

    run()

    assert wrapper.calls == [1, 1, 1], "every timed decode step must use the compiled model"


def test_the_untimed_prefill_still_fills_the_cache(monkeypatch):
    """Moving the prefill off the compiled wrapper must not move it out of the run.

    The decode steps measure attention over a full-length cache. A setup that skipped the
    prefill would leave an empty one, and every column would then time the wrong workload
    while still producing a plausible-looking ratio.
    """
    from .reference import ReferenceModel

    seen: list[int] = []
    original = ReferenceModel.forward

    def recording_forward(self, input_ids, *args, **kwargs):
        seen.append(int(input_ids.shape[1]))
        return original(self, input_ids, *args, **kwargs)

    monkeypatch.setattr(ReferenceModel, "forward", recording_forward)

    setup, run, _compiled = _column_under_test(monkeypatch, context=16, tokens=3)

    setup()
    assert seen == [16], "the prefill must still run, eagerly, before the timed decode"

    run()
    assert seen == [16, 1, 1, 1], "the timed region is the decode steps over the filled cache"
