"""Tests for the batch loop's control flow, with the GPU work stubbed out.

`run_batch` is where the design's load-bearing promises live: a failing slot must not end
the batch, a slot that will not fit must not be started, and every hypothesis must end up
with an outcome. Those are all decidable without a GPU by driving the loop with a fake
runner, and they are exactly the properties that are expensive to discover are wrong on a
rented box.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from .batch import Batch, Hypothesis, Precondition, SlotBudget
from .batch_run import SlotResult, release_compiled_state, run_batch


def hyp(
    slug: str, prediction: str = "inconclusive", kernels=("k",), requires=None, weight_bits=None
) -> Hypothesis:
    return Hypothesis(
        slug=slug,
        kernels=kernels,
        requires=requires,
        weight_bits=weight_bits or {},
        category="A",
        byte_share=0.01,
        mechanism="does a thing",
        prediction=prediction,
        rationale="a rationale long enough to be a claim rather than a label, stated up front",
        # `exact` asserts bit-identity, which only a candidate installing nothing has.
        correctness="exact" if not kernels else "approximate",
        top1_threshold=None if not kernels else 0.9,
        kl_threshold=None if not kernels else 0.01,
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


# -- dynamo's recompile limit, which is a ceiling on how many slots a batch can measure ----


def test_dynamo_falls_back_to_eager_once_a_code_object_is_recompiled_too_often():
    """The mechanism that voided six of rental 35's nine slots, reproduced on a CPU.

    Dynamo caches compiled code per *code object* with guards, and a fresh candidate module
    is a fresh guard. A batch builds one candidate per slot against the same
    `ReferenceModel.forward`, so slot N is cache entry N — and at `recompile_limit` (8 by
    default) dynamo stops compiling that code object **and runs it eagerly, for the rest of
    the process**, with a warning and no error.

    Rental 35 hit it inside slot 2. From slot 3 on, the candidate column's first round
    dropped from ~170s of compiling to a flat ~6s of not compiling, and all six remaining
    ratios clustered at 0.15 regardless of which kernel was installed, because what they
    measured was eager against compiled.
    """
    import torch
    import torch._dynamo as dynamo

    from .config import tiny_config
    from .reference import ReferenceModel

    dynamo.reset()
    compiled_count = 0

    def counting_backend(gm, example_inputs):
        nonlocal compiled_count
        compiled_count += 1
        return gm.forward

    limit = 4
    with dynamo.config.patch(recompile_limit=limit):
        config = tiny_config()
        prompt = torch.randint(0, config.vocab_size, (1, 4))
        for _ in range(limit + 3):
            model = ReferenceModel(config).eval()
            runnable = torch.compile(model, backend=counting_backend, dynamic=False)
            with torch.no_grad():
                runnable(prompt, model.new_cache(1, 8), num_logits_to_keep=1)

    assert compiled_count <= limit, (
        f"dynamo compiled {compiled_count} times against a limit of {limit}; past the limit "
        "it silently runs eager, which is what a slot then measures"
    )


def test_the_batch_raises_the_recompile_limit_to_cover_every_slot():
    """Each slot legitimately needs its own entry, so the default limit caps the batch.

    Raising it is not papering over runaway recompilation: the recompiles here are bounded
    and intended, one candidate per slot plus a few for dynamic shapes. Leaving the default
    in place means a batch longer than a handful of slots reports eager timings as kernel
    results — silently, and with a plausible-looking ratio.
    """
    from .batch_run import recompile_limit_for

    assert recompile_limit_for(9) >= 9
    # Still bounded: a limit that grows without end would hide a real runaway.
    assert recompile_limit_for(9) <= 64
    # Never lowers what torch already allows.
    assert recompile_limit_for(1) >= 8


def test_a_slot_records_how_many_graphs_dynamo_actually_compiled():
    """A ratio from a candidate that never compiled is not a comparison, and looks like one.

    Rental 35 published six of them — 0.146 to 0.157, tight enough to look like a real
    effect — from a candidate dynamo had stopped compiling. Nothing in the record said so;
    the only trace was a warning in the rental log and a first round that was suspiciously
    quick. `AGENT.md` §3 already tells a session to confirm the baseline actually compiled.
    This makes the run confirm it, for both columns, without being asked.
    """
    from .batch_run import graphs_compiled_during

    counters = {"stats": {"unique_graphs": 3}}

    with graphs_compiled_during(counters) as count:
        counters["stats"]["unique_graphs"] = 7

    assert count.compiled == 4


def test_graph_counting_survives_a_torch_that_does_not_offer_the_counter():
    """Private API. Its absence must cost the evidence, not the slot."""
    from .batch_run import graphs_compiled_during

    with graphs_compiled_during({}) as count:
        pass

    assert count.compiled is None


# -- the byte model a slot scores its bandwidth against -------------------------------


def _runner_for(config):
    """A `BatchRunner` with nothing but the fields the byte model reads."""
    from .batch_run import BatchRunner
    from .harness.bench import BenchConfig

    return BatchRunner(
        config=config,
        reference=None,
        prompt=None,
        prompt_ids=None,
        workload={"batch_size": 1, "context_length": 2048, "decode_tokens": 128},
        weights_dtype=None,
        bench_config=BenchConfig(rounds=3, warmup_rounds=1),
        max_new_tokens=8,
        columns=("compiled", "candidate_compiled"),
        log=silent,
    )


def test_a_slot_scores_both_columns_against_the_checkpoints_own_byte_count():
    """Without this the slot record carries a ratio and no way to read it.

    Batch 003's whole finding -- that the kernel was issue-bound, not bandwidth-bound --
    had to be reconstructed by hand from two files after the rental was over.
    """
    from .config import qwen3_5_4b_config

    per_token = _runner_for(qwen3_5_4b_config())._bytes_per_token(hyp("a"))

    assert set(per_token) == {"compiled", "candidate_compiled"}
    assert per_token["compiled"] == pytest.approx(9158.23, abs=1.0)
    assert per_token["candidate_compiled"] == per_token["compiled"], "declares no re-encoding"


def test_a_quantised_slot_is_scored_against_the_bytes_it_actually_moves():
    """Scoring a quantised candidate against bf16 byte counts would report a bandwidth it
    never achieved, and that number is the one the batch is read on."""
    from .config import qwen3_5_4b_config

    runner = _runner_for(qwen3_5_4b_config())
    per_token = runner._bytes_per_token(hyp("a", weight_bits={"layers": 8}))

    # The layer projections are 7140 MB/token of the 9158 total; half of that is removed.
    assert per_token["compiled"] - per_token["candidate_compiled"] == pytest.approx(3570.0, abs=5.0)


def test_a_config_with_no_published_manifest_reports_no_bytes_rather_than_a_guess():
    """`tiny_config` is a CPU fixture, not a checkpoint. No manifest means no byte model."""
    from .config import tiny_config

    assert _runner_for(tiny_config())._bytes_per_token(hyp("a")) == {}


# -- a slot that declines to run ------------------------------------------------------


def test_a_failed_precondition_records_why_rather_than_running_the_slot():
    """`precondition_failed` is distinct from `not_run` on purpose.

    `not_run` means the clock arrived first; `starved` means the rental scored nothing;
    this means the batch decided the slot could not tell us anything. Flattening them would
    erase the evidence for whether the gate was set correctly.
    """
    control = hyp("015-gemv-bf16")
    batch = batch_of(
        control,
        hyp("017-fp8", requires=Precondition("015-gemv-bf16", 0.56, "kernel not memory-bound")),
    )
    runner = FakeRunner(
        {
            "015-gemv-bf16": SlotResult(
                hypothesis=control,
                outcome="loss",
                median_ratio=0.28,
                iqr_ratio=0.01,
                duration_s=100.0,
            )
        }
    )

    results, _, _ = run_batch(runner, batch, budget=SlotBudget(deadline_epoch=1e12), log=silent)

    assert [r.outcome for r in results] == ["loss", "precondition_failed"]
    assert "kernel not memory-bound" in results[1].error
    assert "0.28" in results[1].error, "the observed ratio belongs in the record"
    assert runner.ran == ["015-gemv-bf16"], "the skipped slot must not have been run"


def test_a_met_precondition_runs_the_slot():
    control = hyp("015-gemv-bf16")
    batch = batch_of(
        control,
        hyp("017-fp8", requires=Precondition("015-gemv-bf16", 0.56, "kernel not memory-bound")),
    )
    runner = FakeRunner(
        {
            "015-gemv-bf16": SlotResult(
                hypothesis=control, outcome="loss", median_ratio=0.80, iqr_ratio=0.01, duration_s=100.0
            )
        }
    )

    results, _, _ = run_batch(runner, batch, budget=SlotBudget(deadline_epoch=1e12), log=silent)

    assert runner.ran == ["015-gemv-bf16", "017-fp8"]
    assert results[1].outcome != "precondition_failed"


def test_a_failed_precondition_does_not_stop_the_batch():
    """A later slot may have a different precondition, or none. Breaking would throw away
    every slot behind the first one that declined."""
    control = hyp("015-gemv-bf16")
    batch = batch_of(
        control,
        hyp("017-fp8", requires=Precondition("015-gemv-bf16", 0.56, "not memory-bound")),
        hyp("018-unrelated"),
    )
    runner = FakeRunner(
        {
            "015-gemv-bf16": SlotResult(
                hypothesis=control, outcome="loss", median_ratio=0.28, iqr_ratio=0.01, duration_s=100.0
            )
        }
    )

    results, _, _ = run_batch(runner, batch, budget=SlotBudget(deadline_epoch=1e12), log=silent)

    assert [r.outcome for r in results] == ["loss", "precondition_failed", "inconclusive"]
    assert runner.ran == ["015-gemv-bf16", "018-unrelated"]


def test_a_precondition_on_a_slot_that_errored_skips_rather_than_runs():
    """Fails closed: no ratio is not a good ratio."""
    control = hyp("015-gemv-bf16")
    batch = batch_of(
        control,
        hyp("017-fp8", requires=Precondition("015-gemv-bf16", 0.56, "not memory-bound")),
    )
    runner = FakeRunner({"015-gemv-bf16": SlotResult(hypothesis=control, outcome="error", duration_s=10.0)})

    results, _, _ = run_batch(runner, batch, budget=SlotBudget(deadline_epoch=1e12), log=silent)

    assert results[1].outcome == "precondition_failed"
    assert runner.ran == ["015-gemv-bf16"]
