"""Tests for the batch model, the outcome arithmetic and the deadline policy.

All of it decidable on a CPU with no checkpoint, which is the point of keeping torch out
of `batch.py`: the parts that decide what a number *means* are the parts worth testing
exhaustively, and they must not need a rented GPU to exercise.
"""

from __future__ import annotations

import pytest

from .batch import (
    Batch,
    Hypothesis,
    SlotBudget,
    calibration_holds,
    classify_outcome,
    scoped_registry,
    score_predictions,
)
from .kernels import KernelRegistry, KernelStatus, RegistryError


def make_hypothesis(**overrides) -> Hypothesis:
    base = {
        "slug": "999-test",
        "kernels": ("k1",),
        "category": "B",
        "byte_share": 0.05,
        "mechanism": "moves fewer bytes",
        "prediction": "win",
        "rationale": "because the compiler cannot change the representation",
    }
    base.update(overrides)
    return Hypothesis(**base)


def make_registry() -> KernelRegistry:
    registry = KernelRegistry()
    registry.register("k1", impl=lambda: None, replaces="rms_norm", status=KernelStatus.RETIRED)
    registry.register("k2", impl=lambda: None, replaces="swiglu_mlp", status=KernelStatus.RETIRED)
    registry.register("k3", impl=lambda: None, replaces="rms_norm", status=KernelStatus.RETIRED)
    return registry


# -- Hypothesis validation --------------------------------------------------------------


def test_prediction_must_be_known():
    with pytest.raises(ValueError, match="prediction must be one of"):
        make_hypothesis(prediction="probably")


def test_byte_share_is_a_fraction_not_a_percentage():
    # 6.23% is 0.0623. Writing 6.23 would silently claim the hypothesis attacks 623% of
    # per-token bytes and would sort the backlog wrongly.
    with pytest.raises(ValueError, match="fraction"):
        make_hypothesis(byte_share=6.23)


def test_a_hypothesis_installing_nothing_must_predict_identity():
    with pytest.raises(ValueError, match="identity champion"):
        make_hypothesis(kernels=(), prediction="win")


def test_identity_prediction_requires_installing_nothing():
    with pytest.raises(ValueError, match="bit-identical"):
        make_hypothesis(kernels=("k1",), prediction="identity")


def test_identity_hypothesis_is_valid_and_flagged():
    hyp = make_hypothesis(kernels=(), prediction="identity")
    assert hyp.is_identity


def test_mechanism_is_required():
    with pytest.raises(ValueError, match="not a hypothesis"):
        make_hypothesis(mechanism="   ")


def test_rationale_is_required_because_the_prediction_is_the_result():
    with pytest.raises(ValueError, match="unexplained one is worth nothing"):
        make_hypothesis(rationale="")


# -- Batch ------------------------------------------------------------------------------


def test_batch_rejects_duplicate_slugs():
    hyp = make_hypothesis()
    with pytest.raises(ValueError, match="twice"):
        Batch(batch_id="b", hypotheses=(hyp, hyp))


def test_batch_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        Batch(batch_id="b", hypotheses=())


def test_batch_preserves_manifest_order():
    # Order is load-bearing: calibration first so a broken harness costs three minutes,
    # riskiest last so the cheap results are already on disk when one fails.
    a = make_hypothesis(slug="000-identity", kernels=(), prediction="identity")
    b = make_hypothesis(slug="001-cheap")
    c = make_hypothesis(slug="008-risky")
    batch = Batch(batch_id="b", hypotheses=(a, b, c))
    assert [h.slug for h in batch] == ["000-identity", "001-cheap", "008-risky"]
    assert len(batch) == 3


def test_batch_finds_its_calibration_slot():
    a = make_hypothesis(slug="000-identity", kernels=(), prediction="identity")
    b = make_hypothesis(slug="001-cheap")
    assert Batch(batch_id="b", hypotheses=(a, b)).calibration_slug == "000-identity"
    assert Batch(batch_id="b", hypotheses=(b,)).calibration_slug is None


def test_batch_get_raises_on_unknown_slug():
    batch = Batch(batch_id="b", hypotheses=(make_hypothesis(),))
    with pytest.raises(KeyError):
        batch.get("nope")


# -- scoped_registry --------------------------------------------------------------------


def test_scoped_registry_promotes_only_the_named_kernels():
    source = make_registry()
    scoped = scoped_registry(make_hypothesis(kernels=("k1",)), source)
    assert set(scoped.champions()) == {"rms_norm"}
    assert scoped.champion("rms_norm").name == "k1"


def test_scoped_registry_does_not_mutate_the_source():
    # The whole reason scoped registries exist: a mutation here would leak into the next
    # slot, and a slot that inherited the previous slot's kernels would report a
    # plausible number for the wrong candidate.
    source = make_registry()
    scoped_registry(make_hypothesis(kernels=("k1", "k2")), source)
    assert source.get("k1").status is KernelStatus.RETIRED
    assert source.get("k2").status is KernelStatus.RETIRED
    assert source.champions() == {}


def test_scoped_registry_is_empty_for_the_identity_champion():
    scoped = scoped_registry(make_hypothesis(kernels=(), prediction="identity"), make_registry())
    assert scoped.champions() == {}
    assert len(scoped) == 0


def test_scoped_registry_rejects_two_kernels_replacing_the_same_operation():
    # Caught here rather than at benchmark time, on the ground, before the money.
    with pytest.raises(RegistryError, match="cannot install"):
        scoped_registry(make_hypothesis(kernels=("k1", "k3")), make_registry())


def test_scoped_registry_rejects_an_unknown_kernel_name():
    with pytest.raises(RegistryError, match="no kernel registered"):
        scoped_registry(make_hypothesis(kernels=("typo",)), make_registry())


def test_scoped_registry_records_the_owning_hypothesis():
    scoped = scoped_registry(make_hypothesis(slug="006-gqa", kernels=("k1",)), make_registry())
    assert scoped.get("k1").hypothesis == "006-gqa"


# -- classify_outcome -------------------------------------------------------------------


def test_a_wrong_candidate_is_incorrect_however_fast_it_was():
    assert classify_outcome(2.5, 0.01, correctness_passed=False) == "incorrect"


def test_margin_beyond_the_noise_band_is_a_win():
    assert classify_outcome(1.10, 0.02, correctness_passed=True) == "win"


def test_margin_inside_the_noise_band_is_inconclusive_not_a_win():
    # AGENT.md section 6: recording a noise-band result as a win is how a leaderboard
    # becomes fiction.
    assert classify_outcome(1.01, 0.02, correctness_passed=True) == "inconclusive"
    assert classify_outcome(0.99, 0.02, correctness_passed=True) == "inconclusive"


def test_margin_exactly_at_the_band_edge_is_inconclusive():
    assert classify_outcome(1.02, 0.02, correctness_passed=True) == "inconclusive"


def test_slower_than_the_band_is_a_loss():
    assert classify_outcome(0.90, 0.02, correctness_passed=True) == "loss"


def test_no_ratio_is_an_error_not_a_null_result():
    assert classify_outcome(None, 0.02, correctness_passed=True) == "error"


# -- calibration_holds ------------------------------------------------------------------


def test_identity_at_one_calibrates():
    assert calibration_holds(1.000, 0.01)


def test_identity_far_from_one_does_not_calibrate():
    assert not calibration_holds(1.15, 0.01)


def test_calibration_band_has_a_floor_so_it_stays_falsifiable():
    # An implausibly tight IQR must not reject a deviation that is plainly just noise.
    assert calibration_holds(1.015, 0.0001)
    assert not calibration_holds(1.15, 0.0001)


def test_calibration_uses_the_measured_iqr_when_it_is_wider_than_the_floor():
    assert calibration_holds(1.05, 0.08)


def test_missing_ratio_does_not_calibrate():
    assert not calibration_holds(None, 0.01)


# -- score_predictions ------------------------------------------------------------------


def batch_of_three() -> Batch:
    return Batch(
        batch_id="b",
        hypotheses=(
            make_hypothesis(slug="000-identity", kernels=(), prediction="identity"),
            make_hypothesis(slug="001-a", prediction="inconclusive"),
            make_hypothesis(slug="002-b", prediction="win"),
        ),
    )


def test_predictions_are_scored_against_measured_outcomes():
    scores = score_predictions(
        batch_of_three(),
        {"000-identity": "inconclusive", "001-a": "inconclusive", "002-b": "loss"},
        calibrated=True,
    )
    by_slug = {s.slug: s for s in scores}
    assert by_slug["000-identity"].correct is True
    assert by_slug["001-a"].correct is True
    assert by_slug["002-b"].correct is False


def test_an_errored_slot_scores_none_not_false():
    # The prediction was never tested. Counting it wrong understates the record exactly
    # as counting it right would flatter it.
    scores = score_predictions(batch_of_three(), {"002-b": "error"}, calibrated=True)
    assert {s.slug: s.correct for s in scores}["002-b"] is None


def test_a_slot_that_never_ran_scores_none():
    scores = score_predictions(batch_of_three(), {}, calibrated=True)
    assert all(s.correct is None for s in scores if s.slug != "000-identity")
    assert all(s.outcome == "not_run" for s in scores if s.slug != "000-identity")


def test_the_identity_slot_is_scored_on_calibration_not_on_its_ratio_band():
    scores = score_predictions(batch_of_three(), {"000-identity": "inconclusive"}, calibrated=False)
    assert {s.slug: s.correct for s in scores}["000-identity"] is False


# -- SlotBudget -------------------------------------------------------------------------


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_budget_uses_the_seed_estimate_before_any_slot_has_finished():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=1000.0, clock=clock, seed_estimate_s=240.0)
    assert budget.estimate_s() == 240.0
    assert budget.can_start()


def test_budget_refuses_when_the_remaining_time_will_not_fit_a_slot():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=200.0, clock=clock, seed_estimate_s=240.0)
    assert not budget.can_start()
    assert "Stopping here" in budget.why_not()


def test_budget_estimate_adapts_to_the_slots_actually_observed():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=10_000.0, clock=clock, seed_estimate_s=240.0)
    for duration in (100.0, 110.0, 120.0):
        budget.record(duration)
    assert budget.estimate_s() == 110.0


def test_budget_estimate_is_a_median_so_one_slow_slot_does_not_dominate():
    budget = SlotBudget(deadline_epoch=10_000.0, clock=FakeClock(), seed_estimate_s=240.0)
    for duration in (100.0, 105.0, 110.0, 2000.0):
        budget.record(duration)
    assert budget.estimate_s() == 107.5


def test_budget_applies_a_safety_factor_so_it_stops_early_rather_than_late():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=110.0, clock=clock, seed_estimate_s=1.0, safety_factor=1.2)
    budget.record(100.0)
    # 110s left, a slot takes 100s, but 100 * 1.2 = 120 > 110, so it declines.
    assert not budget.can_start()


def test_budget_counts_down_as_the_clock_advances():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=1000.0, clock=clock, seed_estimate_s=100.0)
    assert budget.can_start()
    clock.advance(900.0)
    assert not budget.can_start()
    assert budget.remaining_s() == 100.0


def test_budget_rejects_a_negative_duration():
    budget = SlotBudget(deadline_epoch=1.0, clock=FakeClock())
    with pytest.raises(ValueError, match="cannot take"):
        budget.record(-1.0)


def test_the_first_two_slots_are_never_capped_below_the_remaining_budget():
    """The cap exists to stop slot 7 eating slot 8.

    Applying it to the first two slots would defeat the guarantee it protects: the
    identity slot plus one kernel slot is the least a rental may produce and still have
    scored a hypothesis.
    """
    budget = SlotBudget(deadline_epoch=1000.0, clock=lambda: 0.0, slot_cap_s=60.0)

    assert budget.cap_for(0) == pytest.approx(1000.0)
    assert budget.cap_for(1) == pytest.approx(1000.0)
    assert budget.cap_for(2) == pytest.approx(60.0)


def test_a_capped_slot_never_outlives_the_deadline():
    """A cap larger than what is left is not a licence to overrun the session."""
    budget = SlotBudget(deadline_epoch=100.0, clock=lambda: 0.0, slot_cap_s=600.0)

    assert budget.cap_for(5) == pytest.approx(100.0)


def test_a_starved_prediction_is_untested_not_wrong():
    """A slot the clock killed tested nothing. Counting it wrong understates the record
    exactly as counting it right would flatter it."""
    batch = Batch(
        batch_id="b",
        hypotheses=(
            make_hypothesis(slug="000-identity", kernels=(), prediction="identity"),
            make_hypothesis(slug="001-x"),
        ),
    )

    scores = score_predictions(batch, {"000-identity": "win", "001-x": "starved"}, calibrated=True)

    assert scores[1].correct is None
    assert scores[1].outcome == "starved"
