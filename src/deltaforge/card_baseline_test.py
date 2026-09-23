"""The pre-flight rental 43 needed and did not have.

These assert on the *text*, because the text is the deliverable: this module exists to
say something at minute 27 that batch 007 only said in its writeup.
"""

from __future__ import annotations

from .card_baseline import (
    RECORDED_REFERENCE_GBPS,
    SLOW_FRACTION,
    card_report,
    recorded_for,
)

RTX_5090 = "NVIDIA GeForce RTX 5090"


def test_rental_43s_card_would_have_been_called_out_loudly():
    """845 GB/s against a recorded best of 1282. The whole point of the file."""
    report = card_report(RTX_5090, 845.3)

    assert "WARNING" in report
    assert "SLOW" in report
    assert "845" in report and "1283" in report


def test_rental_40s_card_reads_as_in_family():
    report = card_report(RTX_5090, 1282.6)

    assert "WARNING" not in report
    assert "In family" in report


def test_rental_42s_card_reads_as_in_family_too():
    """0.95 of the best. The threshold has to sit above rental 43 and below this."""
    assert SLOW_FRACTION < 1223.2 / 1282.6
    assert SLOW_FRACTION > 845.3 / 1282.6
    assert "WARNING" not in card_report(RTX_5090, 1223.2)


def test_an_unrecorded_gpu_model_says_so_rather_than_comparing_against_nothing():
    report = card_report("NVIDIA GeForce RTX 4090", 700.0)

    assert "never recorded this GPU model" in report
    assert "WARNING" not in report
    assert not recorded_for("NVIDIA GeForce RTX 4090")


def test_a_slot_that_declared_no_bytes_is_reported_as_uncharacterised():
    report = card_report(RTX_5090, None)

    assert "uncharacterised" in report
    assert "WARNING" not in report


def test_every_report_says_a_slow_card_does_not_void_a_ratio():
    """The one thing a reader could get wrong from a loud warning.

    Ratios are measured against a reference timed in the same interleaved rounds, so they
    survive a slow card. Absolute predictions and cross-rental comparisons do not.
    """
    for measured in (None, 700.0, 845.3, 1282.6):
        report = card_report(RTX_5090, measured)
        assert "does not void a ratio" in report or "uncharacterised" in report


def test_the_recorded_observations_are_the_rentals_the_writeups_name():
    seen = {o.rental: o.gbps for o in RECORDED_REFERENCE_GBPS[RTX_5090]}

    assert seen == {40: 1282.6, 42: 1223.2, 43: 845.3}
