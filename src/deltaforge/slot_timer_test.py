"""Tests for the cap that stops one slot from taking the rental."""

from __future__ import annotations

import time

import pytest

from .slot_timer import SlotTimeout, slot_deadline


def test_work_that_finishes_in_time_is_untouched():
    with slot_deadline(5.0) as armed:
        result = sum(range(1000))

    assert result == 499500
    assert armed.fired is False


def test_work_that_overruns_is_interrupted():
    """The interrupt arrives as KeyboardInterrupt and leaves as SlotTimeout.

    `BatchRunner.run_slot` isolates a slot by catching `Exception`. A cap that raised
    something outside that hierarchy would take the rental down, which is the failure it
    exists to prevent.
    """
    started = time.monotonic()

    with pytest.raises(SlotTimeout) as excinfo, slot_deadline(0.1):
        while True:
            time.sleep(0.01)

    assert time.monotonic() - started < 5.0
    assert "0.1" in str(excinfo.value)


def test_a_cap_of_zero_or_less_does_not_arm():
    """A slot with no time left is refused by SlotBudget.can_start, not by a timer that
    fires instantly and turns a clean stop into an error record."""
    with slot_deadline(0.0) as armed:
        time.sleep(0.05)

    assert armed.fired is False


def test_the_timer_is_disarmed_when_the_body_raises():
    with pytest.raises(ValueError), slot_deadline(30.0) as armed:
        raise ValueError("boom")

    assert armed.fired is False
    time.sleep(0.05)  # nothing may arrive after the body has left


def test_an_unrelated_keyboard_interrupt_is_not_relabelled():
    """Ctrl-C during a slot is a person stopping the run, not the cap firing. Reporting it
    as a timeout would put a fault in the record that never happened."""
    with pytest.raises(KeyboardInterrupt), slot_deadline(30.0):
        raise KeyboardInterrupt
