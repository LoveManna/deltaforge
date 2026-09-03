"""The layer-2 gate inputs are part of the measurement and must not drift."""

from __future__ import annotations

from .prompts import CORRECTNESS_PROMPTS, PROMPT_DIGEST, digest


def test_there_are_exactly_five_prompts():
    assert len(CORRECTNESS_PROMPTS) == 5


def test_prompts_are_checked_into_the_repo_and_non_empty():
    for prompt in CORRECTNESS_PROMPTS:
        assert prompt.strip()
        assert len(prompt) > 20


def test_prompts_are_distinct():
    assert len(set(CORRECTNESS_PROMPTS)) == len(CORRECTNESS_PROMPTS)


def test_the_digest_is_pinned():
    """Changing the gate's inputs invalidates comparisons with every previously recorded
    result. That has to be a deliberate act, so an accidental edit fails here."""
    assert digest() == PROMPT_DIGEST


def test_the_digest_actually_detects_a_change():
    changed = (*CORRECTNESS_PROMPTS[:-1], CORRECTNESS_PROMPTS[-1] + " ")
    assert digest(changed) != PROMPT_DIGEST


def test_the_digest_is_order_sensitive():
    reordered = tuple(reversed(CORRECTNESS_PROMPTS))
    assert digest(reordered) != PROMPT_DIGEST
