"""The five fixed prompts behind the layer-2 correctness gate.

Checked into the repo so the gate is reproducible: a gate whose inputs vary between
sessions cannot distinguish "this kernel drifts" from "we asked a different question".
Changing this list invalidates comparisons with every previously recorded result, so
``prompts_test.py`` pins the digest — an accidental edit fails the test suite rather
than quietly weakening the gate.

They are deliberately varied in kind (factual, code, instruction-following, arithmetic,
open-ended continuation) so that the 128 greedy-decoded tokens exercise different regions
of the distribution rather than five paraphrases of one thing.
"""

from __future__ import annotations

import hashlib

__all__ = ["CORRECTNESS_PROMPTS", "PROMPT_DIGEST", "digest"]

CORRECTNESS_PROMPTS: tuple[str, ...] = (
    "Explain in plain language why a memory-bound GPU kernel does not get faster when you add more arithmetic.",
    "Write a Python function that merges two sorted lists into one sorted list, without using sorted().",
    "List three reasons a benchmark that reports only the mean of ten runs can be misleading.",
    "A train leaves at 09:15 and arrives at 13:40. How long is the journey? Show the arithmetic.",
    "The old lighthouse keeper had not spoken to anyone in eleven years, and then one morning",
)


def digest(prompts: tuple[str, ...] = CORRECTNESS_PROMPTS) -> str:
    """A stable content hash, so a changed gate input is visible in a diff and a test."""
    hasher = hashlib.sha256()
    for prompt in prompts:
        hasher.update(prompt.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


#: Pinned digest of :data:`CORRECTNESS_PROMPTS`. Update only alongside a deliberate
#: decision to invalidate cross-session correctness comparisons.
PROMPT_DIGEST = "7e7f95150a170228c8ff7e77eed800ca9d92cc6be7e9747958b41dae19e25cfd"
