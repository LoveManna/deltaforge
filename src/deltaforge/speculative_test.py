"""Drafters: the cheapest half of the hypothesis, and the only half testable without a GPU."""

from __future__ import annotations

import torch

from .speculative import FixedTokenDrafter, NgramDrafter


def test_a_fixed_drafter_proposes_the_same_token_every_time():
    """The instrument for measuring `gamma`: acceptance is ~0 by construction, so the slot
    measures what a k+1-token verify costs and nothing else. Spec §6, slot `a`."""
    drafter = FixedTokenDrafter(token_id=7)
    drafter.commit(torch.tensor([[1, 2, 3]]))

    proposal = drafter.propose(torch.tensor([[1, 2, 3]]), k=4)

    assert proposal.shape == (1, 4)
    assert proposal.unique().tolist() == [7]


def test_an_ngram_drafter_copies_the_continuation_of_the_last_match():
    drafter = NgramDrafter(n=2)
    drafter.commit(torch.tensor([[5, 6, 7, 8, 9, 5, 6]]))

    proposal = drafter.propose(torch.tensor([[5, 6, 7, 8, 9, 5, 6]]), k=3)

    assert proposal.tolist() == [[7, 8, 9]]


def test_an_ngram_drafter_with_no_match_still_proposes_k_tokens():
    """A drafter that returned fewer would make the verify shape vary, and a varying shape
    recompiles: batch 008 lost six of nine slots to recompilation."""
    drafter = NgramDrafter(n=2)
    drafter.commit(torch.tensor([[1, 2, 3]]))

    proposal = drafter.propose(torch.tensor([[1, 2, 3]]), k=4)

    assert proposal.shape == (1, 4)


def test_an_ngram_drafter_never_proposes_from_a_match_at_the_very_end():
    """The last n tokens always match themselves; drafting from that proposes nothing."""
    drafter = NgramDrafter(n=2)
    history = torch.tensor([[4, 1, 2]])
    drafter.commit(history)

    proposal = drafter.propose(history, k=2)

    assert proposal.shape == (1, 2)
