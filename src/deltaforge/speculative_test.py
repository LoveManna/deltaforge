"""Drafters: the cheapest half of the hypothesis, and the only half testable without a GPU."""

from __future__ import annotations

import torch

from .config import tiny_config
from .kernels.rollback_state import install_rollback_state
from .model import greedy_decode
from .reference import ReferenceModel
from .speculative import AcceptanceRecord, FixedTokenDrafter, NgramDrafter, install_speculative_loop


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


def test_a_block_size_of_zero_decodes_exactly_like_the_reference():
    """The diagnostic that separates a plumbing bug from a reduction-order flip. It must be
    bit-identical, and it runs here rather than on a rented card."""
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 6))

    plain = ReferenceModel(config).eval()
    spec = ReferenceModel(config).eval()
    spec.load_state_dict(plain.state_dict())
    install_rollback_state(spec)
    install_speculative_loop(spec, FixedTokenDrafter(token_id=3), block_size=0)

    with torch.no_grad():
        expected = greedy_decode(plain, ids, 8, cache=plain.new_cache(1, 32))
        got = greedy_decode(spec, ids, 8, cache=spec.new_cache(1, 32))

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_a_drafter_that_is_always_wrong_still_emits_the_reference_sequence():
    """Acceptance 0 is the worst case and it must still be *correct*, not merely slow."""
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 6))

    plain = ReferenceModel(config).eval()
    spec = ReferenceModel(config).eval()
    spec.load_state_dict(plain.state_dict())
    install_rollback_state(spec)
    install_speculative_loop(spec, FixedTokenDrafter(token_id=3), block_size=4)

    with torch.no_grad():
        expected = greedy_decode(plain, ids, 8, cache=plain.new_cache(1, 40))
        got = greedy_decode(spec, ids, 8, cache=spec.new_cache(1, 40))

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_a_perfect_drafter_is_accepted_every_time_and_emits_the_same_tokens():
    """An oracle drafter proves the accept path, which the always-wrong drafter never takes.

    The truth is decoded twice as long as the loop is asked to emit, so no cycle ever drafts
    past the end of it: a truth exactly `max_new_tokens` long would truncate the final
    cycle's draft, which would measure end-of-sequence behavior rather than the accept path
    this test exists to prove.
    """
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 6))

    plain = ReferenceModel(config).eval()
    with torch.no_grad():
        truth = greedy_decode(plain, ids, 16, cache=plain.new_cache(1, 48))
    expected = truth[:, :8]

    spec = ReferenceModel(config).eval()
    spec.load_state_dict(plain.state_dict())
    install_rollback_state(spec)
    record = AcceptanceRecord(block_size=4)
    install_speculative_loop(
        spec, _OracleDrafter(truth, prompt_len=ids.shape[1]), block_size=4, acceptance=record
    )

    with torch.no_grad():
        got = greedy_decode(spec, ids, 8, cache=spec.new_cache(1, 40))

    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert record.mean_accepted == 4.0
    assert record.cycles == 2


def test_the_loop_records_what_it_accepted():
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 6))
    spec = ReferenceModel(config).eval()
    install_rollback_state(spec)
    record = AcceptanceRecord(block_size=2)
    install_speculative_loop(spec, FixedTokenDrafter(token_id=3), block_size=2, acceptance=record)

    with torch.no_grad():
        greedy_decode(spec, ids, 7, cache=spec.new_cache(1, 40))

    # One token comes from the prefill step, which is never a cycle; each of the remaining
    # six tokens comes from its own cycle, all at acceptance 0.
    assert record.cycles == 6
    assert record.histogram[0] == 6


def test_reset_clears_observations_but_keeps_block_size():
    """I2: `run_slot` reuses this record's `block_size` after clearing it, so a reset that
    also lost `block_size` would corrupt the histogram it is meant to make trustworthy."""
    record = AcceptanceRecord(block_size=4)
    record.observe(4)
    record.observe(2)

    record.reset()

    assert record.accepted == []
    assert record.cycles == 0
    assert record.mean_accepted == 0.0
    assert record.block_size == 4
    assert record.histogram == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}


class _OracleDrafter:
    """Drafts the reference's own continuation. Only a test fixture: it has the answer.

    Its position in `truth` is derived from the loop's own `committed` argument rather than
    from an internal counter: the loop's first `commit` call passes the prompt plus the
    first generated token, so a counter driven by `commit` alone would start offset by the
    prompt's length. `committed` always holds `prompt_len` prompt tokens followed by
    whatever has been generated so far, so `committed.shape[1] - prompt_len` is exactly the
    number of tokens of `truth` already generated, with no bookkeeping in `commit` at all.
    """

    def __init__(self, truth, prompt_len):
        self.truth = truth
        self.prompt_len = prompt_len

    def propose(self, committed, k):
        start = committed.shape[1] - self.prompt_len
        window = self.truth[:, start : start + k]
        if window.shape[1] < k:
            pad = torch.zeros((window.shape[0], k - window.shape[1]), dtype=torch.long)
            window = torch.cat((window, pad), dim=1)
        return window

    def commit(self, tokens):
        return None
