"""Speculative decoding: more than one token per weight-stream.

The weights are read once per forward pass, not once per token, so a pass that verifies k
drafted tokens costs about what a pass producing one costs. Draft k tokens cheaply, verify
them in a single forward, keep every one the verifier's own argmax agrees with. The emitted
sequence is the one greedy decoding would have produced, because every token emitted is an
argmax of logits this model computed.

`docs/superpowers/specs/2026-09-24-speculative-decoding-design.md` has the cost model, the
kill criteria and the reason the recurrent state is the hard part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

__all__ = [
    "SPECULATIVE_CACHE_HEADROOM",
    "AcceptanceRecord",
    "Drafter",
    "FixedTokenDrafter",
    "NgramDrafter",
    "SpeculativeLoop",
    "install_speculative_loop",
]


#: Extra cache positions a speculative decode column needs beyond `context + max_new_tokens`.
#:
#: The reference step path writes exactly one position per call, so `context + decode_tokens`
#: is exactly what it consumes. This loop does not: a verify writes `block_size + 1` positions
#: before `SpeculativeLoop.__call__` rewinds the cache to however many of them were kept, and
#: the *final* cycle can commit up to `block_size` tokens past the `max_new_tokens` budget
#: before the caller's `[:, :max_new_tokens]` slice truncates the returned sequence — the cache
#: itself is written before that truncation happens. Reproduced by the controller at
#: `context=32, tokens=16`: with no headroom, `k=2` overflowed at "46 + 3 > max_seq_len=48".
#: 16 covers every block size this project has registered (max 4) with room to spare.
SPECULATIVE_CACHE_HEADROOM = 16


class Drafter(Protocol):
    """Proposes `k` tokens and is told what was committed. Always proposes exactly `k`.

    Exactly `k` matters more than it looks: the verify tensor's shape is `k+1`, and a
    drafter that sometimes proposed fewer would hand dynamo a second shape to compile.
    """

    def propose(self, committed: Tensor, k: int) -> Tensor: ...

    def commit(self, tokens: Tensor) -> None: ...


class FixedTokenDrafter:
    """Proposes one id, forever. Acceptance ~0 by construction.

    This is an instrument, not a candidate: it measures `gamma(k)`, the cost of a
    `k+1`-token verify, with no drafter quality in the number at all. Its slot is
    registered as a predicted loss and the ratio it returns is `1/gamma`.
    """

    def __init__(self, token_id: int = 0) -> None:
        self.token_id = token_id

    def propose(self, committed: Tensor, k: int) -> Tensor:
        return torch.full((committed.shape[0], k), self.token_id, dtype=torch.long, device=committed.device)

    def commit(self, tokens: Tensor) -> None:
        return None


class NgramDrafter:
    """Prompt-lookup drafting: the continuation that followed these `n` tokens last time.

    Costs no weights and no forward pass, so `d = 0` and the whole downside is `gamma - 1`:
    spec §1 puts break-even at about 10% acceptance. It is also the drafter whose
    acceptance depends most on the prompt, which is why the benchmark grows a text
    workload — on the random token ids the headline workload generates, this measures
    something that is not decoding.
    """

    def __init__(self, n: int = 3, fill_token: int = 0) -> None:
        self.n = n
        self.fill_token = fill_token

    def propose(self, committed: Tensor, k: int) -> Tensor:
        batch = committed.shape[0]
        out = torch.full((batch, k), self.fill_token, dtype=torch.long, device=committed.device)
        for row in range(batch):
            ids = committed[row].tolist()
            if len(ids) <= self.n:
                continue
            needle = ids[-self.n :]
            # Search backwards from the most recent occurrence that is not the tail itself.
            for start in range(len(ids) - self.n - 1, -1, -1):
                if ids[start : start + self.n] != needle:
                    continue
                follow = ids[start + self.n : start + self.n + k]
                if not follow:
                    continue
                out[row, : len(follow)] = torch.tensor(follow, dtype=torch.long, device=committed.device)
                break
        return out

    def commit(self, tokens: Tensor) -> None:
        return None


@dataclass
class AcceptanceRecord:
    """How many drafts each cycle kept. The diagnostic without which a ratio is unreadable.

    `034-static-cache-cudagraphs` won 2.0% for a mechanism that never fired, and the slot
    could not say so because it recorded only a ratio. A speculative slot that came back at
    0.95 could be a bad drafter or an expensive verify, and those license opposite next
    steps: this separates them.
    """

    block_size: int
    accepted: list[int] = field(default_factory=list)

    def observe(self, accepted: int) -> None:
        self.accepted.append(accepted)

    def reset(self) -> None:
        """Discard every observation so far, keeping `block_size`.

        `install_speculative_loop` (Task 5) attaches this once, at candidate-build time, and
        the loop keeps observing into it for the rest of the candidate's life. `run_slot`
        (I2) runs `_run_correctness` -- which free-runs the candidate over five prompts of
        real tokenized text through `check_sequence` -- **before** the benchmark, on the same
        loop and the same record. Without a reset, the histogram this method exists to
        report mixes those correctness-gate cycles into the benchmark's own, and for the
        slots this project built the histogram for, most of the recorded cycles never came
        from the workload at all.
        """
        self.accepted.clear()

    @property
    def cycles(self) -> int:
        return len(self.accepted)

    @property
    def mean_accepted(self) -> float:
        return sum(self.accepted) / len(self.accepted) if self.accepted else 0.0

    @property
    def histogram(self) -> dict[int, int]:
        counts = {step: 0 for step in range(self.block_size + 1)}
        for value in self.accepted:
            counts[value] = counts.get(value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "cycles": self.cycles,
            "mean_accepted": self.mean_accepted,
            "histogram": {str(k): v for k, v in self.histogram.items()},
        }


class SpeculativeLoop:
    """Draft `k`, verify in one pass, keep the prefix the verifier agrees with.

    One cycle:

    1. the drafter proposes `k` tokens from what is committed;
    2. one forward over `[last_committed_token, draft_0 ... draft_{k-1}]` returns `k+1`
       logits — position `i` is the model's own answer for what follows the first `i`
       drafted tokens;
    3. accept the longest prefix where the drafts match those argmaxes, and emit the
       argmax at the first mismatch as well, which is always a correct token;
    4. rewind the cache by `k - j` and tell each layer which state version to keep.

    Step 3 is why the emitted sequence is greedy decoding's: every token emitted is an
    argmax of logits this model computed, and a rejected draft contributes nothing.
    """

    def __init__(self, drafter: Drafter, block_size: int, acceptance: AcceptanceRecord | None = None):
        if block_size < 0:
            raise ValueError(f"block_size must be non-negative, got {block_size}")
        self.drafter = drafter
        self.block_size = block_size
        self.acceptance = acceptance

    def __call__(self, runnable, input_ids: Tensor, max_new_tokens: int, cache) -> Tensor:
        from .kernels.rollback_state import rollback_states  # noqa: PLC0415

        states = rollback_states(runnable)
        logits, _ = runnable(input_ids, cache, num_logits_to_keep=1)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [token]
        self.drafter.commit(torch.cat([input_ids, token], dim=1))

        while sum(piece.shape[1] for piece in generated) < max_new_tokens:
            committed = torch.cat([input_ids, *generated], dim=1)
            if self.block_size == 0:
                logits, _ = runnable(token, cache, num_logits_to_keep=1)
                token = logits[:, -1].argmax(dim=-1, keepdim=True)
                generated.append(token)
                self.drafter.commit(token)
                continue

            drafts = self.drafter.propose(committed, self.block_size)
            verified = torch.cat([token, drafts], dim=1)
            before = cache.seq_len
            logits, _ = runnable(verified, cache, num_logits_to_keep=self.block_size + 1)
            proposals = logits.argmax(dim=-1)

            # The first position where the model's own answer differs from the draft. Batch
            # 1 is what this project measures, so this reads row 0 and is honest about it.
            accepted = 0
            while accepted < self.block_size and bool((proposals[:, accepted] == drafts[:, accepted]).all()):
                accepted += 1

            kept = torch.cat([drafts[:, :accepted], proposals[:, accepted : accepted + 1]], dim=1)
            # The verify wrote `block_size + 1` positions; `accepted + 1` of them survive.
            cache.rewind(before + accepted + 1)
            for state in states:
                state.keep(accepted + 1)
            if self.acceptance is not None:
                self.acceptance.observe(accepted)

            generated.append(kept)
            token = kept[:, -1:]
            self.drafter.commit(kept)

        return torch.cat(generated, dim=1)[:, :max_new_tokens]


def install_speculative_loop(
    model, drafter: Drafter, block_size: int, acceptance: AcceptanceRecord | None = None
) -> None:
    """Put the loop where `greedy_decode` will find it.

    The drafter is held **here**, on the loop, and never registered as a submodule.
    `cli._assert_parameters_are_shared` requires every candidate parameter to share storage
    with a reference parameter of the same name — the check that catches a candidate which
    silently loaded a second 8.4 GB copy of the weights — and draft weights have no
    counterpart. Keeping them off the module tree leaves that check exactly as strict.
    """
    model.decode_loop = SpeculativeLoop(drafter, block_size, acceptance)
