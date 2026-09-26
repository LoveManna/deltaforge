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
    "AcceptanceRecord",
    "Drafter",
    "FixedTokenDrafter",
    "NgramDrafter",
]


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
