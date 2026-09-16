"""The two correctness gates. A fast kernel that is wrong is worse than no kernel.

**Layer 1, per kernel.** `allclose` against the corresponding reference operation at bf16
tolerance, on shapes drawn from the real model configuration. The result is never a bare
bool: max absolute and max relative error are carried as numbers, because the magnitude
is informative even when the gate passes. A kernel that passes at 9e-3 against a 1e-2
tolerance is one shape change away from failing, and the record should say so.

**Layer 2, end to end.** Greedy-decode 128 tokens from the five fixed prompts and require
an exact token-sequence match against eager. This catches the case where every kernel
passes in isolation but the assembled pipeline accumulates drift — which is the failure
mode layer 1 structurally cannot see.

**Layer 2, approximate.** Exact tokens are the right gate for a kernel that claims to
compute the same function. They are the *wrong* gate for a kernel that deliberately
computes a different one: a weight-only quantised candidate fails an exact match by
construction, not by defect, and recording that as `incorrect` would say the kernel is
broken when what it is, is approximate. `check_distribution` is the alternative
`docs/HYPOTHESES.md` specifies — teacher-force both models over the reference's own greedy
continuation and score top-1 agreement and mean KL — and the thresholds it is gated on are
registered per hypothesis in `batches.py` before the rental, because a threshold chosen
after seeing the number is not a gate.

Failures are recorded, not discarded. A candidate that was fast but wrong is among the
most valuable things a future session can read.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from ..model import greedy_decode
from ..reference import ReferenceModel

__all__ = [
    "CorrectnessReport",
    "DistributionCheck",
    "EndToEndCheck",
    "KernelCheck",
    "TokenMatch",
    "check_distribution",
    "check_end_to_end",
    "check_kernel",
    "error_magnitudes",
]

DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2


@dataclass(frozen=True)
class KernelCheck:
    """Layer 1 result for one kernel against one input shape."""

    name: str
    replaces: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    rtol: float
    atol: float
    shape: tuple[int, ...]
    dtype: str
    #: Elements excluded from the relative-error statistic because the reference value
    #: was at or below ``atol``, where a relative error is not meaningful.
    excluded_near_zero: int = 0
    total_elements: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "replaces": self.replaces,
            "passed": self.passed,
            "max_abs_err": self.max_abs_err,
            "max_rel_err": self.max_rel_err,
            "rtol": self.rtol,
            "atol": self.atol,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "excluded_near_zero": self.excluded_near_zero,
            "total_elements": self.total_elements,
            "note": self.note,
        }


def error_magnitudes(
    reference: torch.Tensor, candidate: torch.Tensor, atol: float = DEFAULT_ATOL
) -> tuple[float, float, int, int]:
    """Return ``(max_abs_err, max_rel_err, excluded_near_zero, total_elements)``.

    Relative error is measured only where ``|reference| > atol``. Below that the
    denominator is noise and a "relative error" of 10^6 says nothing about correctness —
    the absolute error already covers those elements. The count of excluded elements is
    reported so the statistic cannot hide a mostly-zero tensor.
    """
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    ref = reference.detach().float()
    cand = candidate.detach().float()
    abs_err = (cand - ref).abs()
    total = ref.numel()

    max_abs = float(abs_err.max()) if total else 0.0
    significant = ref.abs() > atol
    excluded = int(total - int(significant.sum()))
    max_rel = float((abs_err[significant] / ref.abs()[significant]).max()) if bool(significant.any()) else 0.0
    return max_abs, max_rel, excluded, total


def check_kernel(
    name: str,
    reference_fn: Callable[..., torch.Tensor | Sequence[torch.Tensor]],
    candidate_fn: Callable[..., torch.Tensor | Sequence[torch.Tensor]],
    args: Sequence[object] = (),
    kwargs: dict[str, object] | None = None,
    *,
    replaces: str = "",
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    note: str = "",
) -> KernelCheck:
    """Run both implementations on the same inputs and record how far apart they are.

    Multi-output operations are supported: every output is compared and the worst
    magnitude across them is reported.
    """
    kwargs = dict(kwargs or {})
    with torch.no_grad():
        ref_out = reference_fn(*args, **kwargs)
        cand_out = candidate_fn(*args, **kwargs)

    ref_tensors = _as_tensors(ref_out)
    cand_tensors = _as_tensors(cand_out)
    if len(ref_tensors) != len(cand_tensors):
        raise ValueError(
            f"{name}: reference returned {len(ref_tensors)} tensors, candidate returned {len(cand_tensors)}"
        )

    worst_abs = 0.0
    worst_rel = 0.0
    excluded = 0
    total = 0
    passed = True
    for ref, cand in zip(ref_tensors, cand_tensors):
        max_abs, max_rel, exc, tot = error_magnitudes(ref, cand, atol=atol)
        worst_abs = max(worst_abs, max_abs)
        worst_rel = max(worst_rel, max_rel)
        excluded += exc
        total += tot
        passed = passed and bool(
            torch.allclose(cand.detach().float(), ref.detach().float(), rtol=rtol, atol=atol)
        )

    first = ref_tensors[0]
    return KernelCheck(
        name=name,
        replaces=replaces,
        passed=passed,
        max_abs_err=worst_abs,
        max_rel_err=worst_rel,
        rtol=rtol,
        atol=atol,
        shape=tuple(first.shape),
        dtype=str(first.dtype),
        excluded_near_zero=excluded,
        total_elements=total,
        note=note,
    )


def _as_tensors(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, torch.Tensor):
        return (out,)
    if isinstance(out, (tuple, list)):
        tensors = tuple(x for x in out if isinstance(x, torch.Tensor))
        if not tensors:
            raise TypeError("operation returned no tensors to compare")
        return tensors
    raise TypeError(f"cannot compare a {type(out).__name__}")


@dataclass(frozen=True)
class TokenMatch:
    """Layer 2 result for one prompt."""

    prompt_index: int
    matched: bool
    num_tokens: int
    first_divergence: int | None
    reference_tokens: tuple[int, ...]
    candidate_tokens: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_index": self.prompt_index,
            "matched": self.matched,
            "num_tokens": self.num_tokens,
            "first_divergence": self.first_divergence,
            "reference_tokens": list(self.reference_tokens),
            "candidate_tokens": list(self.candidate_tokens),
        }


@dataclass(frozen=True)
class EndToEndCheck:
    max_new_tokens: int
    prompt_digest: str
    per_prompt: tuple[TokenMatch, ...] = ()

    @property
    def passed(self) -> bool:
        return all(match.matched for match in self.per_prompt)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "max_new_tokens": self.max_new_tokens,
            "prompt_digest": self.prompt_digest,
            "num_prompts": len(self.per_prompt),
            "per_prompt": [m.to_dict() for m in self.per_prompt],
        }


def check_end_to_end(
    reference_model: ReferenceModel,
    candidate_model: ReferenceModel,
    prompt_token_ids: Sequence[Sequence[int]],
    *,
    max_new_tokens: int = 128,
    prompt_digest: str = "",
    device: torch.device | str | None = None,
) -> EndToEndCheck:
    """Greedy-decode each prompt through both models; require identical token sequences.

    Prompts arrive as token ids rather than strings: tokenization belongs to the caller,
    which keeps this module dependent on torch and the reference alone.
    """
    if not prompt_token_ids:
        raise ValueError("no prompts supplied to the end-to-end gate")
    device = device or next(reference_model.parameters()).device

    matches: list[TokenMatch] = []
    for index, ids in enumerate(prompt_token_ids):
        if len(ids) == 0:
            raise ValueError(f"prompt {index} is empty")
        input_ids = torch.tensor([list(ids)], dtype=torch.long, device=device)
        ref_tokens = greedy_decode(reference_model, input_ids, max_new_tokens)[0].tolist()
        cand_tokens = greedy_decode(candidate_model, input_ids, max_new_tokens)[0].tolist()

        divergence = next((i for i, (a, b) in enumerate(zip(ref_tokens, cand_tokens)) if a != b), None)
        if divergence is None and len(ref_tokens) != len(cand_tokens):
            divergence = min(len(ref_tokens), len(cand_tokens))
        matches.append(
            TokenMatch(
                prompt_index=index,
                matched=divergence is None,
                num_tokens=len(ref_tokens),
                first_divergence=divergence,
                reference_tokens=tuple(ref_tokens),
                candidate_tokens=tuple(cand_tokens),
            )
        )

    return EndToEndCheck(
        max_new_tokens=max_new_tokens,
        prompt_digest=prompt_digest,
        per_prompt=tuple(matches),
    )


@dataclass(frozen=True)
class DistributionCheck:
    """Layer 2 for a candidate that computes a deliberately different function.

    Two statistics, both over the same teacher-forced positions:

    ``top1_agreement`` — the fraction of positions where the candidate's argmax equals the
    reference's. This is the number that says what the quantised model would actually
    *emit*, and it is measured teacher-forced rather than by free-running decode on
    purpose: once two decoders disagree once they are reading different text, so a
    free-running agreement rate measures the first divergence and then nothing.

    ``mean_kl`` — mean ``KL(P_reference || P_candidate)`` in nats. Agreement alone can hide
    a distribution that has been shredded everywhere the argmax happens to be safe; KL sees
    that and an argmax cannot.

    Both thresholds are inputs, registered on the hypothesis before the rental.
    """

    top1_agreement: float
    mean_kl: float
    max_kl: float
    num_positions: int
    top1_threshold: float
    kl_threshold: float
    prompt_digest: str = ""
    context_tokens: int = 0

    @property
    def passed(self) -> bool:
        return self.top1_agreement >= self.top1_threshold and self.mean_kl <= self.kl_threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": "approximate",
            "passed": self.passed,
            "top1_agreement": self.top1_agreement,
            "mean_kl": self.mean_kl,
            "max_kl": self.max_kl,
            "num_positions": self.num_positions,
            "top1_threshold": self.top1_threshold,
            "kl_threshold": self.kl_threshold,
            "prompt_digest": self.prompt_digest,
            "context_tokens": self.context_tokens,
        }


def _kl_and_agreement(
    ref_logits: torch.Tensor, cand_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(kl_per_position, argmax_matches)`` for one prompt, computed in fp32.

    In fp32 because a KL of 1e-3 nats is the interesting regime and bf16 cannot represent
    the difference of two log-probabilities at that scale — the statistic would be
    quantisation noise about quantisation noise.
    """
    ref = torch.log_softmax(ref_logits.float(), dim=-1)
    cand = torch.log_softmax(cand_logits.float(), dim=-1)
    kl = (ref.exp() * (ref - cand)).sum(dim=-1)
    matches = ref.argmax(dim=-1) == cand.argmax(dim=-1)
    return kl.reshape(-1), matches.reshape(-1)


def check_distribution(
    reference_model: ReferenceModel,
    candidate_model: ReferenceModel,
    prompt_token_ids: Sequence[Sequence[int]],
    *,
    top1_threshold: float,
    kl_threshold: float,
    max_new_tokens: int = 128,
    prompt_digest: str = "",
    device: torch.device | str | None = None,
) -> DistributionCheck:
    """Teacher-force both models over the reference's own continuation and compare.

    The context is ``prompt + reference.greedy_decode(prompt)``, so the positions scored
    are the ones the reference would really have visited — prompt text alone is a few
    dozen positions of a distribution the model has not yet committed to, and the decode
    regime is what the benchmark measures.
    """
    if not prompt_token_ids:
        raise ValueError("no prompts supplied to the distribution gate")
    device = device or next(reference_model.parameters()).device

    kls: list[torch.Tensor] = []
    matches: list[torch.Tensor] = []
    context_tokens = 0
    for index, ids in enumerate(prompt_token_ids):
        if len(ids) == 0:
            raise ValueError(f"prompt {index} is empty")
        input_ids = torch.tensor([list(ids)], dtype=torch.long, device=device)
        continuation = greedy_decode(reference_model, input_ids, max_new_tokens)
        context = torch.cat((input_ids, continuation), dim=1)
        context_tokens += int(context.shape[1])

        with torch.no_grad():
            ref_logits, _ = reference_model(context)
            cand_logits, _ = candidate_model(context)
        kl, match = _kl_and_agreement(ref_logits, cand_logits)
        kls.append(kl)
        matches.append(match)

    all_kl = torch.cat(kls)
    all_matches = torch.cat(matches)
    return DistributionCheck(
        top1_agreement=float(all_matches.float().mean()),
        mean_kl=float(all_kl.mean()),
        max_kl=float(all_kl.max()),
        num_positions=int(all_kl.numel()),
        top1_threshold=top1_threshold,
        kl_threshold=kl_threshold,
        prompt_digest=prompt_digest,
        context_tokens=context_tokens,
    )


@dataclass(frozen=True)
class CorrectnessReport:
    """Both gates together. ``passed`` requires both."""

    kernel_checks: tuple[KernelCheck, ...] = ()
    end_to_end: EndToEndCheck | None = None
    #: Set instead of ``end_to_end`` for a hypothesis whose candidate is approximate by
    #: design. Exactly one of the two is populated; a report with neither has no layer 2.
    distribution: DistributionCheck | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        layer1 = all(check.passed for check in self.kernel_checks)
        layer2 = True
        if self.end_to_end is not None:
            layer2 = layer2 and self.end_to_end.passed
        if self.distribution is not None:
            layer2 = layer2 and self.distribution.passed
        return layer1 and layer2

    @property
    def worst_max_abs_err(self) -> float:
        return max((c.max_abs_err for c in self.kernel_checks), default=0.0)

    @property
    def worst_max_rel_err(self) -> float:
        return max((c.max_rel_err for c in self.kernel_checks), default=0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "worst_max_abs_err": self.worst_max_abs_err,
            "worst_max_rel_err": self.worst_max_rel_err,
            "layer1_kernel_checks": [c.to_dict() for c in self.kernel_checks],
            "layer2_end_to_end": self.end_to_end.to_dict() if self.end_to_end else None,
            "layer2_distribution": self.distribution.to_dict() if self.distribution else None,
            "layer2_policy": "approximate" if self.distribution is not None else "exact",
            "metadata": dict(self.metadata),
        }
