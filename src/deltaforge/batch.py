"""Batches: many hypotheses measured on one rental.

A rental's cost is roughly 15 minutes of fixed setup — container image, torch, a 9.32 GB
checkpoint, the GPU test suite — plus 2-4 minutes per hypothesis. Testing one hypothesis
per rental pays the fixed cost to buy a single measurement. Batching amortises it across
7-12, and `docs/superpowers/specs/2026-09-06-batched-hypotheses-design.md` has the
arithmetic.

**Nothing here imports torch.** The batch model, the outcome arithmetic and the deadline
policy are the parts most worth testing exhaustively, and they are all decidable on a
CPU with no checkpoint. `cli.cmd_batch` holds the part that needs a GPU.

The three ideas in this module:

* A **hypothesis is data**, not global registry state. `kernels.REGISTRY` is a
  process-wide singleton with a "one champion per operation" invariant, so it can express
  exactly one candidate per process. A batch needs N, so `scoped_registry` builds a fresh
  registry per hypothesis and the global one goes back to being a catalogue of what
  exists rather than a statement about what is under test.

* A hypothesis carries its **prediction, registered before the run**. `AGENT.md` §1 says
  the finding is a mechanistic account stated in advance and then confirmed — that being
  right in advance *is* the result. Until now nothing in the repo recorded a prediction
  anywhere it could be scored. `score_predictions` scores them.

* The batch **ends itself** rather than being killed. A watchdog firing is a reportable
  fault; `SlotBudget` stops the batch while its results are written and the instance is
  still healthy.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

from .kernels import REGISTRY, KernelRegistry, KernelStatus, RegistryError

__all__ = [
    "BATCH_OUTCOMES",
    "COLD_PHASE_ESTIMATES",
    "CORRECTNESS_POLICIES",
    "PREDICTIONS",
    "Batch",
    "Hypothesis",
    "Precondition",
    "PredictionScore",
    "SlotBudget",
    "calibration_holds",
    "classify_outcome",
    "precondition_holds",
    "registry_for",
    "score_predictions",
    "scoped_registry",
    "session_fits_one_hypothesis",
    "unpaired_slots",
]


#: How a hypothesis's layer-2 gate is scored.
#:
#: `exact` — the candidate claims to compute the same function as the reference, so its
#:           greedy token sequence must match it exactly. The right gate for every kernel
#:           this project has written so far.
#: `approximate` — the candidate deliberately computes a *different* function, and a
#:           quantised one cannot match bf16 tokens however correct it is. Scored on top-1
#:           agreement and mean KL against thresholds this hypothesis registers below.
#:           `harness.correctness.check_distribution` measures them.
#: `sequence` — the candidate carries the reference's *own* weights but replaces the decode
#:           loop, so `approximate`'s teacher-forcing cannot fail it and `exact`'s bit-exact
#:           match cannot pass it: a multi-token verify reduces in a different order from
#:           single-token decode and lands off by a ULP on this checkpoint. Scored on where
#:           the free-run token sequences first diverge and how confident the reference was
#:           there, against the `divergence_gap_ceiling` this hypothesis registers below.
#:           `harness.correctness.check_sequence` measures it.
CORRECTNESS_POLICIES = ("exact", "approximate", "sequence")

#: What a hypothesis may predict, recorded in the manifest before the rental.
#:
#: `identity` is not a hedge — it is the strongest claim in the set. It says the candidate
#: is bit-identical to the reference and must therefore measure 1.00 within the noise
#: band. A miss there means the harness is measuring something other than what it says,
#: which invalidates every other slot in the batch.
PREDICTIONS = ("win", "loss", "inconclusive", "identity")

#: Outcomes a slot may reach. Extends `report.OUTCOMES` with the two states that only
#: exist once hypotheses share a rental.
#:
#: `error` — the kernel raised. Under one-hypothesis-per-rental this ended the session;
#:           here it costs a slot, and the traceback is itself worth reading.
#: `not_run` — the deadline arrived first. Explicitly recorded, never silently omitted:
#:           a hypothesis missing from a batch record must be distinguishable from one
#:           that ran and produced nothing.
#: `starved` — the clock ended the rental before any slot scored. Distinct from `not_run`
#:           on purpose: `not_run` means the batch stopped early having already measured
#:           something, `starved` means the rental produced nothing and the session gate
#:           is the reason. It is the evidence for raising that gate.
#: `precondition_failed` — an earlier slot's ratio settled this one, so it was not run.
#:           Distinct from `not_run` again: the clock did not arrive, the batch decided.
#:           Flattening the two would erase the evidence for whether the floor was right.
BATCH_OUTCOMES = (
    "win",
    "loss",
    "inconclusive",
    "incorrect",
    "error",
    "not_run",
    "starved",
    "precondition_failed",
)

#: Outcomes that tested no prediction, so scoring one either way would be a fiction.
UNSCORED_OUTCOMES = ("error", "not_run", "starved", "precondition_failed")


class Precondition(NamedTuple):
    """A floor an earlier slot must clear for this slot to be worth running.

    Batch 003 is why. Its slot 1 measured a hand-written GEMV against cuBLAS on identical
    bytes and returned 0.2801, which settled every quantisation slot behind it: a kernel at
    28% of the baseline's byte rate cannot collect a byte saving. Five slots then re-measured
    that fact at different bit widths. The batch could not stop, and the ordering rule that
    put the informative slot first had no way to act on what it found.

    ``versus`` puts the floor on a **comparison** -- ``ratio[slug] - ratio[versus]`` -- rather
    than on an absolute ratio. Rental 46 is why that exists. `061-int4-mlp-torch-dequant`,
    52.75% of per-token bytes and the largest prize in the backlog, depended on the
    proposition "inductor can fuse a grouped dequantisation into a GEMV prologue", whose
    measurement is `056` against `054`: **+3.2%, nowhere near any band**. It was gated on
    `056` reaching 1.02 *absolute*, `056` returned 1.0171, and the slot declined by 0.3% on a
    quantity that depends on what fraction of the step the head happens to be and on how fast
    the card is that hour. A precondition must name the proposition its slot depends on, and
    where that proposition is a comparison, the floor belongs on the comparison.
    """

    slug: str
    floor: float
    reason: str
    #: When set, the floor is on ``ratio[slug] - ratio[versus]``, and both slots must be
    #: earlier in the batch. A margin between two slots measured in the same process on the
    #: same card is the one quantity here that does not move with the hour.
    versus: str | None = None

    @property
    def slugs(self) -> tuple[str, ...]:
        """Every slot this precondition reads. All of them must run before the gated slot."""
        return (self.slug,) if self.versus is None else (self.slug, self.versus)

    def describe(self, ratios_by_slug: dict[str, float | None]) -> str:
        """What was read and what was needed -- the whole content of a `precondition_failed`.

        A declined slot leaves one line in the record, and rental 46's could not be read
        against the proposition it was protecting without redoing the arithmetic by hand.
        This writes the arithmetic down.
        """

        def seen(slug: str) -> str:
            value = ratios_by_slug.get(slug)
            return "no ratio" if value is None else f"{value:.4f}"

        if self.versus is None:
            return f"{self.slug} measured {seen(self.slug)} against a floor of {self.floor}"
        left = ratios_by_slug.get(self.slug)
        right = ratios_by_slug.get(self.versus)
        margin = f" = {left - right:+.4f}" if left is not None and right is not None else ""
        return (
            f"{self.slug} - {self.versus} measured {seen(self.slug)} - {seen(self.versus)}"
            f"{margin} against a floor of {self.floor}"
        )


def precondition_holds(precondition: Precondition | None, ratios_by_slug: dict[str, float | None]) -> bool:
    """Fails closed. A slot that errored has no ratio, and no ratio is not a good one."""
    if precondition is None:
        return True
    ratio = ratios_by_slug.get(precondition.slug)
    if ratio is None:
        return False
    if precondition.versus is None:
        return ratio >= precondition.floor
    against = ratios_by_slug.get(precondition.versus)
    return against is not None and (ratio - against) >= precondition.floor


@dataclass(frozen=True)
class Hypothesis:
    """One hypothesis: what to install, what it attacks, and what we predict.

    ``kernels`` names entries in the kernel registry. An empty tuple is the *identity
    champion* — a candidate that installs nothing and is therefore bit-identical to the
    reference. That is the harness's calibration instrument, and it must come first in
    any batch that means to be believed.
    """

    slug: str
    kernels: tuple[str, ...]
    category: str  # "A" | "B" | "C" | "calibration" — see docs/HYPOTHESES.md
    byte_share: float  # share of per-token bytes attacked, from docs/roofline.py
    mechanism: str  # one sentence: how it wins
    prediction: str
    rationale: str  # why we predict that, written before the measurement
    replaces: tuple[str, ...] = ()
    notes: str = ""
    #: `exact` or `approximate` — see `CORRECTNESS_POLICIES`.
    correctness: str = "exact"
    #: Only for `approximate`. The floor on teacher-forced top-1 agreement with the
    #: reference, and the ceiling on mean KL in nats. Registered here, before the rental,
    #: for the same reason the prediction is: a bar moved after seeing the number is not a
    #: bar. Both are read by `batch_run.BatchRunner._run_correctness`.
    top1_threshold: float | None = None
    kl_threshold: float | None = None
    #: How many teacher-forced positions the batch expects to score this hypothesis over.
    #: Declaring it is what lets `__post_init__` refuse a bar the sample cannot resolve.
    #: `None` means undeclared, and an undeclared n is not checked: inventing one would be
    #: worse than not checking.
    correctness_positions: int | None = None
    #: Only for `sequence`. The largest top-2 logit gap at which a token divergence is still
    #: attributable to reduction order rather than to a bug. Registered before the rental
    #: for the same reason every other bar is.
    divergence_gap_ceiling: float | None = None
    #: Registered before 2026-09-17, when `exact` was still accepted for a kernel that
    #: computes the same *function* as the reference. Batch 003 proved that is not the same
    #: property as producing the same *bits*, and the gate is now refused — but batches 001
    #: to 003 ran under it, and rewriting a gate a rental already ran under would falsify
    #: the record exactly as rewriting a prediction would. This flag keeps those manifests
    #: constructible and says why. **Never set it on a new hypothesis**; `batches_test`
    #: asserts that nothing after batch 003 does.
    historical_exact_gate: bool = False
    #: A floor an earlier slot in the same batch must clear for this one to be worth
    #: running. `None` means the slot runs whenever the clock allows.
    requires: Precondition | None = None
    #: The earlier slot this one differs from **by one variable**, named so the batch says
    #: what the comparison is before it is run.
    #:
    #: Every finding this project holds came from a pair of slots measured in one process
    #: that differ in exactly one thing: `044` against `045` (the four taps as a custom op
    #: or as torch, 37%) and `054` against `056` (the same dequantise-GEMV as a kernel or as
    #: torch, 3.2%). Every wasted rental came from a slot whose comparison was implicit and
    #: reconstructed afterwards: three rentals reasoned about a composition effect between
    #: two installers because no slot had measured the conv alone.
    #:
    #: `None` means "this slot is an ingredient measured alone", which `unpaired_slots`
    #: checks rather than takes on trust: a slot that re-installs a kernel an earlier slot
    #: already measured is a comparison whether or not it says so.
    contrast_with: str | None = None
    #: Which weight regions this candidate re-encodes, and at how many bits — the keys
    #: `harness.bytes_model.WEIGHT_REGIONS` names, plus the alias `layers`. The bench
    #: divides bytes by time, so a hypothesis that did not declare this would be scored
    #: against bf16 byte counts and report a bandwidth it never achieved. Empty means the
    #: candidate streams the reference's own bytes, which is what makes a bf16 control a
    #: control.
    weight_bits: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.correctness not in CORRECTNESS_POLICIES:
            raise ValueError(f"correctness must be one of {CORRECTNESS_POLICIES}, got {self.correctness!r}")
        # Checked ahead of the `approximate` block below, not merged into it: that block's
        # own "carries approximate thresholds" branch fires whenever a threshold is set and
        # the policy is not `approximate`, which would misreport a `sequence` hypothesis
        # carrying them as though it were `exact`. Raising here first pre-empts that.
        if self.correctness == "sequence":
            if self.divergence_gap_ceiling is None:
                raise ValueError(
                    f"{self.slug!r} is gated on its token sequence and must register a "
                    "divergence_gap_ceiling before the rental. Without one the gate passes "
                    "any divergence, including a corrupted recurrent state."
                )
            if self.top1_threshold is not None or self.kl_threshold is not None:
                raise ValueError(
                    f"{self.slug!r} is gated 'sequence' but carries teacher-forced bars. "
                    "check_distribution never calls the decode loop, and a candidate that "
                    "replaces the loop while sharing the reference's weights scores "
                    "perfectly on it whatever it did."
                )
        elif self.divergence_gap_ceiling is not None:
            raise ValueError(
                f"{self.slug!r} registers a divergence_gap_ceiling but is gated "
                f"{self.correctness!r}, which never reads it."
            )
        if self.correctness == "approximate":
            if self.top1_threshold is None or self.kl_threshold is None:
                raise ValueError(
                    f"{self.slug!r} is scored approximately and must register both a "
                    "top1_threshold and a kl_threshold before the rental. An approximate "
                    "gate with no bar passes everything, including a broken kernel."
                )
            if not 0.0 <= self.top1_threshold <= 1.0:
                raise ValueError(f"top1_threshold is a fraction, got {self.top1_threshold!r}")
            if self.kl_threshold < 0.0:
                raise ValueError(f"kl_threshold is in nats and cannot be negative, got {self.kl_threshold!r}")
            self._check_top1_resolution()
        elif self.top1_threshold is not None or self.kl_threshold is not None:
            raise ValueError(
                f"{self.slug!r} is scored exactly but carries approximate thresholds. "
                "Exact means the token sequences match; a threshold there would never be read."
            )
        if self.is_identity and self.correctness != "exact":
            raise ValueError(
                f"{self.slug!r} installs nothing, so it is bit-identical to the reference and "
                "must be gated exactly. An identity slot that could not match tokens would "
                "calibrate nothing."
            )
        if self.correctness == "exact" and not self.is_identity and not self.historical_exact_gate:
            raise ValueError(
                f"{self.slug!r} installs {self.kernels} and is gated 'exact', but only the "
                "identity champion is bit-identical to the reference. Computing the same "
                "function and producing the same bits are different properties: summing K in "
                "a different order from cuBLAS lands one bf16 ULP away, and one ULP flips an "
                "argmax on this model. 009-gemv-bf16-control matched 1 of 5 prompts on exactly "
                "this. Gate it 'approximate' and set a KL bar."
            )
        if self.prediction not in PREDICTIONS:
            raise ValueError(f"prediction must be one of {PREDICTIONS}, got {self.prediction!r}")
        if not 0.0 <= self.byte_share <= 1.0:
            raise ValueError(f"byte_share is a fraction in [0, 1], got {self.byte_share!r}")
        if self.is_identity and self.prediction != "identity":
            raise ValueError(
                f"{self.slug!r} installs no kernels, so it is the identity champion and must "
                f"predict 'identity', not {self.prediction!r}"
            )
        if self.prediction == "identity" and not self.is_identity:
            raise ValueError(
                f"{self.slug!r} predicts 'identity' but installs {self.kernels}. Only a "
                "candidate that installs nothing is bit-identical to the reference."
            )
        if not self.mechanism.strip():
            raise ValueError(f"{self.slug!r} has no mechanism. 'Fuse it and see' is not a hypothesis.")
        if not self.rationale.strip():
            raise ValueError(
                f"{self.slug!r} predicts {self.prediction!r} with no rationale. The prediction is "
                "the result; an unexplained one is worth nothing."
            )
        if self.weight_bits:
            # Validated here so a manifest typo fails on a laptop rather than scoring a
            # candidate against the wrong byte count on a rented box.
            from .harness.bytes_model import validate_weight_bits  # noqa: PLC0415

            validate_weight_bits(self.weight_bits)
        if self.requires is not None and not self.requires.reason.strip():
            raise ValueError(
                f"{self.slug!r} carries a precondition with no reason. The reason is the only "
                "thing a `precondition_failed` record says; without it the skip is unreadable."
            )

    def _check_top1_resolution(self) -> None:
        """A top-1 bar must be a count the sample can actually land on.

        `013-int8-full` missed a 0.97 bar at n = 264 by 0.000303, where one position is
        0.0038. 0.97 x 264 is 256.08, so the bar written as "97%" really meant "at most 7
        flips" — 0.92 of a sample away from where it was written, and the slot failed in
        that gap. A bar expressible as an exact k/n says which count it means; one that is
        not is claiming a precision the statistic does not have.
        """
        n = self.correctness_positions
        if n is None:
            return
        if n <= 0:
            raise ValueError(f"correctness_positions must be positive, got {n!r}")
        exact = self.top1_threshold * n
        if abs(exact - round(exact)) > 1e-9:
            nearest = round(exact)
            raise ValueError(
                f"{self.slug!r} sets top1_threshold={self.top1_threshold!r} over "
                f"{n} positions, which is finer than one sample: agreement quantises to "
                f"1/{n} and the bar falls at {exact:.4f} positions. Write it as a count, "
                f"e.g. {nearest}/{n}."
            )

    @property
    def is_identity(self) -> bool:
        return not self.kernels


@dataclass(frozen=True)
class Batch:
    """An ordered run of hypotheses on one rental.

    Order is load-bearing and is not sorted here. A manifest puts the calibration slot
    first so a broken harness is discovered in three minutes rather than at the end of the
    rental, and puts
    the riskiest kernels last so everything cheap is already on disk when one of them
    fails.
    """

    batch_id: str
    hypotheses: tuple[Hypothesis, ...]
    description: str = ""
    #: A batch whose product is the cost measurement itself rather than a ranked set of
    #: hypotheses. Exempt from the 7-12 floor in `docs/BATCHES.md`, which exists to
    #: amortise a rental's fixed cost across many measurements — an argument that cannot
    #: apply to the rental that is measuring what that fixed cost actually is.
    is_calibration: bool = False

    def __post_init__(self) -> None:
        if not self.hypotheses:
            raise ValueError(f"batch {self.batch_id!r} is empty")
        seen: set[str] = set()
        for hyp in self.hypotheses:
            if hyp.slug in seen:
                raise ValueError(f"batch {self.batch_id!r} lists {hyp.slug!r} twice")
            # Checked against slots already seen, so a precondition on a later slot -- or
            # on a slug this batch does not hold -- is refused here rather than silently
            # never firing, which would look exactly like a slot that had no precondition.
            if hyp.contrast_with is not None and hyp.contrast_with not in seen:
                raise ValueError(
                    f"{hyp.slug!r} contrasts with {hyp.contrast_with!r}, which must name an earlier "
                    f"slot in batch {self.batch_id!r}; earlier slots are {sorted(seen)}"
                )
            if hyp.requires is not None:
                for required in hyp.requires.slugs:
                    if required not in seen:
                        raise ValueError(
                            f"{hyp.slug!r} requires {required!r}, which must name an earlier slot "
                            f"in batch {self.batch_id!r}; earlier slots are {sorted(seen)}"
                        )
            seen.add(hyp.slug)

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __iter__(self) -> Iterator[Hypothesis]:
        return iter(self.hypotheses)

    def get(self, slug: str) -> Hypothesis:
        for hyp in self.hypotheses:
            if hyp.slug == slug:
                return hyp
        raise KeyError(f"batch {self.batch_id!r} has no hypothesis {slug!r}")

    @property
    def calibration_slug(self) -> str | None:
        """The identity hypothesis, if the batch has one."""
        for hyp in self.hypotheses:
            if hyp.is_identity:
                return hyp.slug
        return None


def scoped_registry(hypothesis: Hypothesis, source: KernelRegistry | None = None) -> KernelRegistry:
    """A registry holding exactly this hypothesis's kernels, each as champion.

    Built fresh per hypothesis rather than by mutating the global registry. Mutating the
    global one would work for a single measurement and then leak into the next slot — and
    a slot that quietly inherited the previous slot's kernels would produce a plausible
    number for the wrong candidate, which is the one failure this harness must not have.

    The "one champion per operation" invariant is enforced by `KernelRegistry.register`,
    so two kernels in the same hypothesis that replace the same operation raise here
    rather than at benchmark time.
    """
    return registry_for(hypothesis.kernels, source=source, label=hypothesis.slug)


def registry_for(
    names: tuple[str, ...] | list[str],
    source: KernelRegistry | None = None,
    *,
    label: str = "",
) -> KernelRegistry:
    """A registry holding exactly ``names``, each as champion of what it replaces.

    Separated from `scoped_registry` so that something which is not a hypothesis can ask
    for the same thing. The `output_code` dump is the case that needed it: entry 5 of
    `docs/HYPOTHESES.md` closed a 6.23% hypothesis for the price of a step and then said
    the next dump must be pointed at *the batch's own slots*, because rental 38's was
    pointed at a model with no kernel in it. A dump of one named composition is what
    `029`'s unexplained 20% now needs, and it is a set of kernel names rather than a slot.
    """
    source = REGISTRY if source is None else source
    scoped = KernelRegistry()
    for name in names:
        entry = source.get(name)  # raises RegistryError on an unknown name
        try:
            scoped.register(
                entry.name,
                impl=entry.impl,
                replaces=entry.replaces,
                status=KernelStatus.CHAMPION,
                hypothesis=label or entry.hypothesis,
                notes=entry.notes,
            )
        except RegistryError as exc:
            raise RegistryError(f"{label or 'registry'} cannot install {name!r}: {exc}") from exc
    scoped.check_invariants()
    return scoped


def unpaired_slots(batch: Batch) -> tuple[str, ...]:
    """Slugs that re-measure a kernel an earlier slot already ran without saying so.

    A batch is a set of controlled contrasts or it is a set of anecdotes. Two shapes are
    legitimate: an **ingredient measured alone** — the first slot in the batch to install a
    given kernel — and a **one-variable contrast** against an earlier slot, which says so
    with `Hypothesis.contrast_with`. Anything else is a slot whose comparison will be
    reconstructed after the fact, which is how rentals 42, 43 and 45 spent three writeups on
    a composition effect that did not exist.

    Returns the offending slugs in batch order, so a manifest can be refused on a laptop.
    """
    seen_kernels: set[str] = set()
    unpaired: list[str] = []
    for hyp in batch:
        reused = seen_kernels.intersection(hyp.kernels)
        if reused and hyp.contrast_with is None:
            unpaired.append(hyp.slug)
        seen_kernels.update(hyp.kernels)
    return tuple(unpaired)


def _exceeds(margin: float, band: float) -> bool:
    """``margin > band``, with floating-point dust treated as a tie.

    A promotion must not turn on representation error. ``1.02 - 1.0`` is
    ``0.020000000000000018`` in binary floating point, so a margin sitting exactly on the
    noise band would otherwise be promoted by 1.8e-17 of nothing. A tie goes to
    ``inconclusive``, which is the conservative direction: the bar is *beating* the noise
    band, not equalling it.
    """
    return margin > band and not math.isclose(margin, band, rel_tol=1e-9, abs_tol=1e-12)


def classify_outcome(
    median_ratio: float | None,
    iqr: float,
    *,
    correctness_passed: bool,
) -> str:
    """Turn one measurement into an outcome.

    A candidate that is wrong is `incorrect` whatever it measured — a fast wrong kernel
    is a useful record but never a win, and checking correctness first makes that
    ordering explicit rather than incidental.

    Otherwise the margin is compared against the run's **own** noise band. `AGENT.md` §6:
    a margin inside the interquartile spread of the scoring rounds is `inconclusive`, not
    a win. Recording a noise-band result as a win is how a leaderboard becomes fiction.
    """
    if not correctness_passed:
        return "incorrect"
    if median_ratio is None:
        return "error"
    margin = median_ratio - 1.0
    if _exceeds(margin, iqr):
        return "win"
    if _exceeds(-margin, iqr):
        return "loss"
    return "inconclusive"


def calibration_holds(median_ratio: float | None, iqr: float, *, floor: float = 0.02) -> bool:
    """Whether the identity champion measured 1.00 within the noise band.

    ``floor`` keeps an implausibly tight IQR from making this unfalsifiable: a run whose
    rounds happened to agree to four decimal places would otherwise reject a 0.5%
    deviation that is plainly just noise. The band is the wider of the measured IQR and
    ``floor``.

    When this is False every other slot in the batch is void. The candidate installed
    nothing, so it *is* the reference — a ratio away from 1.00 means the harness is
    measuring something other than the kernel under test, and no number it produced that
    day means what it says.
    """
    if median_ratio is None:
        return False
    return abs(median_ratio - 1.0) <= max(iqr, floor)


@dataclass(frozen=True)
class PredictionScore:
    slug: str
    predicted: str
    outcome: str
    correct: bool | None  # None when the slot produced no verdict to score against

    def to_dict(self) -> dict[str, object]:
        return {
            "slug": self.slug,
            "predicted": self.predicted,
            "outcome": self.outcome,
            "correct": self.correct,
        }


def score_predictions(
    batch: Batch,
    outcomes: dict[str, str],
    *,
    calibrated: bool | None = None,
) -> tuple[PredictionScore, ...]:
    """Score each registered prediction against what was measured.

    A slot that errored, never ran, or declined on a precondition scores ``None``, not
    ``False``: the prediction was never tested, and counting an untested prediction as
    wrong would understate the record exactly as counting it right would flatter it.

    The identity slot is scored against ``calibrated`` rather than against its outcome,
    because "the candidate is the reference" is a claim about the harness, not about a
    ratio being inside a band.
    """
    scores = []
    for hyp in batch:
        outcome = outcomes.get(hyp.slug, "not_run")
        if outcome in UNSCORED_OUTCOMES:
            correct: bool | None = None
        elif hyp.prediction == "identity":
            correct = calibrated
        else:
            correct = outcome == hyp.prediction
        scores.append(
            PredictionScore(slug=hyp.slug, predicted=hyp.prediction, outcome=outcome, correct=correct)
        )
    return tuple(scores)


#: Worst-case cold-cache phase costs in seconds, from §4.1 of
#: `docs/superpowers/specs/2026-09-10-compile-cost-and-memory-design.md`.
#:
#: **These are estimates**, used only until a rental has measured the real ones into
#: `cache/compile/<key>/phases.env`. Exactly one of them was observed rather than reasoned:
#: `reference_compile_s`, from rental 22's ~40-minute cold `max-autotune` compile. They are
#: deliberately pessimistic, because the cost of overestimating is a session that waits and
#: the cost of underestimating is a rental that buys nothing.
COLD_PHASE_ESTIMATES = {
    "setup_s": 1500.0,
    "reference_compile_s": 2400.0,
    "slot_s": 1980.0,
}


def session_fits_one_hypothesis(
    remaining_s: float,
    phases: dict[str, float] | None = None,
    reserve_s: float = 720.0,
) -> tuple[bool, float]:
    """``(fits, shortfall_s)`` for the least a rental may produce and still be worth it.

    That minimum is **two** slots: the identity champion plus one kernel. The identity slot
    calibrates the harness — without it every other number in the batch is void — but it
    scores no hypothesis, so a rental that fits only that has bought no science.

    Nine rentals have been billed on this project without producing a number. A session
    that cannot reach the minimum should not rent at all, and the shortfall says how far
    the gate is from being enough — which is the number a future session needs in order to
    raise it, rather than guessing again.
    """
    # A zero means "not measured" -- a phases.env written by a rental that never reached
    # that phase -- so it falls back to the cold estimate rather than claiming the phase is
    # free. Optimism here spends a rental.
    known = dict(COLD_PHASE_ESTIMATES)
    known.update({name: value for name, value in (phases or {}).items() if value > 0})
    needed = known["setup_s"] + known["reference_compile_s"] + 2 * known["slot_s"] + reserve_s
    shortfall = needed - remaining_s
    return (shortfall <= 0, max(0.0, shortfall))


@dataclass
class SlotBudget:
    """Decides whether the next hypothesis fits before the deadline.

    The batch must end *itself*, with its results written and the instance healthy.
    `AGENT.md` §5 treats a watchdog firing as a reportable fault, and a hypothesis killed
    mid-benchmark wastes the money already spent on it and records nothing in exchange.

    The estimate is the median of the slots already completed, which adapts to the card
    and the model actually in front of it rather than to a number someone guessed. Until
    a slot has finished there is nothing to take a median of, so it is seeded — generously,
    because the first slot is the one that pays for any lazily-initialised CUDA state.
    """

    deadline_epoch: float
    clock: Callable[[], float] = time.time
    seed_estimate_s: float = 240.0
    #: Slots overrun; a batch that stops one slot early has lost 3 minutes, and one that
    #: stops one slot late has lost the whole slot plus a fault in the writeup.
    safety_factor: float = 1.2
    #: Ceiling on a single slot once it has started. `can_start` bounds what a slot may
    #: *begin*; nothing bounded what it could then do, and rental 22 spent ~40 minutes
    #: inside one slot's cold compile before the session gate ended the run.
    slot_cap_s: float = 1800.0
    #: How many leading slots are exempt from that ceiling. The identity champion plus one
    #: kernel slot is the least a rental may produce and still have scored a hypothesis,
    #: so capping those two would defeat the guarantee the cap exists to protect.
    uncapped_slots: int = 2
    durations_s: list[float] = field(default_factory=list)

    def record(self, duration_s: float) -> None:
        if duration_s < 0:
            raise ValueError(f"a slot cannot take {duration_s} seconds")
        self.durations_s.append(duration_s)

    def estimate_s(self) -> float:
        if not self.durations_s:
            return self.seed_estimate_s
        ordered = sorted(self.durations_s)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def remaining_s(self) -> float:
        return self.deadline_epoch - self.clock()

    def can_start(self) -> bool:
        return self.remaining_s() >= self.estimate_s() * self.safety_factor

    def cap_for(self, index: int) -> float:
        """Seconds slot ``index`` may take. Never longer than the session has left."""
        remaining = self.remaining_s()
        if index < self.uncapped_slots:
            return remaining
        return min(self.slot_cap_s, remaining)

    def why_not(self) -> str:
        return (
            f"{self.remaining_s():.0f}s left before the session deadline; a slot is taking "
            f"~{self.estimate_s():.0f}s and the budget needs "
            f"{self.estimate_s() * self.safety_factor:.0f}s to start another. Stopping here "
            "with results written rather than being cut off mid-benchmark."
        )
