"""What this GPU model has run the reference at before, and whether today's card matches.

**Rental 43 is why this file exists.** Two RTX 5090s reporting the same memory clock, the
same driver and the same torch ran the reference **1.61x apart**: 1282.6 GB/s on rental
40 against 845.3 on rental 43. Every prediction in batch 007 was derived from the first
number, every one of them was wrong by roughly that factor, and the batch found out in
the writeup rather than at minute 27 — even though the instrument was already in the run.

`000-identity` measures the reference column's achieved bandwidth before any kernel slot
starts, so the comparison costs nothing. This module is that comparison, and the rule it
encodes is the one `results/batches/007-compose-and-retile/README.md` asked for:

> A pre-flight that compares that figure against what this GPU model has recorded before,
> and says so loudly, costs nothing and would have framed every number below correctly
> from minute 27 instead of from the writeup. Whether it should *abort* is a separate
> question and the answer is probably no: a slow card still produces valid within-slot
> ratios, and this rental's most valuable finding came from one.

So this **reports and never decides**. It cannot fail a batch, skip a slot or change a
threshold. A slow card is not a broken card: ratios are measured inside one process
against a reference timed in the same interleaved rounds, so they stay valid — what stops
being valid is carrying an *absolute* prediction, or a number, from another rental.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CARD_SPREAD_NOTE",
    "CLOCK_DRIFT_NOTE",
    "ROUNDS_NOTE",
    "RECORDED_REFERENCE_GBPS",
    "ReferenceObservation",
    "card_report",
    "recorded_for",
]


@dataclass(frozen=True)
class ReferenceObservation:
    """One rental's identity-slot measurement of the reference column.

    The identity slot rather than the batch baseline, because it is the one figure every
    rental produces the same way: the `compiled` column's achieved GB/s inside the slot
    that installs nothing.
    """

    rental: int
    date: str
    gbps: float
    note: str = ""


#: Keyed by `torch.cuda.get_device_name()`, which is what `summary.json` records.
#:
#: **Add a row here when a batch runs**, from `results/batches/<id>/000-identity.json`:
#: ``bench.achieved_gbps["compiled"]``. A table that stops being updated is worse than no
#: table, because it silently turns a normal card into an outlier.
RECORDED_REFERENCE_GBPS: dict[str, tuple[ReferenceObservation, ...]] = {
    "NVIDIA GeForce RTX 5090": (
        ReferenceObservation(40, "2026-09-19", 1282.6, "the champion's rental; 6.70 ms/token"),
        ReferenceObservation(42, "2026-09-20", 1223.2, "7.17 ms/token"),
        ReferenceObservation(
            43,
            "2026-09-20",
            845.3,
            "HiveOS host, machine 9105, reliability 0.9808; every measured effect went to zero",
        ),
        ReferenceObservation(
            45,
            "2026-09-23",
            1197.0,
            "healthy at slot 0 and downclocked 2910 -> 2400 MHz by slot 4; see below",
        ),
        ReferenceObservation(
            46,
            "2026-09-23",
            1214.0,
            "machine 140734 again -- the same host as rental 45, and it drifted the same way",
        ),
    ),
}

#: **A pre-flight tests the card you were given, not the card you will still have.**
#: Rental 45 reported 1197 GB/s here and passed, then lost 17% of its SM clock at slot 4
#: and held the lower clock for the rest of the batch -- the reference column drifting
#: from 7.17 to 7.92 ms/token inside one rental. Interleaved rounds divide that out of
#: every ratio, so no slot is void; what it costs is *resolution*, and six of eleven slots
#: came back `inconclusive` at IQRs of 0.019-0.151 against rental 40's 0.00034. This
#: module cannot see that coming and should not pretend to. The instrument that answers it
#: is more scoring rounds when the IQR is wide.
CLOCK_DRIFT_NOTE = (
    "Rental 45's SM clock fell 2910 -> 2400 MHz at slot 4. Ratios survive it because each "
    "slot times its own reference in the same interleaved rounds; resolution does not."
)

#: **And resolution is buyable.** Rental 46 rented the same host, drifted the same way, and
#: ran 15 scoring rounds instead of 5: IQRs fell from 0.0074-0.1511 to 0.0072-0.0213 and
#: one slot of ten came back `inconclusive` against six of eleven. Three of that batch's
#: conclusions were unavailable at the old sample size. See `cli.BATCH_ROUNDS`.
ROUNDS_NOTE = (
    "A drifting card is a resolution problem before it is a measurement problem, and more "
    "scoring rounds are the instrument: 5 -> 15 rounds took the worst IQR from 0.15 to 0.02."
)

#: How far below the best recorded observation counts as "this is a different card".
#: Rental 43 sat at 0.66 of rental 40 and rental 42 at 0.95, so the line goes between
#: them — and it is a **reporting** threshold, not a gate.
SLOW_FRACTION = 0.85

CARD_SPREAD_NOTE = (
    "A slow card does not void a ratio: every slot times its own reference in the same "
    "interleaved rounds. What it voids is an absolute prediction, and any comparison "
    "against a number from another rental."
)


def recorded_for(gpu_name: str) -> tuple[ReferenceObservation, ...]:
    return RECORDED_REFERENCE_GBPS.get(gpu_name, ())


def card_report(gpu_name: str, measured_gbps: float | None) -> str:
    """One block of text for the run log, said before any kernel slot is read.

    Returns a string rather than logging, so the CPU suite can assert on what a given
    measurement would have said rather than on a side effect.
    """
    head = f"[batch] card pre-flight: {gpu_name or 'unknown GPU'}"
    if measured_gbps is None:
        return (
            f"{head} -- the identity slot reported no achieved bandwidth, so this rental's "
            "card is uncharacterised. Read every absolute number below with that in mind."
        )

    recorded = recorded_for(gpu_name)
    line = f"{head} ran the reference at {measured_gbps:.0f} GB/s in the identity slot."
    if not recorded:
        return (
            f"{line} **This project has never recorded this GPU model before**, so there "
            "is nothing to compare it against and no prediction here was derived from it. "
            f"{CARD_SPREAD_NOTE}"
        )

    history = ", ".join(f"rental {o.rental} {o.gbps:.0f}" for o in recorded)
    best = max(o.gbps for o in recorded)
    fraction = measured_gbps / best
    verdict = f"{fraction:.2f}x the best recorded ({best:.0f} GB/s). Previously: {history} GB/s."
    if fraction < SLOW_FRACTION:
        return (
            f"{line} WARNING -- this card is SLOW: {verdict} Rental 43 sat at 0.66 here and "
            "every mechanism in that batch measured zero against an identity slot carrying "
            f"+1.01%. Read every slot below against this rental's own identity. {CARD_SPREAD_NOTE}"
        )
    return f"{line} In family: {verdict} {CARD_SPREAD_NOTE}"
