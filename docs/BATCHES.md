# Batches: many hypotheses per rental

**Read this before filling a batch.** `AGENT.md` §4 is the session workflow;
`docs/superpowers/specs/2026-09-06-batched-hypotheses-design.md` is the design and the
reasoning. This file is the practical part.

---

## Why

A rental's cost splits in two:

| | Cost | Paid |
|---|---:|---|
| Container image, torch, 9.32 GB checkpoint, GPU suite | ~10-15 min | **once per rental** |
| Reference `max-autotune` compile | ~3-4 min *(estimated — see below)* | **once per rental** |
| Candidate compile + correctness gate + benchmark | ~2-4 min *(estimated — see below)* | **per hypothesis** |

> **These two estimates have never been measured, and the first evidence contradicts them.**
> Rental 22 (2026-09-08) spent **~40 minutes inside a single slot** without finishing it —
> `nvidia-smi` showing the GPU at 0% and python at 129% CPU, i.e. a cold `max-autotune`
> compile, not a hang. The session gate ended the run, not the science.
>
> If a first compile really costs 40 minutes, **the arithmetic below does not hold and a
> nine-slot batch does not fit a two-hour session either.** Measure one compilation before
> filling another batch, then either bring it down (`reduce-overhead`, a warm inductor
> cache carried between rentals, on-disk `torch.compiler` caching) or re-cut the batch
> around what it actually costs.

One hypothesis per rental pays fifteen minutes of fixed cost to buy three minutes of
science. Nine rentals were billed that way and none produced a number. A batch pays the
same fifteen minutes and buys 7-12 measurements.

**7 is the floor** — fewer does not justify the fixed cost. **12 is the ceiling** — more
does not fit the 180-minute session gate. Both numbers assume the per-slot cost above, and
that estimate is the thing rental 22 contradicted: re-cut them the moment a compile has
actually been timed.

**A calibration batch is exempt from the floor.** `Batch(is_calibration=True)` says so, and
`002-compile-cost` is one: three slots whose product is the clock rather than the ratios.
The floor exists to amortise a rental's fixed cost across many measurements, which cannot
be an argument against the rental that is measuring what that fixed cost is.

## What a hypothesis has to say for itself

From `src/deltaforge/batch.py`:

```python
Hypothesis(
    slug="006-gqa-no-expand",
    kernels=("gqa_decode",),  # registry names; () is the identity champion
    category="B",  # A, B or C from docs/HYPOTHESES.md
    byte_share=0.0623,  # a FRACTION, from docs/roofline.py
    replaces=("gqa_attention",),
    mechanism="Index the unexpanded KV cache directly instead of ...",
    prediction="win",  # win | loss | inconclusive | identity
    rationale="The only slot here with a byte share worth anything ...",
)
```

`mechanism` and `rationale` are validated as non-empty, and `rationale` is expected to be a
paragraph rather than a label. That is not bureaucracy. `AGENT.md` §1 says the finding is a
mechanistic account registered *before* the measurement — that being right in advance is
the result — and until batches arrived nothing in this repo recorded a prediction anywhere
it could be scored. **The batch summary is a prediction scorecard**, and it is worth more
than any single ratio in it.

Predict honestly. A batch that predicted a win everywhere would not be a prediction, it
would be hope; `batches_test.py` asserts that only the highest byte-share slot predicts one.

## Ordering is load-bearing and is never sorted

1. **Identity champion first.** It installs nothing, so it *is* the reference and must
   measure 1.00 ± noise. A broken harness then costs three minutes instead of a rental —
   and if it misses, **every other number in the batch is void.**
2. **Cheapest and most diagnostic next.**
3. **Riskiest last**, so everything already measured is on disk before one of them fails.

## What batching changes about the graveyard

`docs/HYPOTHESES.md` closed five hypotheses as "unmeasurable": ceiling below the noise
band, not worth a rental.

**That was an argument about cost, not truth.** At three minutes a slot it no longer holds.
A measured null carrying a real ratio and IQR from a real card is a stronger record than an
arithmetic prediction of a null — and they are not the same claim, because a measurement
also contains launch overhead, CUDA-graph behaviour, and whatever inductor actually emitted,
none of which appear in a byte count.

Still rank by byte share. Still refuse to promote on a noise-band margin. But stop using
"too small to measure" as a reason not to fill a slot.

## Guarantees the runner makes

* **A failing slot costs a slot, not the rental.** Each hypothesis runs in its own
  try/except; an exception is recorded as `error` with its traceback and the batch goes on.
* **Records are written as each slot finishes.** A hard crash at slot 8 leaves 0-7 on disk.
* **Teardown pulls results before destroying.** The trap could previously only destroy, and
  cannot rsync from a dead box.
* **The batch stops itself** before a slot that will not fit the deadline, recording the
  rest as `not_run`. A watchdog firing is a reportable fault; this is what prevents it.
* **A no-op candidate is refused, not measured.** If a non-identity hypothesis patches no
  module class, the slot errors. A candidate identical to the reference measures 1.00 and
  is otherwise indistinguishable from a well-behaved null result.

## Two things the design does *not* give you

**Cross-slot comparability.** Nine hypotheses on one card share one thermal history. The
interleaving inside a slot divides out drift *within* it, not *between* slots. Ratios are
comparable within a hypothesis and only loosely across them — say so in any writeup that
compares two slots.

**A weaker correctness bar than you think.** Layer-1 per-kernel checks run in full. The
layer-2 exact-token gate runs at **32** tokens rather than 128, recorded in every result.
Re-check anything being promoted at 128.

## Adding a batch

1. Write the kernels, each with an installer in `model.INSTALLERS` and checks in
   `kernels.CHECK_BUILDERS` — **both keyed by kernel name, not by the operation replaced.**
   Several kernels routinely attack the same operation, and an operation-keyed table would
   silently run one kernel's checks against another and report `pass`.
2. Add the `Batch` to `src/deltaforge/batches.py`.
3. `uv run pytest` — `batches_test.py` proves every hypothesis installs and actually
   changes the model, on a CPU, before any money is spent.
4. `remote/run_remote.sh --dry-run --session-id smoke --batch NNN-slug`.
5. Commit, **then** rent. The predictions must be committed before the measurement or they
   are not predictions.

## After the rental

The batch is not finished when the instance is destroyed.

6. **Write `results/batches/NNN-slug/README.md`.** What is now settled, where the batch
   died if it died, what is still open, and what it cost. The prediction scorecard belongs
   here: which predictions were scored, which were right, and which slots never tested
   theirs. A slot that errored tested nothing — counting it wrong understates the record
   exactly as counting it right would flatter it.
7. **Reconcile this file with what the rental measured.** The per-slot and fixed costs
   above are estimates; a rental that timed them replaces them. **An estimate a rental has
   contradicted is worse than no estimate**, and the batch size is derived from it.
8. **Check `cache/compile/<key>/phases.env` came home.** The batch writes its measured
   phase costs there and `run_remote.sh` sources them next time, so the pre-flight check
   uses what this rental actually cost rather than the pessimistic cold estimates in
   `batch.COLD_PHASE_ESTIMATES`. No file means the next session pays the cold assumption.
9. **Then the rest of the docs** — `AGENT.md` §6.1 lists which, and when each is worth
   touching. Void batches are written up too: `results/batches/001-calibration/` is what a
   batch that produced no ratio at all still owes the next session.
