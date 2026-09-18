# Batches: many hypotheses per rental

**Read this before filling a batch.** `AGENT.md` §4 is the session workflow;
`docs/superpowers/specs/2026-09-06-batched-hypotheses-design.md` is the design and the
reasoning. This file is the practical part.

---

## Why

A rental's cost splits in two:

| | Cost | Paid |
|---|---:|---|
| Container image, torch, 9.32 GB checkpoint, GPU suite | ~19 min *(measured, rentals 34-35)* | **once per rental** |
| Reference `max-autotune` compile | **57 s warm / 268 s off a 47 MB cache** *(measured)* | **once per rental** |
| Candidate compile + correctness gate + benchmark | **173-376 s** *(measured, rental 35)* | **per hypothesis** |

> **Measured at last, on 2026-09-14, and the original estimates were not far wrong.** The
> ~40-minute compile that dominated this file for six days was never the cost of compiling
> this model. `gated_delta_rule` unrolls its sequence scan into the graph, so the benchmark's
> 2048-token prefill handed inductor ~1.08M FX nodes — and `run_interleaved` excludes every
> `setup` from the timed region, so none of it was ever measured. The prefill now runs eager.
>
> | Rental | Cores | Slot 0 reference compile |
> |---|---:|---|
> | 30 | 64 | ~52 min, host dropped before it finished |
> | 31 | 256 | 3162 s, then `BackendCompilerFailed` (live autograd) |
> | 32 | 256 | 6983 s — hit its cap without completing |
> | 34 | 96 | **267.5 s** — first cold compile ever to finish |
> | 35 | 96 | **57.4 s** — off the 298 MB cache rental 34 brought home |
>
> Rental 35 ran a full nine-slot batch in **44 minutes** of batch time, inside a 52.65-minute
> rental. Every timed round is ~900 ms at an IQR under 0.005.

> **The real ceiling on batch size is not the clock.** Dynamo caches compiled code per code
> object, and each slot's candidate is a new cache entry, so at the default `recompile_limit`
> of 8 a batch stops compiling candidates part-way through and **silently times eager ones**.
> Rental 35 tripped it inside slot 2 and lost six of nine slots to plausible-looking ratios
> that measured nothing. `recompile_limit_for` in `batch_run.py` raises the limit to cover
> every slot, and each slot record now carries `graphs_compiled` so a candidate that did not
> compile says so. **Neither has run on a GPU yet.** If you widen a batch, widen that too.

One hypothesis per rental pays fifteen minutes of fixed cost to buy three minutes of
science. Nine rentals were billed that way and none produced a number. A batch pays the
same fifteen minutes and buys 7-12 measurements.

**7 is the floor** — fewer does not justify the fixed cost. **12 is the ceiling** — more
does not fit the 180-minute session gate. Both now rest on measured numbers rather than
guesses, and the measurement is kinder than the guess: at ~200 s a slot and ~19 min of
fixed cost, twelve slots fit comfortably. The binding constraint is the recompile limit
above, not the clock.

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


## Measured costs, rental 37 (2026-09-16) — batch 003, seven slots

The first batch to run end to end with nothing voided, so these are the numbers to cost the
next one with. RTX 5090, **warm** compile cache pulled from rental 35.

| | Measured |
|---|---|
| Whole rental, provisioning to destroy | **55.55 minutes, $0.3786** at $0.4089/hr |
| Fixed cost before slot 0 | ~19 minutes (image, torch, 9.32 GB checkpoint, GPU suite) |
| Slot 0 — identity, includes the reference compile | **179 s** |
| Slots 1-6 | **293-428 s**, median 306 s |
| Of which the approximate correctness gate | 10-19 s |
| Of which the benchmark | 164-418 s |
| Peak memory | 9.0 GiB bf16, **12.9 GiB** with int8 copies resident, of 31.36 |

**A seven-slot batch fits comfortably in a 180-minute session** — it used 55. The earlier
estimate of 173-376 s per slot from rental 35 held; the cap is memory and reading time, not
the clock. Note that the candidate *compile* no longer shows as a separate phase: with a
warm cache `compile_candidate_compiled` rounds to 0 s and the cost has moved inside the
benchmark's first warmup round.

**The `compiled` column drifted 875 → 987 ms across the batch, 13%.** That is thermal, it is
expected, and it is exactly why the score is a within-slot ratio and absolute times are
provenance only. A batch that compared slot 6's candidate against slot 0's reference would
have invented a 13% effect out of the cooling fan.

## Measured costs, rental 38 (2026-09-17) — batch 004, two slots run and five declined

The first batch to stop itself. RTX 5090 at $0.4363/hr, warm compile cache pulled from
rental 37.

| | Measured |
|---|---|
| Whole rental, provisioning to destroy | **39.65 minutes, $0.2883** |
| Fixed cost before slot 0 | **~30 minutes** — of which the checkpoint fetch alone was **10m49s** |
| The `output_code` dump step | ~4 minutes, two `max-autotune` compiles, both cached forward |
| Slot 0 — identity, includes the reference compile | **187 s** |
| Slot 1 — one Triton kernel on 248 sites | **306 s** (benchmark 293 s, correctness gate 13 s) |
| Slots 2-6 | **0 s** — `precondition_failed`, never built |

**The fixed cost is not ~19 minutes; it is 19-30 and the variable is the network.** The
9.32 GB checkpoint took 10m49s here against a few minutes on rental 37, from the same
unauthenticated HuggingFace endpoint. Budget the pessimistic figure: the difference is a
whole slot.

**The `output_code` dump costs about one slot and pays for two.** Both of its compiles land
in the shared fx-graph and autotune caches, so the batch behind it starts warm on the
reference and on the identity slot — and what it produced closed a 6.23% backlog entry that
five rentals of kernel work had not touched. Run it every time.

**Declining a slot is free.** Five `precondition_failed` records cost nothing but the line
that wrote them; the batch went from a projected ~70 minutes to 39.65.

## The gap this batch found: slots cannot be conditional — closed 2026-09-17

**Built on rental 38 and it fired on its first outing.** `Hypothesis.requires` names an
earlier slug and a floor; `precondition_holds` fails closed, so a slot that errored and has
no ratio declines the slots behind it rather than letting them run; `precondition_failed` is
its own outcome, distinct from `not_run` (the clock) and `starved` (the rental scored
nothing), and it scores no prediction because it tested none.

Batch 004's control returned 0.1934 against a 0.56 floor and all five quantised slots
declined. The batch ended at 39.65 billed minutes against batch 003's 55.55 in the identical
situation — where batch 003 had re-measured the same settled fact at five bit widths.

Two things learned from using it:

* **Put the floor's *reason* in the manifest, not just the number.** It is the only text a
  `precondition_failed` record carries, and it is what a reader needs to judge whether the
  floor was right rather than merely whether it fired.
* **A declined slot writes no individual result file.** `run_batch` appends the result but
  does not call `on_slot`, so the batch directory holds a file per slot that *ran* and the
  skips live only in `summary.json`. That matches how `not_run` already behaves; it is worth
  knowing before looking for a file that is not there.

The historical account of the gap follows.

### How the gap looked before it was closed

Batch 003 spent **five of seven slots** on quantisation variants whose outcome was fully
determined the moment slot 1 returned 0.2801. `009-gemv-bf16-control` moves exactly cuBLAS's
bytes; at 0.28 it says the kernel is nowhere near the roofline, and therefore that no
byte-saving variant built on it can win. Every slot after it re-measured that fact at a
different bit width.

Nothing in `Batch` or `run_batch` can express "stop here". The ordering rule — calibration
first, cheapest and most diagnostic next, riskiest last — already encodes the *intent* that
early slots inform later ones, but the runner cannot act on it.

**What to add:** an optional predicate per hypothesis, evaluated against the slots already
finished, that turns the remaining slots into `not_run` with a recorded reason rather than
running them. It belongs in `batch.py` beside `SlotBudget`, which is the existing precedent
for a slot the batch declines to start, and it should record *why* — `not_run` because a
precondition failed is a different fact from `not_run` because the clock ran out, and
flattening them would erase the evidence.

The first user was `docs/HYPOTHESES.md` entry 6's gate — **"the bf16 GEMV control must reach
≥ 0.56 before any quantised slot is worth running"** — and it is the one that fired.
