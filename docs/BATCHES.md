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

Predict honestly, and prefer a range to a verdict — batch 005 registered "1.05-1.12" for
the slot that measured 1.0791, which is a sharper claim than "win" and was scored the same.
A batch that predicted a win everywhere would be hope rather than prediction; batch 001 is
pinned to exactly one winner by `batches_test.py`, and later batches carry per-batch tests
instead, because a batch of six variations on one mechanism legitimately predicts several.

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

## Measured costs, rental 40 (2026-09-19) — batch 005, six slots run and two declined

The batch that produced the project's first champion. RTX 5090 at $0.4363/hr, warm compile
cache, and **the cheapest fixed cost this project has recorded.**

| | Measured |
|---|---|
| Whole rental, provisioning to destroy | **32.32 minutes, $0.2350** |
| Fixed cost before slot 0 | **~13 minutes** — of which the 9.32 GB checkpoint was **1m24s** |
| The `output_code` dump step | ~2 minutes, both compiles off the warm cache |
| Slot 0 — identity, includes the reference's first compiled call | **116 s** |
| Slots 1-5 | **147-292 s** |
| Of which the approximate correctness gate | 5.8-9.2 s |
| Slots 6-7 | **0 s** — `precondition_failed` |
| Peak memory | 8.07 GiB bf16, **16.73 GiB** with the int4 head resident, of 31.36 |

**The fixed cost is 13-30 minutes and the variable is the network, plus one avoidable
1.4 GB.** Rental 38 paid ~30 minutes and rental 40 paid ~13. Part of that is the
checkpoint endpoint being fast this time; part is that the repo sync had been shipping the
compile cache — 1.4 GB against 9.5 MB of repository — buried inside it, to a path nothing
reads, on every rental since one first came home. It is excluded now.
[`docs/GPU-ACCESS.md`](GPU-ACCESS.md) blocker 17 has the account.

**A cost the model does not carry: a candidate that replaces the root model class pays a
full `max-autotune` recompile in its first warmup round.** The identity candidate reuses the
reference's compiled code and took 858 ms; `021-static-cache-cudagraphs` took **71.7 s** and
`025-fused-causal-conv` **218 s**, because a new class is a new dynamo code object. Warmup
rounds are discarded so no ratio moves, but budget 1-4 extra minutes for any slot whose
installer swaps `type(model)`, and do not mistake it for a hang.

## What a declined slot costs, which is not nothing

Batch 004's preconditions declined five slots and saved 16 billed minutes. That was right,
and batch 005's declined two more for the same good reason.

**What comes with the saving is that the declined code never runs.**
`_tiled_gemv_scaled_kernel` was written for batch 004, carried five slots, was declined five
times, and shipped. It passed `uv run pytest` and it was wrong: it declared `SCALE` and
`HAS_SCALE` and read neither, so int8 and fp8 returned unscaled integer dot products —
layer-1 relative error **4511** when batch 005 finally ran it. Its `other=0` also refuses to
cast to e4m3, so the fp8 slot did not compile at all.

A Triton body is a string compiled on a GPU. The type checker cannot see into it, the linter
cannot, and neither can the CPU suite — so **"declined" means unexecuted, not verified**.
Two things follow:

* `kernels/kernel_contract_test.py` checks what the AST *can* see: a parameter declared and
  never read, and a dtype-polymorphic masked load whose `other` only one dtype accepts. Both
  of rental 40's defects are that shape, and both now fail on a laptop.
* **When a batch picks up a kernel an earlier batch declined, order it as new code** —
  early, where a failure costs a slot and informs the rest, not last among the riskiest.


## Measured costs, rental 42 (2026-09-20) — batch 006, five slots run and three declined

RTX 5090 at $0.5163/hr, warm compile cache, and **the most expensive fixed cost since
rental 38** — because this rental *sends the cache up* as well as pulling it down.

| | Measured |
|---|---|
| Whole rental, provisioning to destroy | **50.82 minutes, $0.4373** |
| Fixed cost before slot 0 | **27.12 minutes** — of which the 9.32 GB checkpoint was only **1m27s** |
| Slot 0 — identity | **179 s** (bench 162 s, correctness 16 s) |
| Slots 1-4 | **178-362 s** |
| Of which the approximate correctness gate | 10.3-16.4 s |
| Of which install-time tile tuning | **12.1 s** for one shape, **14.8 s** for two |
| A candidate that replaces the root model class | **+198 s** in its first warmup round |
| Declined slots | **0 s** |
| Peak memory | 8.39 GiB allocated, 16.61 GiB reserved, of 31.36 |

**The fixed cost is 13-30 minutes and the network is only half the variable.** Rental 40
paid ~13 and this one paid 27.12 with a *faster* checkpoint fetch (1m27s against 10m49s on
rental 38). The difference is the container pull and the **1.4 GB compile cache going
up**. That upload is deliberate and it buys a warm reference compile —
`compile_candidate_compiled` rounds to 0.0 s on every slot — but on a 50-minute rental it
cost more than the 3.5 minutes of cold compile it saved. Worth reconsidering for a short
batch; worth keeping for a long one.

**The root-class recompile now has a number rather than a range.** Rental 40 saw 72 s and
218 s and recorded "1-4 minutes"; rental 42 put `029` at 197.7 s and `030` at 199.2 s.
Budget ~200 s for any slot whose installer swaps `type(model)`, and note it is *separate*
from install-time work, which `candidate_build` times on its own.

**Three launches produced no rental and cost $0.0337 between them.** One died on CUDA
`Error 804` after 5.13 billed minutes; two refused before provisioning — one to an HTTP
429 on the offer search, one to a genuinely empty pool. Only the log line distinguishes
those two, and they call for opposite responses: retry, or move a filter. See
`docs/GPU-ACCESS.md`.

## Measured costs, rental 43 (2026-09-20) — batch 007, six slots run and three declined

RTX 5090 at $0.4622/hr, warm compile cache, **cheapest full batch this project has run**
and the first where the run plan's estimate was high rather than low.

| | Measured |
|---|---|
| Whole rental, provisioning to destroy | **42.67 minutes, $0.3287** |
| Fixed cost before slot 0 | **~27 minutes**, including the composition dump |
| Slot 0 — identity | **243 s** (bench 221 s, correctness 21.8 s) |
| Slots that compiled a new candidate graph | **252-270 s** |
| **Slots that reused a cached graph** | **192-193 s** |
| Of which the approximate correctness gate | 14.4-15.9 s |
| A candidate that replaces the root model class | **+63-70 s** in its first warmup round |
| Declined slots | **0 s** |
| Peak memory | 8.39 GiB allocated, 16.61 GiB reserved, of 31.36 |

**The run plan estimated 75-95 minutes and $0.65-0.85; it took 42.67 and $0.3287.** Two
reasons, and the second is new:

1. Three of nine slots declined on their preconditions, which is the mechanism working.
2. **Two slots reused a compiled graph and paid no compile at all.** `037` and `038` pin a
   tile, and the tile is a launch parameter rather than graph structure, so dynamo's guards
   passed and the candidate reused `035`'s graph. Those slots ran in **192-193 s against
   252-270** for the slots that compiled — and they reported `graphs_compiled: 0`, which is
   a **cache hit, not an eager fallback** (see `AGENT.md` §8).

**So a batch whose slots differ only in launch parameters is materially cheaper than its
slot count suggests**, and `COLD_PHASE_ESTIMATES` has no way to know that in advance. The
pessimistic estimate is still the right gate; this is a note about why a batch can finish
early, not a licence to plan against the optimistic number.

**The root-class recompile fell from ~200 s to 63-70 s** on a warm cache for this card.
Rental 42 put it at 197.7 and 199.2 s and this file said "budget ~200 s". That is now
**63-70 s warm, ~200 s cold**; budget the cold number, because the cache key includes the
card and the market decides which card.

## Measured costs, rental 45 (2026-09-23) — batch 008, eleven slots, all of them run

RTX 5090 at $0.4896/hr. **The largest batch this project has run, and the worst ratio of
fixed cost to science it has recorded.**

| | Measured |
|---|---|
| Whole rental, provisioning to destroy | **115.52 minutes, $0.9427** |
| Fixed cost before slot 0 | **~72 minutes** |
| — of which the compile-cache **push** | **~33 minutes for 2.2 GB** |
| — of which the container image pull | ~25 minutes |
| — of which the composition dump | ~6 minutes |
| Eleven slots, total | **~39 minutes** |
| Slot 0 — identity | 183 s (bench 166 s, correctness 16.3 s) |
| Slots that compiled a new candidate graph | 189-197 s typical, **321-335 s** for two |
| Slot that reused a cached graph (`047`, pinned tile) | **141 s** |
| The approximate correctness gate | 10.5-11.9 s |
| Peak memory | 14.5 GiB allocated of 31.36 |

**Seventy-two minutes of fixed cost to buy thirty-nine of measurement, and most of the
excess has a name.** The compile cache for this card has grown to **2.2 GB**, and pushing
it took about 33 billed minutes at ~1.05 MB/s of home uplink — to save a warm reference
compile worth **268 s cold against 57 s warm, about 3.5 minutes**. The teardown pull then
timed out at `DF_CACHE_PULL_TIMEOUT` and the rental banked nothing for it.

**The cache's cost scales with history and its saving does not.** A warm compile saves the
same ~3.5 minutes whether the directory holds 200 MB or 2.2 GB, because inductor only reads
the entries this run needs. So the transport is now bounded: `run_remote.sh` refuses a push
above `DF_CACHE_MAX_PUSH_MB` (default 512) and says why, failing toward a **cold compile**,
which costs minutes, rather than a stalled upload, which costs tens of them. The real fix
is pruning the cache or pushing it concurrently with the 9.32 GB checkpoint fetch, and
neither is done.

**Two slots cost 321-335 s rather than ~190**, both of them candidates that had never
compiled on this card in any form — `045`'s first warmup round took **194.8 s** and `049`'s
was similar. That is the root-class-recompile line of rental 43 in a new costume: a *new
kernel* pays it too, and on a cold-for-this-graph cache it is nearer 200 s than 70.

**Eleven slots fit comfortably.** The batch's floor of 7 and ceiling of 12 held: eleven
slots ran in 39 minutes, and `SlotBudget` never had to stop one. The binding constraint on
batch size is not the clock — it is the fixed cost in front of it.

## Measured costs, rental 46 (2026-09-23) — batch 009, eight slots run, one errored, two declined

RTX 5090 at $0.4896/hr on **machine 140734, the same physical host as rental 45**, with
**15 scoring rounds instead of 5** and the compile-cache push refused by its new ceiling.

| | Measured |
|---|---|
| Whole rental, provisioning to destroy | **79.00 minutes, $0.6446** |
| Fixed cost before slot 0 | **~22 minutes** (rental 45, same host: ~72) |
| — the compile-cache push | **skipped**: 2236 MB over the 512 MB ceiling |
| Eight slots run | **364-514 s each** (batch 008: 129-335 s) |
| Slot that errored at trace time | **34 s** |
| Two declined slots | 0 s |
| Round 0 of a candidate compiling cold | **186 s** |
| Round 0 of a candidate reusing most of a graph | **54 s** |
| Peak memory | 14.5 GiB of 31.36 |

**`DF_CACHE_MAX_PUSH_MB` paid for itself on its first rental.** Fixed cost fell from ~72
minutes to ~22 on the same host and the same image, by trading a 33-minute 2.2 GB upload
for ~3.5 minutes of cold compiling. The ceiling is a stopgap; pruning the cache or pushing
it concurrently with the checkpoint fetch is still the right fix.

### A benchmark round costs ~18 s, not ~2 s, and the estimate that said otherwise was 9x low

`cli.BATCH_ROUNDS` was raised from 7 to 17 on the arithmetic that "a round is ~2 s of wall
clock for both columns" — read off the ~900 ms median each column spends **inside the timed
region**. Measured on the box, consecutive rounds are **~18 s apart**.

The gap is `run_interleaved`'s `setup`, which is excluded from every timed region by design
and is where the 2048-token prefill runs. So ten extra rounds cost **~180 s a slot**, not
20, and about 25 minutes across a ten-slot batch.

**It was worth it and the arithmetic should still have been right.** Budget a batch round
at **~18 s**, and remember that the number the bench reports as `median_ms` is the timed
region alone — it is not what a round costs the rental.

### What 15 scoring rounds bought

| | rental 45 (5 scoring rounds) | **rental 46 (15)** |
|---|---|---|
| IQR range across slots | 0.0074 – **0.1511** | 0.0072 – **0.0213** |
| identity IQR | 0.0193 | **0.0115** |
| `inconclusive` slots | **6 of 11** | **1 of 10** |

Same host, same mid-rental downclock. **Three of batch 009's conclusions were unavailable at
the old sample size**: that the int4 head loses (0.9851 ± 0.0099 against rental 45's 1.0105
± 0.0263), that the static cache wins (1.0150 ± 0.0123, after two unresolvable attempts),
and that the cache adds nothing to the fused conv — a difference batch 008 reported as +3%
inside a band of 0.1511.

## What a prediction scorecard looks like when the premise is wrong

Batch 006 scored **1 of 4**. Every slot predicted `win`, and `docs/BATCHES.md` has said
since it was written that "a batch that predicted a win everywhere would be hope rather
than prediction". This batch was built under an explicit instruction that every slot have
a real chance of beating the incumbent, which is a legitimate brief and is **not the same
as predicting that every slot will**.

The lesson is narrower and it is about *correlation*, not optimism: four of the eight
slots shared one unexamined premise — that the tile was the problem — so the first slot
to test it determined the rest. That is the same structure batch 003 had, where five
slots were decided the moment `009` returned 0.2801, and the preconditions that exist
because of it worked again here (three slots declined, ~15 minutes saved).

**So: when several slots rest on one premise, put the slot that tests the premise first
and gate the others on it** — which batch 006 did — **and register at least one slot
whose prediction does not depend on that premise.** `034` was that slot, and it is the
only prediction the batch got right.
