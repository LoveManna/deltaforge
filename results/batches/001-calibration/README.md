# Batch 001 — calibration. Void again, and further along.

**No hypothesis in this batch has been measured.** Two rentals on 2026-09-08 (21 and 22)
reached the batch loop; neither produced a ratio. The identity champion never returned a
number, so the batch is **void by its own rule** — `calibrated: false`,
`counts: {"error": 9}`, `prediction_record: {"correct": 0, "scored": 0}`.

Not one of the nine predictions was scored. A slot that errored never tested its
prediction, and counting those as wrong would understate the record exactly as counting
them right would flatter it.

The JSON records here are **rental 21's**. Rental 22 completed no slot, so it overwrote
nothing.

---

## What is now settled

**Blocker 6 is fixed and proven.** Rental 16 lost all nine slots inside
`BatchRunner._build_candidate` to a double allocation of the weights. On rental 21 every
slot built its candidate and cleared the correctness gates. Construction is no longer where
this dies, and the meta-device fix in `1d5f2a9` needs no further testing.

**Blocker 7 was found and fixed** — the readiness poll waited on the rental contract rather
than the container. See `docs/GPU-ACCESS.md`; it had been costing rentals since 11 and
masquerading as flaky hosts.

**A kernel bug that had never run.** `007-gated-delta-fused-step` errored in 0s with
`ImportError: cannot import name 'STATE_DTYPE' from 'deltaforge.config'` — it lives in
`reference`. Kernels import from the package inside function bodies to keep Triton off the
CPU import path, so nothing checked that name until the function ran on a rented GPU.
`registry_test.py` now resolves every deferred relative import statically, and reverting
the fix reproduces the rental's exact message on CPU.

## Where it dies now: the benchmark, not the build

Rental 21 lost eight slots to a CUDA OOM at **30.71 GiB of 31.36**, every one of them at
the *benchmarking* step, 16-18s in — past construction, past correctness. The batch
recorded peak memory only for slots that **succeeded**, which is precisely the empty set
when memory is the problem, so nine identical failures said 30.71 GiB and nothing about
where it went. That instrumentation now runs on the error path too.

Rental 22 ran with only the two scoring columns and reported the breakdown:

| After | Allocated |
|---|---:|
| loading weights | 7.83 GiB |
| reference column `compiled` | 7.95 GiB |
| candidate build | 7.96 GiB |
| candidate column `candidate_compiled` | 8.07 GiB |

**Construction is nearly free.** Weights are 7.83 GiB and the candidate shares them, adding
0.11 GiB. Four columns cannot cost 30 GiB at construction either — so the ~22 GiB in rental
21 was accumulated *during warmup*, by the two eager columns that the default includes and
the scoring pair does not need. `run_remote.sh --columns` now exists to say so; batch mode
had always accepted the flag and nothing passed it.

**This is a hypothesis, not a result.** Rental 22 was stopped by the session gate before
warmup completed, so the two-column configuration has never been observed to survive the
benchmark. It is the next thing to test, and it is cheap to test.

## The wall in front of the next session

Rental 22 spent **~40 minutes inside slot 0 without finishing it.** `nvidia-smi` showed the
GPU at 0% with python at 129% CPU and an inductor worker alongside: a cold
`max-autotune` compile, not a hang.

That breaks the arithmetic the whole batch design rests on. `AGENT.md` and
`docs/BATCHES.md` cost a rental at ~15 minutes fixed plus 2-4 per slot, which is what makes
7-12 hypotheses per rental worth doing. A 40-minute first compile means a nine-slot batch
does not fit a 90-minute session at all, and the session gate — not the science — decided
when this run ended.

**Do not fill another batch until that number is known.** Time one `max-autotune`
compilation, then either bring it down (`mode="reduce-overhead"`, a warm inductor cache
carried between rentals, `torch.compiler` caching to disk) or re-cut the batch size around
what a compile actually costs. Either is a better use of the next rental than nine more
slots that will not run.

## What stands between this batch and a number — 2026-09-11

No rental happened on this date; this is a bench session that cleared what it could clear
without one. Four things were in the way, and two of them are now gone.

**Cleared: the pull.** Docker Hub credentials are in `.env`, so the create request carries
`image_login` and the default `vastai/*` image is pulled authenticated. Blocker 1 recurred
as recently as rental 23, on the default image, so this was live and not historical.

**Cleared: the trap in the offer search.** Blocker 10 had no fix, only `--exclude-machines`
applied by a human who knew to reach for it. The search now emits the five cheapest offers
and the create step walks them, so a listed-but-unrentable ask costs one warning instead of
a run. Proven by tests against a stub API, not by a GPU.

**Restored, not proven: the reference.** `transformers` was installed from a floor, which
let the oracle move between rentals; it is now pinned to 5.16.1, the version that passed on
2026-09-07. The GPU suite gate refuses to run a batch on an unvalidated reference, so until
that test passes on a card, batch 001 cannot legally produce a number. **This is the one
remaining hard blocker, and it is settled by the first rental that runs.**

**Untouched: the arithmetic.** A cold cache still needs ~131 minutes to reach the end of
the *second* slot — 1500s setup, 2400s reference compile, two slots at 1980s — against a
180-minute gate. Batch 001 has nine slots. **It will not finish in one rental and should
not be expected to.**

That is not a reason to re-cut it. The batch stops itself before a slot it cannot finish,
records each slot as it completes, and the compile cache now comes home, so the honest
shape of this batch is:

* **Rental A (cold).** Oracle result settles blocker 13. Slot 0 calibrates. Slot 1 is the
  first measured hypothesis this project has ever produced. The batch stops itself, and
  `phases.env` comes home carrying the first *measured* compile cost — which replaces the
  pessimistic cold estimates in `batch.COLD_PHASE_ESTIMATES` for every rental after it.
* **Rental B (warm, same card).** The pre-flight gate now reasons from measured numbers
  instead of guesses, and the batch resumes into the slots A could not reach.

Two rentals to finish batch 001 is the plan, not a failure of one. The alternative —
shrinking the batch to fit a cold cache — throws away the only thing that makes the
remaining slots cheap.

## Cost

Five rentals, 91.39 billed minutes, $0.567 — session `batch001-20260908T211825Z`, which
ended at its 90-minute gate. Zero leaked instances. Three of the five died to blocker 7
before it was understood.
