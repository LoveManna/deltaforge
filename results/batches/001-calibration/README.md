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

## Rental 28, 2026-09-12 — the transformers hypothesis is dead

**No slot ran. The batch loop was not reached.** The GPU suite refused it, which is what
the suite is for. Session `batch001-20260912T021420Z`: instance 50682531, RTX 4090, machine
5629, $0.4030/hr, **7.12 billed minutes, $0.0478**, destroyed cleanly, nothing leaked.

**Two things worked for the first time.**

*The authenticated Docker Hub pull.* Credentials went into `.env` and the create request
carried `image_login`. The default `vastai/base-image` pulled and sshd answered in about a
minute. Eight rentals never got one layer to "Pull complete" anonymously and rental 23 hit
the same wall on 2026-09-10, so blocker 1 is now fixed *and proven*, not fixed and hoped.

*The offer walk.* No RTX 5090 met the filters, the search fell through to the 4090, found
one candidate, and created on the first try. Blocker 10's fix was exercised on a real
market; it was not stressed, because nothing refused.

**And the thing that was supposed to work did not.** With `transformers==5.16.1` installed
and verified in the log (`accelerate 1.15.0 transformers 5.16.1`):

```
FAILED src/deltaforge/oracle_test.py::test_reference_greedy_decode_matches_the_oracle_token_for_token
At index 2 diff: 11540 != 1528
```

**The same index. The same two token ids. As rental 27.** Rental 27 was an RTX 5090 running
transformers 5.17.0; rental 28 was an RTX 4090 running 5.16.1. Neither the card nor the
library version moved the result by a single token.

### The hypothesis is refuted, and refuted cleanly

The floor `>=5.16,<6` resolved to **5.16.1 on 2026-09-07 as well** — 5.17.0 did not exist
until 2026-09-09. So the rental that passed and the rental that failed ran *the same
transformers*. The release-date argument that made 5.17.0 look guilty was real but
irrelevant: it established that the input *could* have moved, not that it *did* move on the
day that mattered. Pinning was still correct — an unpinned oracle is a defect whatever the
outcome — but it was not the cause, and blocker 13 is not fixed.

### What is left, after everything checkable was checked

| input | 2026-09-06 (passed) | 2026-09-12 (failed) |
|---|---|---|
| `reference.py`, `model.py`, `config.py`, `weights.py` | unchanged since 2026-09-06 | identical |
| `oracle_test.py` and its prompt | unchanged since 2026-09-06 | identical |
| `transformers` | 5.16.1 | 5.16.1 |
| checkpoint `Qwen/Qwen3.5-4B` | last modified 2026-03-02 | identical |
| torch / triton / python | 2.11.0+cu128 / 3.6.0 / 3.12.3 | identical |
| GPU | RTX 5090 | RTX 4090 (and rental 27's 5090 diverged identically) |

Every input anyone has named is constant. That reframes the question: not *what moved*, but
**whether exact token equality was ever the right gate.**

### The reading that now fits the evidence

The test is an exact-equality check over 32 sequential argmaxes in bf16. The prompt is
"The chunked delta-rule recurrence is a sequential scan with matrix-valued state." Both
models emit `\nThe`; ours then continues by echoing the prompt (`chunked`), HuggingFace by
picking `state` — token 14 of the same prompt. **Both are ordinary continuations**, and
they are exactly the kind of pair a sub-ULP difference reorders.

Note what commit `5722aaa` did on 2026-09-06. It found three fp32-era bounds in this very
file that no correct bf16 implementation could meet, fixed all three to score relative to
the logit scale, and left *this* test alone — deliberately, "until the oracle could
adjudicate". The oracle adjudicated once, in its favour, and has now contradicted itself
twice without a single input changing. A gate that flips on an unchanged system is
measuring something other than what it claims to.

**That is a reading, not a result, and it must not be promoted without the number.** A
correct reference and a fragile gate look identical from here; so does a real bug in the
reference that the logits test is too coarse to see.

### The number that decides it, and what it costs

`test_report_the_first_greedy_divergence` is now in `oracle_test.py`. It asserts nothing,
fails nothing, and changes no gate. It replays both models to the diverging step and writes
to `results/diagnostics/oracle-divergence.json` — which teardown pulls home before
destroying the instance, as it did on this rental:

* how decisively each model preferred its own token, in logits;
* each model's own top-2 gap at that step;
* the largest disagreement between the two models' logits anywhere in that row, and one
  bf16 ULP at that scale.

If the gap that separates `11540` from `1528` is smaller than the disagreement the two
models show anyway — measured at 1-2 ULP (0.28125) on 2026-09-07 — they are splitting a
tie and the gate is wrong. If it is much larger, the reference has a real bug and this
gate has been right all along and rental 50123509 was the fluke.

Rental 28 cost $0.0478 to reach the GPU suite. **The number costs about the same**, because
the suite runs before the batch regardless and this test runs inside it.

## What stood between this batch and a number — 2026-09-11

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

> **Rental 28 settled it, against the pin.** The floor had already resolved to 5.16.1 on
> 2026-09-07, so the pin changed nothing about that day's stack and the test failed
> identically. See the section above; this paragraph is kept as written because the
> prediction it made is what the rental tested.

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
