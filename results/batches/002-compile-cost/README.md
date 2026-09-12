# Batch 002 — compile cost. Never started, and the reason is worth more than the batch.

**No slot in this batch ran.** The batch loop was never reached on any of five rentals, so
there is no ratio, no compile timing, and no calibration. `002-compile-cost` remains
entirely unmeasured, and the arithmetic it was built to settle — is a rental ~15 minutes
fixed plus 2-4 per slot, or is one cold compile ~40 minutes? — is still an estimate that no
measurement supports.

What this session bought instead was **five defects in the path between renting a box and
running a hypothesis**, four of them fixed with tests, and one genuine contradiction of a
claim the project has been treating as settled.

---

## The one that is not a tooling bug

`test_reference_greedy_decode_matches_the_oracle_token_for_token` **failed** on rental 27:

```
assert ours == theirs
At index 2 diff: 11540 != 1528
```

This test passed on 2026-09-07 and `LEADERBOARD.md` has carried "the reference is
validated" ever since. On 2026-09-10 it does not hold on the stack it was run on.

**What it is not.** It is not the missing Python headers that killed the other eleven
failures in the same suite: `reference.py` contains no Triton and the oracle path never
calls `torch.compile`, so the shim that could not build is not on this code path. It is
also not a structural model error of the kind `AGENT.md` §8 lists — in the same run, the
logits oracle **passed** (`relative < 1e-2` against the tensor's own scale) and so did the
mRoPE reduction test. Head dim, the `1 + weight` convention, the gate type and the RoPE
section are all still corroborated.

**So the disagreement is exactly one argmax, at token 2 of 32, with logits that still
agree within the bf16 bound.** That is the failure mode the test's own docstring predicts:
"Logit closeness is necessary but not sufficient: small drifts change argmax."

**Leading suspect: `transformers` drifted.** The install is
`'transformers>=5.16,<6'`, which is a *floor, not a pin* — it was chosen to stop the Qwen3.5
architecture disappearing mid-run, not to hold behaviour fixed. Rental 27 resolved it to
**5.17.0**; the run that passed on 2026-09-07 predates that release. Since the assertion
compares our tokens against *HuggingFace's*, a change on their side moves `theirs` and
fails the test without anything in this repo changing.

**This is a hypothesis and it has not been tested.** Ruling it in or out costs one GPU
minute — pin `transformers==5.16.*`, re-run the one test — and until someone does, the
honest statement is that the reference is validated **against transformers 5.16 on
2026-09-07** and contradicted against 5.17.0 on 2026-09-10, cause unknown. Do not promote
either reading.

The gate behaved correctly throughout: the GPU suite refused to let the batch run on an
unvalidated reference, which is precisely what it is for.

## Five rentals, and what each one bought

| # | Instance | GPU | Billed | Cost | Died on |
|---|---|---|---:|---:|---|
| 23 | 50526206 | RTX 4090 | 8.17 min | $0.0402 | Docker Hub pull refused (blocker 1, recurred) |
| 24 | 50526868 | RTX 4090 | 5.65 min | $0.0310 | **stall guard killed a healthy box** (blocker 9) |
| 25 | 50528615 | RTX 4090 | 11.78 min | $0.0687 | host reported `GPU error, unable to start instance` |
| 26 | 50529356 | RTX 4090 | 6.08 min | $0.0353 | CUDA `Error 804`, driver older than our wheels (blocker 11) |
| 27 | 50530926 | RTX 5090 | 88.55 min | $0.6242 | `Python.h` missing; oracle divergence (blockers 12, 13) |

**$0.7995 for the session, 120.23 billed minutes, zero leaked instances.** Three further
attempts were refused by the API before any instance existed and cost nothing.

The session ended the way `AGENT.md` §5 says it should: the pre-flight gate refused the
next rental with exit 5 — 59.8 minutes left against the 143 a cold hypothesis needs — and
the remaining work needed no GPU.

## The defect that matters most, because it was ours

**Rental 24 was healthy and we destroyed it.** The stall guard fired at 300s on

```
2741c81b500d: Verifying Checksum2741c81b500d: Download complete
```

The guard's premise is that a live pull keeps rewriting `status_msg` with new byte counts,
so a frozen message means a stuck one. That premise **expires the moment the last layer
finishes downloading**: verification, extraction and container start emit no updates, so a
healthy instance necessarily goes quiet in exactly the window before it becomes reachable.
The guard fired at the one point where the thing it guards had stopped being possible.

This is blocker 7's lesson inverted. That one was *a guard firing before the thing it
guarded was ever reached*; this one is *a guard firing after it had already succeeded*. The
loop even contained the right reasoning one branch lower, for a different cause: an empty
`status_msg` was already exempted because applying the budget to it "would destroy healthy
instances at 300s for the crime of being quiet".

Fixed in `bf2eb9d`, and **proven on a GPU**: rentals 26 and 27 both walked through the
identical message and reached sshd.

## What is now settled

* **Blocker 9 (stall guard) — fixed and proven.** Three instances passed through the
  settled-pull state; two reached sshd, one reached the GPU suite.
* **Blocker 11 (driver vs wheels) — fixed, proven by selection.** `DF_MIN_CUDA` defaults to
  12.8 to match the cu128 index, and rental 27 was placed on a CUDA 13.0 card and got
  `torch 2.11.0+cu128 … NVIDIA GeForce RTX 5090` where rental 26 had died.
* **A refused create now explains itself.** `RESPONSE=$(df_api …)` was a plain assignment,
  so `set -e` killed the run on `curl: (22)` one line before the diagnostic written to
  explain it. The body was fetched, stored, and discarded. It paid off immediately:
  `no_such_ask  Instance type by id 49864003 is not available`.
* **Blocker 12 (Python headers) — fixed, unproven.** Installed beside g++ and before the
  suite; no rental has run since.

## What is still open

1. **The oracle divergence** above. Still open, but no longer unexplained: PyPI release
   dates show the floor resolved to 5.16.1 on 2026-09-07 and to 5.17.0 on 2026-09-10, with
   5.17.0 released 2026-09-09 — between the two runs. The install is now pinned to
   `transformers==5.16.1` (2026-09-11). That restores the validated stack; it does not
   prove the pin is the cure, and the next rental's oracle result is what does.
2. ~~**Blocker 10 — a phantom ask traps the offer search.**~~ **Fixed 2026-09-11.**
   `select_offers` emits the `DF_OFFER_CANDIDATES` cheapest offers and the create step
   walks them, treating a refusal as "try the next". `try_create_instance` returns a status
   where `create_instance` used to `df_die`. Two tests drive provision.sh against a stub
   API — one refuses the two cheapest asks and asserts the third is created and is the
   offer the ledger names, the other refuses all of them and asserts exit 4 with no ledger
   row.
3. **Everything batch 002 was for.** Compile cost, warm vs cold, and whether two scoring
   columns survive the benchmark — none of it has been observed. Rental 27 reported
   `compile workers: 96 cores`, which kills the leading theory for rental 22's ~40-minute
   compile (a one-core container), but it never compiled anything, so that is a theory
   removed rather than a number obtained.
4. **The image is now a fork in the road.** The Docker Hub default failed on the first
   host; `ghcr.io/ai-dock/base-image` pulls reliably but ships no Python headers and no
   `python`-as-`python3`. Both gaps are patched at run time. Whether to make ghcr the
   default should be decided by the next session that gets through, not now.

## Cost of the session in one line

120.23 billed minutes, $0.7995, 5 rentals, 0 leaked, 0 hypotheses measured — and the
project's 27th rental was the first to reach the GPU test suite with a validated path in
front of it.
