# Leaderboard

## Champion

**None — no baseline has been recorded yet.**

| | |
|---|---|
| Kernel | — |
| Replaces | — |
| Median ratio vs `torch.compile(max-autotune)` | — |
| IQR of the scoring rounds | — |
| Last verified on | — |
| Result record | — |

**No benchmark has ever run.** Nothing here is estimated, projected, or placeheld. The
first session to get a working GPU records the baseline; until then this table stays empty.

**The reference's reading of the weight *values* is no longer settled** — it matched
HuggingFace token-for-token on 2026-09-07 and diverged on 2026-09-10 and again on
2026-09-12; see below. That was the outstanding precondition, so it is now the outstanding
*question*. The oracle's version is pinned rather than floored since 2026-09-11, which was
right on its own terms and ruled itself out as the cause on the next rental. Also still missing
is a *calibrated harness*: an identity champion that measures 1.00 ± noise. Until both
exist there is no baseline, and `AGENT.md` §4 still governs what to do about it.

### The reference was validated on 2026-09-07, and disagreed on 2026-09-10.

**2026-09-07 — the weight-value oracle passed for the first time.**

```
test_reference_greedy_decode_matches_the_oracle_token_for_token  PASSED
```

Our from-scratch `reference.py` greedy-decoded 32 tokens **identically to HuggingFace's own
Qwen3.5-4B**, on the real checkpoint, on an RTX 5090. `AGENT.md` §4 calls this the
precondition for every number downstream of it, and it had never run in nine previous
rentals. It means `head_dim` 256 (not 160), the `1 + weight` RMSNorm convention, the
sigmoid output gate, partial mRoPE, the fp32 recurrent state and the GatedDeltaNet
projection layout were all corroborated.

**2026-09-10 — the same test failed on rental 27.**

```
assert ours == theirs
At index 2 diff: 11540 != 1528
```

**This claim is therefore no longer settled, and nothing should be built on it until it
is.** What did *not* change: in the same run the logits oracle passed (`relative < 1e-2`
against the tensor's own scale) and so did the mRoPE reduction test, so the architecture
facts above are still corroborated. The disagreement is one argmax at token 2 of 32, with
logits inside the bf16 bound — the exact failure the test's docstring anticipates, since
"small drifts change argmax".

The leading suspect is that **`transformers` moved under us**: the install is
`>=5.16,<6`, a floor rather than a pin, and rental 27 resolved it to 5.17.0 where the
2026-09-07 run predates that release. The assertion compares our tokens to *theirs*, so a
change on their side fails it with nothing in this repo changing. **That is a hypothesis,
not a finding** — one GPU minute settles it. Full account in
`results/batches/002-compile-cost/README.md`.

**No benchmark ratio exists yet.** Batch 001 has now been attempted on two more rentals and
is still void: the identity champion has never returned a number, so nothing else it reports
would mean anything. Full account: `results/batches/001-calibration/README.md`.

What moved on 2026-09-08: candidate *construction* is fixed and proven, the rental path is
fixed and proven, and the failure has relocated to the benchmark's own warmup — a CUDA OOM
at 30.71 GiB of 31.36 with the default four columns. A cold `max-autotune` compile also
turns out to cost ~40 minutes rather than the 3-4 the batch cost model assumes, which is now
the binding constraint on how many hypotheses fit a rental.

Twenty-seven rentals have now been billed across the project, $2.314 lifetime, **zero
leaked**. Five of those were 2026-09-10, which measured no hypothesis and instead found
five defects between renting a box and running one — including a stall guard that destroyed
a healthy rental at the moment its container pull succeeded. See
`results/batches/002-compile-cost/README.md`.

### The chain of blockers, and where it stands

Each rental that got further than its predecessor did so by exposing the next problem:

| | Blocker | Status |
|---|---|---|
| 1 | Anonymous Docker Hub pulls stall from vast egress ranges | fixed |
| 2 | The image refuses the account's ssh key | fixed, **proven** on a live box |
| 3 | The image has `python3` but no `python` | fixed, proven |
| 4 | `accelerate` absent, so the oracle cannot even be constructed | fixed, proven |
| 5 | Four `oracle_test.py` bounds were fp32-era absolutes applied to bf16 | fixed, proven |
| 6 | Candidate construction double-allocates the 8.4 GB of weights | fixed, **proven** — every slot on rental 21 built and passed correctness |
| 7 | Readiness waited on `cur_state` (the rental contract), not `actual_status` (the container) | fixed, proven |
| 8 | The benchmark OOMs at warmup with the default four columns | fix written, **untested** |
| 9 | **The stall guard destroyed a healthy rental** the moment its pull finished | fixed, **proven** — three instances have since passed through that state |
| 10 | A phantom ask traps the deterministic, price-ordered offer search | manual `--exclude-machines` only, **no real fix** |
| 11 | Host driver older than our torch build → CUDA `Error 804` | fixed, **proven** — `DF_MIN_CUDA` placed rental 27 on a working card |
| 12 | The ghcr image ships no Python headers, so Triton's JIT shim will not build | fix written, **untested** |
| 13 | The oracle's greedy decode no longer matches HuggingFace | **open** — the `transformers` explanation was refuted on rental 28 |

Blocker 7 is worth reading even though it is closed: it had been costing rentals since 11
while wearing a convincing disguise as flaky hosts, and the "2 in 17 rentals go to hosts
that never answer sshd" line this file used to carry has been withdrawn.

**The next session still starts on blocker 13**, because an unvalidated reference makes
every number downstream of it meaningless. The `transformers` explanation is dead: 5.17.0
was released 2026-09-09, so the floor already resolved to 5.16.1 on the day the test
*passed*, and rental 28 pinned 5.16.1 and failed byte-identically to rental 27 on different
hardware. Every other input — model code, test, prompt, checkpoint revision, torch, triton,
python — is unchanged between the two.

What is left is the gate itself: exact equality over 32 sequential bf16 argmaxes.
`test_report_the_first_greedy_divergence` measures the top-two logit gap at the diverging
step and writes it to `results/diagnostics/`, asserting nothing. It runs inside the suite
that already runs before every batch, so the number costs no extra GPU time.

Then blockers 8 and 12, and the compile time behind them. See `docs/GPU-ACCESS.md`,
`results/batches/002-compile-cost/README.md`, and
`results/batches/001-calibration/README.md`.

## Baseline

| | |
|---|---|
| Status | **not recorded** |
| Definition | `src/deltaforge/reference.py` under `torch.compile(mode="max-autotune")` |
| Headline workload | batch 1, context 2048, 128 decoded tokens |
| Secondary workload | batch 32, context 2048, 128 decoded tokens |
| Model | `Qwen/Qwen3.5-4B` (see `docs/ARCHITECTURE.md` on why not a newer one) |
| Roofline at headline | 5.11 ms/token, 196 tok/s on an RTX 5090 — `docs/roofline.py` |
| Record | `results/baseline/` (empty) |

## Hypotheses

Nine written and shipped. **None measured.**

| ID | Hypothesis | Replaces | Median ratio | IQR | GPU | Correctness | Outcome | Record |
|---|---|---|---:|---:|---|---|---|---|
| 001 | Fused residual add + RMSNorm | `rms_norm_residual` | — | — | RTX 5090 | not reached (slot OOMed) | `error` — batch void | [dir](results/batches/001-calibration/) |
| 002 | Standalone Triton RMSNorm, hidden-size sites | `rms_norm` | — | — | RTX 5090 | not reached (slot OOMed) | `error` — batch void | [dir](results/batches/001-calibration/) |
| 003 | The same kernel on `q_norm`/`k_norm` | `rms_norm` | — | — | RTX 5090 | not reached (slot OOMed) | `error` — batch void | [dir](results/batches/001-calibration/) |
| 004 | Fused SwiGLU activation | `swiglu_mlp` | — | — | RTX 5090 | not reached (slot OOMed) | `error` — batch void | [dir](results/batches/001-calibration/) |
| 005 | Fused partial mRoPE | `qkv_projection_rope` | — | — | RTX 5090 | not reached (slot OOMed) | `error` — batch void | [dir](results/batches/001-calibration/) |
| 006 | GQA decode without the head expansion | `gqa_attention` | — | — | RTX 5090 | not reached (slot OOMed) | `error` — batch void | [dir](results/batches/001-calibration/) |
| 007 | Fused gated delta-rule step | `gated_delta_rule` | — | — | RTX 5090 | not reached (slot OOMed) | `error` — batch void | [dir](results/batches/001-calibration/) |
| 008 | Split-KV flash decode | `gqa_attention` | — | — | RTX 5090 | not reached (slot OOMed) | `error` — batch void | [dir](results/batches/001-calibration/) |

**2026-09-13 (rentals 30-32): still no ratio, but the reason is now known.** The oracle
gate passes on three independent hosts with zero tie-breaks, and correctness gates pass.
Slot 0 errored in all three rentals, so `CALIBRATION FAILED` fired each time and correctly
voided everything else. The cause of two of those errors was one missing `no_grad` in
`harness/bench.py` (fixed in `f461025`); the third is that a cold `max-autotune` compile
does not finish inside a session — which no run has ever managed. See
`results/batches/001-calibration/README.md`.

**All nine slots carry no ratio, and every prediction is unscored.** They are written,
gated on CPU, and shipped; they simply have not been measured. A slot that errored never
tested its prediction, so `summary.json` records `0 correct of 0 scored` rather than 0 of 9
— counting an untested prediction as wrong would understate the record exactly as counting
it right would flatter it.

**001 is no longer graveyarded on mechanism.** It was closed on the argument that a 0.018%
ceiling is not worth a rental — an argument about *cost*, which batching dissolves. At
roughly three minutes a slot it is worth measuring, and it is now slot 1 of batch 001
awaiting a working harness. Its arithmetic still stands as the *prediction*; what changed is
that the prediction is now falsifiable in practice. Same for 004 and 005.

### Column definitions

- **ID** — `NNN`, matching the branch `hyp/NNN-slug` and `results/hypotheses/NNN-slug.json`.
- **Hypothesis** — one line, stating the mechanism expected to produce the win.
- **Replaces** — the reference operation the kernel substitutes, from `REPLACEABLE_OPS`.
- **Median ratio** — `median(t_compiled / t_candidate)` over the scoring rounds. Greater
  than 1 means faster than the compiler. This is the score; absolute milliseconds are
  provenance and live in the result record.
- **IQR** — interquartile spread of those per-round ratios: the noise band. A margin
  smaller than this is not a win.
- **GPU** — the card it was measured on. Every session rents a different one, so a ratio
  is only comparable within its own run.
- **Correctness** — `pass` only when both gates pass. Otherwise the worst error magnitude,
  or the prompt and token index where the end-to-end match diverged.
- **Outcome** — one of:
  - `win` — beat the incumbent by more than the IQR. Promoted to champion.
  - `loss` — slower. Goes to the graveyard with the mechanism that failed.
  - `inconclusive` — inside the noise band. Neither promoted nor buried; the hypothesis
    stays open for a cleaner measurement.
  - `graveyarded on mechanism` — closed by arithmetic before or instead of measurement,
    because its ceiling is below the noise band. Cheaper than measuring, and a legitimate
    result: `docs/HYPOTHESES.md` records the numbers that closed it.
  - `incorrect` — failed a correctness gate. Recorded with its error magnitudes, because a
    candidate that was fast but wrong is among the most useful things to read.
