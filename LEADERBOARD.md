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

**The reference is now proven to interpret the weight *values* correctly** — see below —
which was the outstanding precondition. What is still missing is a *calibrated harness*: an
identity champion that measures 1.00 ± noise. Until that exists there is no baseline, and
`AGENT.md` §4 still governs what to do about it.

### The reference is validated. The harness is not.

**2026-09-07 — the weight-value oracle passed for the first time.**

```
test_reference_greedy_decode_matches_the_oracle_token_for_token  PASSED
```

Our from-scratch `reference.py` greedy-decodes 32 tokens **identically to HuggingFace's own
Qwen3.5-4B**, on the real checkpoint, on an RTX 5090. `AGENT.md` §4 calls this the
precondition for every number downstream of it, and it had never run in nine previous
rentals. It means `head_dim` 256 (not 160), the `1 + weight` RMSNorm convention, the
sigmoid output gate, partial mRoPE, the fp32 recurrent state and the GatedDeltaNet
projection layout are all correct.

**No benchmark ratio exists yet.** Batch 001 ran all nine of its slots and all nine errored
with the same CUDA OOM, the identity champion among them, so the batch is void by its own
rule and reports itself that way. Full account: `results/batches/001-calibration/README.md`.

Seventeen rentals have now been billed across the project, $0.561 lifetime, **zero leaked**.

### The chain of blockers, and where it stands

Each rental that got further than its predecessor did so by exposing the next problem:

| | Blocker | Status |
|---|---|---|
| 1 | Anonymous Docker Hub pulls stall from vast egress ranges | fixed |
| 2 | The image refuses the account's ssh key | fixed, **proven** on a live box |
| 3 | The image has `python3` but no `python` | fixed, proven |
| 4 | `accelerate` absent, so the oracle cannot even be constructed | fixed, proven |
| 5 | Four `oracle_test.py` bounds were fp32-era absolutes applied to bf16 | fixed, proven |
| 6 | Candidate construction double-allocates the 8.4 GB of weights | fixed, **unverified on a GPU** |

Blocker 6 is where the next session starts, and testing it is cheap: if the identity slot
returns 1.00 ± noise, the harness is calibrated and the remaining eight slots are a ~25
minute run. See `docs/GPU-ACCESS.md`.

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
