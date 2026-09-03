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

The bootstrap session built the harness, the reference implementation, the correctness
gates and the cost machinery. It had no GPU available — the Vast.ai account was unfunded
and the local machine has no CUDA — so **no benchmark has been run and no number exists.**

Nothing here is estimated, projected, or placeheld. The first funded session records the
baseline; until then this table stays empty.

**Before the first hypothesis:** run the deferred correctness oracle
(`pytest -m "gpu and weights"`). The reference is currently proven structurally correct —
shapes against the real checkpoint, cache contract, causality, the mRoPE reduction — but
not yet proven to interpret the weight *values* correctly. Every subsequent number depends
on that.

**GPU access is blocked again, for a different reason.** The account is email-verified
and carries signup credit, and provisioning works — eight instances were created, billed
and destroyed on 2026-09-03. None of them ever finished pulling a container image, across
four machines, three images and two registries. The account has never made a payment
(`paid_verified: 0.0`, `has_billing: false`), which is the strongest remaining explanation:
instances are created and billed but not permitted to pull. Adding a payment method and
re-running `remote/run_remote.sh` unchanged is the cheap test. Full account of the eight
rentals: `results/hypotheses/001-fused-rmsnorm-residual/README.md`.

## Baseline

| | |
|---|---|
| Status | **not recorded** |
| Definition | `src/deltaforge/reference.py` under `torch.compile(mode="max-autotune")` |
| Headline workload | batch 1, context 2048, 128 decoded tokens |
| Secondary workload | batch 32, context 2048, 128 decoded tokens |
| Record | `results/baseline/` (empty) |

## Hypotheses

None attempted.

| ID | Hypothesis | Replaces | Median ratio | IQR | GPU | Correctness | Outcome | Record |
|---|---|---|---:|---:|---|---|---|---|
| — | — | — | — | — | — | — | — | — |

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
  - `incorrect` — failed a correctness gate. Recorded with its error magnitudes, because a
    candidate that was fast but wrong is among the most useful things to read.
