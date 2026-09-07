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

**What the first working session should do** is in `AGENT.md` §4 — the weight-value oracle,
then an *identity champion* to calibrate the harness, then a profile. Not a kernel. The
reference is currently proven structurally correct — shapes against the real checkpoint,
cache contract, causality, the mRoPE reduction — but not yet proven to interpret the weight
*values* correctly, and every number depends on that.

### GPU access: cause found and fixed, not yet through to a benchmark

Provisioning works. Nine instances have been created, billed and destroyed (98.8 min,
$0.5054, **zero leaked**). None produced a number.

**Rentals 1–8 (2026-09-03) all died on a Docker Hub image pull.** Four machines, three
images, 2.5–14.1 GB — not one layer ever reached "Pull complete".

**Rental 9 (2026-09-05) identified the cause.** Same filters, same account, one variable
changed: a `ghcr.io` image instead of Docker Hub. It **pulled in about three minutes and the
container started** — the first time that has ever happened. Anonymous Docker Hub pull limits
are the cause; the never-paid-account theory is ruled out, since the account state was
unchanged.

That rental then hit a **second blocker behind the first**: the container ran sshd and
refused our key (`Permission denied (publickey)`), because a non-Vast image provisions
`authorized_keys` by its own convention. Fixed by injecting the key through both
`PUBLIC_KEY` and `onstart`. Billed 4.57 min, $0.0271.

Three fixes are in and tested: registry credentials via `image_login`, explicit key
injection, and a stall budget on **both** readiness loops so a stuck run costs ~$0.03
instead of $0.12. Full account and what to do next: **`docs/GPU-ACCESS.md`**.

**Still unproven:** that the injected key is accepted, and therefore that any run gets past
sshd to the gates. That is what the next rental tests.

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

One attempted, none measured.

| ID | Hypothesis | Replaces | Median ratio | IQR | GPU | Correctness | Outcome | Record |
|---|---|---|---:|---:|---|---|---|---|
| 001 | Fused residual add + RMSNorm | `rms_norm_residual` | — | — | — | CPU gates pass; GPU gates never ran | **graveyarded on mechanism** | [dir](results/hypotheses/001-fused-rmsnorm-residual/) |

**001 carries no ratio and never will.** It was graveyarded by arithmetic rather than by
measurement: the operations it fuses move 0.018% of per-token bytes, so its ceiling is
below the harness's own noise band. The eight failed rentals are incidental — even a clean
measurement could not have shown a win. See `docs/HYPOTHESES.md` for the full reasoning and
the lesson that reordered the backlog.

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
