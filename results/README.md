# Results

**Empty. No benchmark has been run and no baseline has been recorded.**

The bootstrap session had no GPU, so every number-producing step is wired and deferred
rather than faked. Nothing in this directory is estimated or placeheld; when a file
appears here it came off a real card.

## Layout

| Path | Contents |
|---|---|
| `baseline/<gpu>-<date>.json` | The reference under `torch.compile(max-autotune)`, re-recorded each session. |
| `hypotheses/NNN-slug.json` | One record per hypothesis run, **win or lose**. |

## Record format

Written by `deltaforge.harness.report.ResultRecord`, schema version 1. Every field exists
so a reader can argue with the result rather than take it on trust.

| Field | Contents |
|---|---|
| `schema_version` | Record format version. |
| `kind` | `baseline` or `hypothesis`. |
| `outcome` | `baseline`, `win`, `loss`, `inconclusive`, `incorrect`, or `error`. |
| `timestamp` | UTC, ISO 8601. |
| `hypothesis` | `{id, slug, statement}` — the mechanism expected to produce the win. |
| `workload` | Batch size, context length, decoded tokens. |
| `git` | `{sha, branch, dirty}`. A dirty tree is recorded, not hidden. |
| `environment` | GPU model, driver, observed clocks, torch/Triton/CUDA versions, compute capability. |
| `bench` | Per-round raw timings for all four columns, median ms, per-round ratios, median ratio, IQR, and the actual call order. |
| `correctness` | Layer-1 per-kernel magnitudes and layer-2 token sequences. |
| `cost` | Instance id, GPU, hourly rate, minutes, dollars. |
| `notes` | Anything a reader needs that the fields do not cover. |

### Why the raw timings are kept

`bench.timings_ms` holds every round including the two discarded warmup rounds, and
`bench.call_order` records the exact sequence of timed calls. Together they let a reader
verify that the measurement really was interleaved and recompute the score independently.
A median ratio you cannot recompute is a number someone typed.

### Why clocks are recorded but not locked

Clock locking (`nvidia-smi -lgc`) needs privileged container access we will not reliably
have, so the methodology never depends on it. Observed clocks are captured as provenance;
interleaving the four columns within each round is what absorbs thermal drift and
noisy-neighbour effects.

### Why absolute milliseconds are not the score

Every session provisions a different physical GPU. Absolute times are not comparable
across sessions and must never be the win condition. The score is
`median(t_compiled / t_candidate)` measured within one process on one card.

## Failures are recorded, not discarded

A candidate that was fast but incorrect gets a record here with its error magnitudes, and
a graveyard entry in `docs/HYPOTHESES.md`. That is among the most valuable things a future
session can read, and deleting it would make the log a highlight reel.
