# Batch 001 — calibration. Void, and why.

**No hypothesis in this batch was measured.** All nine slots errored with the same CUDA
out-of-memory, the identity champion among them, so the batch is **void by its own rule**:
if the calibration slot cannot show the harness measuring 1.00 on a candidate that *is* the
reference, nothing else it reports means anything.

Recorded here for diagnosis, not as findings. `summary.json` says the same thing in the
machine-readable record: `calibrated: false`, `counts: {"error": 9}`,
`prediction_record: {"correct": 0, "scored": 0}`.

**Not one of the nine predictions was scored, and that is the correct handling.** A slot
that errored never tested its prediction; counting those as wrong would understate the
record exactly as counting them right would flatter it.

---

## What failed

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 46.00 MiB.
GPU 0 has a total capacity of 31.36 GiB of which 43.88 MiB is free.
Of the allocated memory 30.63 GiB is allocated by PyTorch.
```

Every slot, at the same place: `BatchRunner._build_candidate`. Between 32 and 58 seconds
each, on an RTX 5090 with 31.36 GiB usable.

**The cause is a double allocation of the model weights.**
`ReferenceModel(config).to("cuda")` materialises a full fresh 8.4 GB of parameters, and
only *then* does `load_state_dict(assign=True)` rebind them onto the reference's tensors and
free the duplicates. Peak is 16.8 GB of weights for a model that needs 8.4.

That is survivable in the single-hypothesis path — it happens once, before anything has been
compiled. It is fatal in a batch, because batch mode's whole saving is keeping the
reference's compiled state resident, and that state is already on the card when each
candidate is built. The batch design created the pressure that exposed a latent bug in code
it inherited.

**Fixed after this run** (`batch_run.py`): candidates are constructed under
`torch.device("meta")`, which allocates nothing, then bound to the reference's tensors by
assignment. Non-persistent buffers need explicit handling — `rotary_emb.inv_freq` never
appears in a `state_dict`, so `assign=True` leaves it on meta and the first forward dies
with "Cannot copy out of meta tensor". Verified on CPU with the tiny config: no meta
leftovers, parameters shared, forward bit-identical to the reference.

**The fix is unverified on a GPU.** The session's 90-minute gate was reached before another
rental could test it. That is the first thing the next session should do, and it is cheap:
if slot 0 comes back at 1.00 ± noise, the harness is calibrated and the remaining eight
slots are a ~25 minute run.

## What this run did prove

The failure was total, which made it an unusually complete test of the machinery around it.
Every one of these is a property batch mode promised and had never demonstrated:

| Promise | Evidence |
|---|---|
| A failing slot costs a slot, not the rental | Nine slots failed; the run continued through all nine and exited 0 |
| Records are written as each slot finishes | All nine `<slug>.json` files present |
| Teardown pulls results before destroying | All ten files reached the local checkout from a run that failed |
| A void batch says so | `calibrated: false`, and the markdown leads with **CALIBRATION FAILED** |
| Unscored predictions are not scored | `prediction_record: {"correct": 0, "scored": 0}`, not 0/9 |

Under the old one-hypothesis-per-rental workflow this OOM would have produced a single
failed run and no record at all.

## Provenance

| | |
|---|---|
| Session | `batch001-20260907T032449Z` |
| Instance | 50125608, RTX 5090, $0.3830/hr, 17.35 min |
| Environment | torch 2.11.0+cu128, Triton 3.6.0, CUDA 12.8, driver via host |
| Workload | headline — batch 1, context 2048, 128 decoded tokens |
| Commit | see `git` block in `summary.json` |

## The one number this session did produce

Not in this directory, because it is not a benchmark: on instance 50123509 the
**weight-value oracle passed**. `test_reference_greedy_decode_matches_the_oracle_token_for_token`
greedy-decodes 32 tokens identically to HuggingFace's own Qwen3.5-4B.

`AGENT.md` §4 calls that the precondition for every number downstream of it. It had never
run before this session. It passes, which means `head_dim` 256, the `1 + weight` RMSNorm
convention, the sigmoid output gate, partial mRoPE, the fp32 recurrent state and the
GatedDeltaNet projection layout are all correct.

That result is worth more than the nine ratios this batch failed to produce, and it is not
void — it does not depend on the harness being calibrated, only on the model being right.
