# Batch 005 — the first win, and the slot that proved it had not been tested

> **Rental 40, 2026-09-19.** RTX 5090, instance 51508818, machine 140887, $0.4363/hr,
> **32.32 billed minutes, $0.2350**, destroyed cleanly. Rental 39 died first, in the repo
> sync, for a further 3.53 minutes and $0.0257. Lifetime **$6.706 across 40 rentals**.
>
> `calibrated: true` — `000-identity` measured **1.0008** at an IQR of **0.00017**, the
> tightest calibration this project has recorded. `graphs_compiled: 3` on every slot that
> ran. Six slots ran, two declined.
>
> **Predictions scored 3 of 5.**
>
> **`022-int4-head` is the first candidate in this project to beat
> `torch.compile(mode="max-autotune")`: median ratio 1.0791, IQR 0.00034.** It is the
> champion. `025-fused-causal-conv` is the second win, at 1.0144.

---

## The scorecard

| Slot | Outcome | Ratio | IQR | Layer 1 | Layer 2 | Predicted |
|---|---|---:|---:|---|---|---|
| `000-identity` | calibrated | **1.0008** | 0.00017 | — | exact | `identity` ✅ |
| `021-static-cache-cudagraphs` | `loss` | 0.9986 | 0.00058 | pass (0.0) | 264/264, 0.00000 nats | `win` ❌ |
| `022-int4-head` | **`win`** | **1.0791** | 0.00034 | pass (7.8e-3) | 0.9318, 0.01674 nats | `win` ✅ |
| `023-int8-head` | `incorrect` | 1.0373 | 0.00011 | **fail — rel 4511** | 0.4394, 9473 nats | `win` ❌ |
| `024-fp8-head` | `error` | — | — | — | — | `win` (untested) |
| `025-fused-causal-conv` | **`win`** | **1.0144** | 0.00107 | pass (**0.0**) | 264/264, 0.00000 nats | `win` ✅ |
| `026-int4-head-static-cache` | `precondition_failed` | — | — | — | — | `win` (untested) |
| `027-conv-head-static-cache` | `precondition_failed` | — | — | — | — | `win` (untested) |

Baseline on this card: **857.5 ms per 128 tokens = 6.70 ms/token, 8587.80 MB/token,
1282 GB/s — 71.5% of an RTX 5090's 1792 GB/s vendor peak.** Faster than rental 38's 1177,
on a different physical card; ratios are within-slot and absolute times are provenance.

---

## 1. The win, and why this site and not the other 248

`022-int4-head` installs group-128 int4 on the **tied LM head and nothing else** —
248320 × 2560, 1271.40 MB/token, 14.80% of everything the compiled column moves, in one
matmul. Its ceiling was 1.123×; it returned **1.0791**, collecting **70% of the available
saving**.

Two rentals had already run a hand-written GEMV on this model and lost badly — 0.2801 with
one kernel structure, 0.1934 with the opposite one. Both installed on **all 248 layer
projections at once** and reported one aggregate byte rate, which cannot tell a kernel that
is slow everywhere from one that is slow where there is no parallelism to have. This batch
separated them, and the separation is the whole result:

| Where the same kernel family ran | achieved |
|---|---:|
| `009` naive GEMV, 248 layer projections (rental 37) | 319 GB/s |
| `015` tiled GEMV, 248 layer projections (rental 38) | 228 GB/s |
| `022` int4, **the LM head alone** (rental 40) | **656 GB/s** |
| `023` int8, **the LM head alone** (rental 40) | **847 GB/s** |

The head-site figures are backed out of the measured ratio: the candidate differs from the
baseline only at that one site, so its head time is
`t_head_baseline + (t_candidate − t_baseline)` = 0.992 + (6.208 − 6.701) = **0.499 ms** for
327.78 MB, and 0.751 ms for 635.7 MB at int8. (`t_head_baseline` assumes the baseline moves
the head's bytes at its own model-wide rate; it is the only estimate in this section.)

**At BLOCK_N=64 the head launches 3880 programs; `in_proj_a` is 32 channels wide and
launches four.** Batch 004's writeup listed three suspects for its kernel's slowness — flop
padding, split-K overhead, forfeited fusion — and this rental adds the one it could not see,
which turns out to dominate: **on the sites it was measured on, the kernel never had enough
output parallelism to fill the card.** The flop padding and the split-K pass are unchanged
here and the kernel is 2-3× faster, so neither of them was the binding cost.

### int8 against int4 answers the question two rentals could not

Ranked by **byte rate**, int8 beats int4: 847 against 656 GB/s. The nibble unpack costs
**29% of the achieved bandwidth**, which is real and is on the critical path. Ranked by
**time**, int4 wins, because it moves half as many bytes and 2 × 0.71 > 1.

So the kernel at this site is **substantially bandwidth-bound with a measurable unpack
tax** — not the issue-bound regime batch 003 diagnosed, where removing bytes made the
kernel *slower*. That regime was a property of the sites, not of the kernel.

> `023`'s ratio is recorded as provenance for that comparison, **not as a result**: the
> kernel was wrong (§3). The defect is one missing multiply in the epilogue, worth a
> fraction of a microsecond, so the timing is usable as a byte-rate datum while the slot
> stays `incorrect`.

---

## 2. The second win: 72 launches that move 0.05% of the bytes

`025-fused-causal-conv` replaces the `cat` + cuDNN `extern_kernels.convolution` + `copy_`
triple in each linear-attention layer with one Triton kernel. **1.0144**, IQR 0.00107, and
**layer 1 at exactly 0.0 absolute error at batch 1 and batch 32** — bit-identical, which is
what the kernel was designed for: fp32 accumulate, round to bf16, *then* silu, because
`F.conv1d` returns bf16 and `F.silu` runs on that.

It saved **0.0934 ms/token of 6.70, or 1.39%**, against a registered prediction of
0.08-0.24 ms. Achieved bandwidth went **up**, 1282 → 1300 GB/s, on identical bytes — which
is what removing launches from a step looks like.

This is the smallest byte share this project has ever promoted anything on — **0.055%** —
and it won, because the hypothesis was never about the bytes. 72 of 508 launches is 14.2%
of the dispatch, and inductor cannot close it: a scheduler cannot fuse a producer and a
consumer across an opaque `extern_kernels` call, so the `cat` and the `copy_` are stranded
either side of it by construction.

---

## 3. Three slots that did not test what they were built to test

**All three failed in a way the record can name**, which is the part worth more than the
ratios.

### 3a. `021` — the CUDA graphs were never recorded

`cudagraph_nodes: 0`, `cudagraph_skips: 127`. The candidate was refused for mutated inputs
exactly as the reference is, so **the hypothesis was not measured**: 0.9986 says nothing
about what CUDA graphs are worth on this model.

That distinction exists only because this batch added the counter. Without it the slot
reads as a clean refutation of the largest hypothesis in the backlog — which is blocker 16
in a different costume, and the reason `AGENT.md` §8 now carries both shapes.

What *is* established:

* **The baseline's behaviour is now measured rather than read out of a dump.** Slot 0
  recorded `cudagraph_nodes: 0, cudagraph_skips: 128` — 128 decode steps, each its own
  function id, each refused. Every rental this project has run has been timing a decode step
  with no CUDA graphs anywhere in it.
* **The install is not the problem.** Layer 1 passed: every one of the decode cache's
  tensors carried `_dynamo_static_input_type` after `new_cache`, and the reference's own
  cache did not. `mark_static_address` did what it says.

So the break is between the mark and `func.static_input_idxs`, and this rental cannot say
where. **The next rental can, for free:** `TORCH_LOGS=cudagraph_static_inputs` prints
`Adding static input pos %s for source %s` at trace time and
`check mutation static input indices: %s` at run time, which brackets the gap exactly.
Ranked suspects: the cache tensors reach the graph through a plain Python object
(`DecodeCache.layers[i].conv`) rather than an nn.Module attribute, and
`_extract_tensor_dict` only stamps `tensor_dict` on placeholders dynamo wraps by a source
it tracks; and the warm fx-graph cache may be returning a `CompiledFxGraph` whose
`static_input_idxs` were computed on a run where nothing was marked.

### 3b. `023` — a kernel that took a scale and threw it away

Layer 1 caught it at a **relative error of 4511** on the first probe. `_reduce_partials_kernel`
declared `SCALE` and `HAS_SCALE` in its signature and **read neither**, so
`tiled_gemv_int8` and `tiled_gemv_fp8` returned unscaled integer dot products — int8 values
run to ±127 against weights near 0.05, which is a factor of ~2500 and exactly the 9473 nats
layer 2 reported.

The epilogue was not missing by design. `quantised_linear.py:355` has always done it:

```python
scale = tl.load(SCALE + offs_n, mask=mask_n, other=0.0)
tl.store(OUT + pid_m * stride_om + offs_n, (acc * scale).to(OUT.dtype.element_ty), mask=mask_n)
```

Batch 004's rewrite moved that epilogue into the split-K reduction kernel and dropped the
multiply. **A pure regression, in code that had never been executed** — see §5.

### 3c. `024` — one character, and only for one dtype

```
CompilationError: cannot cast int32[constexpr[64], constexpr[64]] to <['64', '64'], fp8e4nv>
```

`other=0` in the masked weight load is an int32 literal. Triton casts it to int8 without
complaint and refuses it for e4m3 — so the int8 slot compiled and ran while the fp8 slot,
sharing the same kernel body, did not compile at all. `other=0.0` casts to both.

Both defects are fixed, and both now have a CPU test that catches the *class*:
`kernels/kernel_contract_test.py` walks every `@triton.jit` body's AST and fails a
parameter that is declared and never read, or a dtype-polymorphic masked load whose `other`
only one dtype accepts.

---

## 4. What the preconditions bought, and what they cost

`026` and `027` declined on `021 ≥ 1.02`, correctly: with no CUDA graph recorded, both
would have re-measured numbers the batch already had. Six slots ran instead of eight.

**And the same machinery is why §3b reached a rental at all.** Batch 004 built five slots on
`_tiled_gemv_scaled_kernel` and declined every one, saving 16 billed minutes — and shipping
a kernel that had never executed. It passed CI, it passed review, and it was wrong, because
**the only thing that can test a Triton body is running it.**

That is not an argument against preconditions; batch 004's saved a rental from
re-measuring a settled fact five times. It is a new fact about them, and it has a cheap
remedy: a declined slot's code is *unexecuted*, not *verified*, and the next batch that
picks one up should treat it as new. See `docs/BATCHES.md`.

---

## 5. Costs, measured

| | Rental 40 |
|---|---|
| Whole rental, provisioning to destroy | **32.32 minutes, $0.2350** |
| Fixed cost before slot 0 | **~13 minutes** — checkpoint fetch 1m24s, GPU suite, dump |
| Slot 0 — identity, includes the reference's first compiled call | **116 s** (bench 107 s) |
| Slots 1-5 | **147-292 s** |
| Of which the approximate correctness gate | 5.8-9.2 s |
| A candidate that changes the root class | **+72 s to +218 s** in its first warmup round |
| Peak memory | 8.07 GiB bf16, **16.73 GiB** with the int4 head resident, of 31.36 |

**The fixed cost halved, and the repo sync is why.** Rental 38 paid ~30 minutes; this one
paid ~13. The checkpoint came down in 1m24s against 10m49s, and the repo sync no longer
ships the 1.4 GB compile cache alongside 9.5 MB of repository — it goes up separately, to
the path the run actually reads. See `docs/GPU-ACCESS.md` blocker 17.

**A new cost this batch found: a candidate that replaces the root model class pays a full
`max-autotune` recompile.** The identity slot's candidate reuses the reference's compiled
code and its first round took 858 ms; `021`'s took **71.7 s** and `025`'s **218 s**, because
a new class is a new dynamo code object. Both rounds are discarded warmup so no score is
affected, but it is 1-4 minutes a slot and it is not in the cost model.

---

## 6. What is settled, and what the next rental should do

**Settled:**

* **A hand-written Triton kernel beats `torch.compile(mode="max-autotune")` on this model**,
  by 1.0791 at an IQR of 0.00034, correct at 0.9318 top-1 and 0.01674 nats. Also by 1.0144,
  with a second kernel, on a completely different mechanism.
* **The GEMV was grid-starved, not structurally slow.** 656-847 GB/s on the head against
  228-319 averaged over the layer projections, with the flop padding and split-K unchanged.
* **At this site the kernel is bandwidth-bound with a 29% nibble-unpack tax**: int8 beats
  int4 on byte rate and loses on time.
* **The compiled baseline runs with no CUDA graphs**, measured now rather than inferred:
  0 nodes, 128 skips, on every slot.
* **Launch count is worth something.** Removing 48 of 508 launches, moving 0.05% of the
  bytes, bought 1.39%.

**Not settled:** what CUDA graphs are worth here, because `021` never recorded one.

**The next rental.** The backlog's entry 1 is reachable again and its arithmetic is
untouched: the head is 14.80% of the bytes and the layer projections are 77.9%, and the
reason they lost was parallelism, which is a property of the *tile* and not of the model.
So: `_launch_shape` sized for the narrow sites — split-K far harder than the current cap of
8, and a tile that does not need `BLOCK_N ≥ 16` — re-run on the same sites batches 003 and
004 lost on. Then `023` and `024` again, now that the scale is applied and fp8 compiles;
`026` and `027`, which have still never been tested; and `021` with
`TORCH_LOGS=cudagraph_static_inputs`, which costs nothing and settles §3a either way.
