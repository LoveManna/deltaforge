# 012-the-matmul-dtype — rental 57, 2026-10-02

**The matmul dtype is a no-op at every site width this model has, and the reason is that
the source dtype does not control the dtype of the buffer inductor materialises.**

RTX 5090, host driver 580.159.03, torch 2.11.0+cu128, triton 3.6.0. Instance 53932253,
**56.85 billed minutes, $0.388.** Eight slots registered, six run, two declined.

---

## What the rental settled

### 1. The dtype does nothing, at either width

| pair | fp32 arm | bf16 arm | margin | both arms' IQR |
|---|---:|---:|---:|---|
| tied head, 248320 wide | **1.0251** | **1.0218** | **−0.0033** | 0.0104 / 0.0081 |
| 96 MLP projections, 9216 wide | **0.6390** | **0.6452** | **+0.0062** | 0.0060 / 0.0259 |

Both margins sit inside their own interquartile spreads, and they point in **opposite
directions**. The candidate columns agree to the digit on achieved bandwidth — 1068 against
1071 GB/s at the head, **458 against 458** at the MLP — and layer 2 agrees to five decimal
places at the head (mean KL 0.0167417 against 0.0167412, top-1 identical at
0.9318181872367859) and four at the MLP (0.0491396 against 0.0491607).

**`torch_dequant_gemv_int4`'s `flat.float() @ dense.float()` was never the lever.**

### 2. Why: inductor materialises *before* the cast

`--dump-install int4_mlp_torch_dequant_bf16` put `TORCH_LOGS=output_code` on the **bf16**
candidate. Its graph contains:

* The reference's graphs (`[0/0]`, `[0/1]`) hold **zero** dequantised-weight buffers and
  **zero** unpack launches, which is what attributes everything below to the candidate.
* The candidate's two graphs (`[0/2]`, `[0/3]`, the dynamic-shape variants) each hold **16
  `empty_strided_cuda((1, 2560, 9216), …, torch.float32)` buffers at 94.37 MB** plus **one**
  of the `down_proj` shape — **17 of the 96 MLP sites per decode step** — and **zero**
  buffers of either shape in `torch.bfloat16`.
* `triton_poi_fused___rshift____to_copy_bitwise_and_cat_mm_mul_silu_sub_unsqueeze_view_15`,
  **32 launches per graph**: the grouped nibble unpack as a *pointwise* kernel.
* `triton_red_fused_mm_16` and `triton_red_fused_mm_25`, **32 launches each per graph**: the
  matmuls, as separate reductions carrying nothing but `mm`.
* **Zero `extern_kernels.mm`** in the whole file — not a cuBLAS fallback, a chosen
  materialisation, exactly as rental 56 recorded for the fp32 arm.

So a candidate that asks for a **bf16** matmul still materialises an **fp32** weight. The
dequantise arithmetic is fp32 because the group scales are fp32, inductor picks its
materialisation point *upstream of* `.to(x.dtype)`, and the cast it was asked for is applied
to a buffer that has already been written at full fp32 width. **Changing the matmul's dtype
cannot move a decision that is made before the matmul.**

### 3. The 52.75% prize, finally executed — and it is a loss

`077-int4-mlp-torch-dequant` had been registered **twice** as a predicted win (`061`, rental
46; `071`, rental 56) and **declined both times unexecuted**. It ran here ungated and
returned **0.6390 at an IQR of 0.0060**, the tightest band in the batch. The registered
*win at 1.25–1.55* is now refuted by measurement rather than by a dump.

### 4. The 9216-wide sites materialise and the 2560-wide ones mostly do not

Counted per decode step from the candidate's graph, not inferred from a ratio:

| dequantised operand | the sites it belongs to | buffers per graph | of how many sites |
|---|---|---:|---:|
| `(1, 2560, 9216)` fp32 | `gate_proj` / `up_proj`, K=2560 N=9216 | **16** | 64 |
| `(1, 9216, 2560)` fp32 | `down_proj`, K=9216 N=2560 | **1** | 32 |

**17 of 96.** Rental 56's ranked reading — *inductor materialises when it can afford to* —
predicted that the cheaper, narrower operands would materialise **more**. The census says the
opposite: the 9216-wide output sites carry nearly all of the buffers and the 2560-wide
`down_proj` carries one. `081` taking the same construction to 200 sites and 97.85% of the
bytes returned **0.6150**, no worse than the MLP's 0.639 despite 104 extra sites, which is
consistent with the added narrower sites mostly fusing.

### 5. The materialisation does not account for the size of the loss, and I do not know what does

This is the batch's loose end and it is mine, not the manifest's.

| | |
|---|---:|
| reference traffic | 8.59 GB/token |
| candidate's own byte model | 5.19 GB/token |
| candidate's **actual** traffic, from achieved GB/s | **13.44 GB/token** |
| gap to explain | 8.25 GB |
| fp32 weight write + read-back, 17 operands | **3.21 GB** |
| **unattributed** | **~5.04 GB** |

My first reading of these ratios — written into this file before the dump was counted —
back-solved a "multiplier on the quantised weight" of 2.07x at the MLP and 1.64x at the wide
sites, and treated the difference as evidence about where fusion happens. **That arithmetic
assumed every extra byte was the materialised weight, and the dump refutes it**: only 17
sites hold a weight buffer and they account for 3.21 of the 8.25 GB. The remaining ~5 GB is
real, measured, and unexplained — candidate pointwise passes over 9216-wide intermediates,
the `cat`, the silu and re-reads are the obvious suspects and none is counted.

**None of this touches the batch's finding.** The dtype being a no-op rests on two arms
measuring 0.6390 and 0.6452 with identical achieved bandwidth and identical allocations — a
direct comparison that needs no byte model at all. What the byte model cannot yet do is say
why the *level* is 0.64 rather than 1.02, which is what 17 materialised operands alone would
predict.

---

## Prediction scorecard

Registered in `src/deltaforge/batches.py` and committed as `bbc9f36` **before** the rental.

| # | slot | predicted | outcome | ratio | IQR | verdict right? | magnitude right? |
|---|---|---|---|---:|---:|:--:|:--:|
| 0 | `000-identity` | identity | inconclusive | 1.0024 | 0.0089 | yes | yes |
| 1 | `075-int4-head-torch-dequant` | win, 1.00–1.04 | **win** | 1.0251 | 0.0104 | yes | yes |
| 2 | `076-int4-head-torch-dequant-bf16` | inconclusive, 0.99–1.04 | win | 1.0218 | 0.0081 | **no** | yes |
| 3 | `077-int4-mlp-torch-dequant` | loss, 0.33–0.45 | **loss** | 0.6390 | 0.0060 | yes | **no** |
| 4 | `078-int4-mlp-torch-dequant-bf16` | loss, ~0.60 | **loss** | 0.6452 | 0.0259 | yes | close |
| 5 | `079-int4-mlp-and-head-bf16` | win | `precondition_failed` | — | — | — | — |
| 6 | `080-conv-mlp-and-head-bf16` | win | `precondition_failed` | — | — | — | — |
| 7 | `081-int4-wide-torch-dequant-bf16` | loss, 0.40–0.55 | **loss** | 0.6150 | 0.0021 | yes | **no** |

**5 of 6 verdicts, 3 of 6 magnitudes, the central mechanism refuted, and the loss level still
unexplained (§5).** The
mechanism claim was that bf16 would halve the materialisation. It halved nothing: both arms
move the same traffic and allocate the same fp32 buffers. A registered prediction that gets
the verdict right for the wrong reason is worth less than this table makes it look, and the
three-zone structure (0.37 / 0.60 / ≥1.09) is what makes that legible — **the measurement
landed between two zones at both widths**, which is what said the model was wrong.

`076`'s verdict miss is the harness classifying a within-noise margin as a win on its own
IQR. The ratio was inside the registered range; the label was not. The quantity the slot
existed for — the margin against `075` — is −0.0033.

---

## Two defects this batch found in its own machinery

### A zero-floor margin gate passes on noise

`081` was gated on `ratio[078] - ratio[077] >= 0.0`, deliberately a *margin* rather than an
absolute floor, because rental 46 declined the largest hypothesis in the backlog on an
absolute floor. The margin came in at **+0.0062 against an IQR of 0.0259** — pure noise — and
the gate passed. It happened to pass toward running the slot, and the slot was worth running,
but the gate tested nothing: it would have declined 97.85% of the model's bytes on a sign
flip of the same noise.

**`Precondition(floor=0.0, versus=...)` is under-specified in the same way an absolute floor
is.** A margin gate's floor belongs **above the IQR of the slots it reads**, not at zero.
Nothing in `batch.py` enforces that and nothing warns about it.

### The compile cache can no longer be pushed, and that is now permanent

`DF_CACHE_MAX_PUSH_MB` refused a **3411 MB** push (ceiling 512) and the rental compiled cold,
which is rental 45's fix working — it failed toward minutes rather than tens of minutes. But
the cache only grows, so **every future rental on this card now compiles cold**, and the
pruning that `deltaforge-fixed-cost-dominates` recorded as "not done" has gone from deferred
to binding. The teardown pulled the cache home, so the directory grew again.

---

## What this closes

**The quantisation-in-torch line is closed at the MLP and wider.** Both spellings of the
dequantise-GEMV lose at 52.75% of per-token bytes, and both lose for a reason that is not
addressable from the source expression: inductor chooses to materialise, in fp32, upstream of
any cast the source applies. `079`, `080` and the 67.55% rung are not worth re-registering
in this form.

**What survives is the head**, where `075` returned **1.0251** — the best this champion has
ever measured, against 1.0171 on rental 46 and 0.9971 on rental 56 — and where the fusion is
real and the dump shows no weight-sized buffer.

## What is still open

* **Where the unattributed ~5 GB/token goes** (§5). This is the first question to answer and
  it needs no rental: the dump is home and the candidate's non-weight buffers can be counted
  the same way the weight buffers were.
* **Why 16 of 64 `gate`/`up` sites materialise and 31 of 32 `down_proj` sites do not.** Half
  of one shape and none of the other is not an affordability threshold, and the boundary is
  now counted rather than guessed. No slot needed; the dump is already home.
* **Whether the materialisation point can be moved from the source at all.** Doing the
  dequantise so that the fp32 intermediate never exists — scales in bf16, or the unpack
  expressed so the cast precedes the arithmetic — is a different hypothesis from this batch's
  and is untested. It is the only remaining route to the 91.85%.
* **`inline_causal_conv` was not re-measured this rental.** The shipped pair's 1.2179 from
  rental 56 still carries its cross-slot caveat and a +2.36% identity offset.

## Environment notes

* Identity **1.0024 at an IQR of 0.0089**, reference at **1220 GB/s** — "in family", 0.95x
  the best recorded. `DF_MAX_CUDA` rejecting the CUDA 13.3 branch is why: rental 56's 441
  GB/s card was on 13.3, and the offer search took a 13.0 host instead.
* **Power telemetry is in the record for the first time**: 426.14 W of 500 at slot 0, throttle
  reasons `0x0`, so `GPU-ACCESS.md` blocker 18's persistent-power-cap suspect is ruled out
  for this host. SM clock did drift, 2887 MHz at slot 0 to 2430 by slot 3; interleaved rounds
  absorb it and the IQRs stayed tight.
* **A scoring round cost ~1.8 s of wall time here, not ~18 s.** `docs/BATCHES.md` budgets 15
  rounds at ~180 s a slot; the measured slots ran 105–312 s *including* build, correctness and
  compile, with `bench` itself at 136 s for `077`. The 18 s figure needs re-deriving against
  this rental before the next batch is sized by it.
