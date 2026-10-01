# Batch 011 — the compiler's dequant fusion is site-dependent, and the card was the driver

> **Two findings, neither of them the one this batch was built to get.**
>
> **1. Inductor does not fuse a grouped int4 dequantisation into an MLP matmul.** It
> materialises the dequantised weight as a **full 2560x9216 fp32 tensor** — 94.4 MB a site,
> 18.12 GB/token of extra traffic across 96 sites — and runs the matmul as a separate
> kernel. The `output_code` dump says so in six lines. So `056`'s win at the tied head was
> never the general capability this repository read it as, and `071-int4-mlp-torch-dequant`
> would have returned **~0.37**, not the 1.25-1.55 registered for it. The slot never ran;
> the dump answered it anyway, for nothing, and refuted the prediction outright.
>
> **2. The card variable is the host driver, and it has been in every record since rental
> 40.** This card ran the reference at **441 GB/s, 0.34x** the best recorded — the worst in
> the project's history — with **no throttle flags** and the same GPU, compute capability,
> torch and Triton as the healthy ones. Seven identity slots separate on one field: every
> RTX 5090 driver seen twice sits in **1197-1283 GB/s**; the only two outliers are the only
> two drivers seen once. Blocker 18 has been open since rental 43 for want of this.
>
> **And a third, smaller, which is the one the batch registered:** `068-champion-pair` —
> the pair this repository ships and had never benchmarked — returned **1.2179**, the largest
> raw margin ever measured here, and **16% above the product of its two ingredients**
> measured alone in the same process.

Rental 56, RTX 5090, 2026-10-01. Six slots measured, four declined. 98.65 billed minutes,
**$0.769**; rental 55 before it cost $0.378 and measured nothing (blocker 21).

---

## The scorecard

Identity carried **+2.36%**, so the corrected column is what the slots are worth. Every
ratio's IQR is 0.010-0.032 against rental 54's 0.0001 — this card is noisy as well as slow.

| Slot | Outcome | Ratio | IQR | corrected | ms/tok cand | ms/tok ref | ref GB/s | cand GB/s | Predicted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `000-identity` | `inconclusive` | **1.0236** | 0.0260 | — | 19.131 | 19.482 | 441 | 449 | `identity` ⚠️ |
| `066-inline-causal-conv` | **`win`** | **1.0778** | 0.0322 | ~1.053 | 18.740 | 20.132 | 427 | 458 | `win` ✅ |
| `067-int4-head-torch-dequant` | `inconclusive` | **0.9971** | 0.0230 | ~0.974 | 20.349 | 20.312 | 423 | 375 | `win` ❌ |
| `068-champion-pair` | **`win`** | **1.2179** | 0.0281 | ~1.190 | 16.295 | 19.896 | 432 | 469 | `win` ✅ (range ❌, low) |
| `069-verify-inflation-k1` | **`loss`** | **0.4956** | 0.0197 | γ(1)=2.018 | 42.113 | 20.351 | 422 | 204 | `loss` ✅ (0.786 ❌) |
| `070-verify-inflation-k2` | **`loss`** | **0.4708** | 0.0097 | γ(2)=2.124 | 43.227 | 20.241 | 424 | 199 | `loss` ✅ |
| `071-int4-mlp-torch-dequant` | `precondition_failed` | — | — | — | — | — | — | — | `win`, **refuted by the dump** |
| `072-int4-mlp-and-head` | `precondition_failed` | — | — | — | — | — | — | — | — |
| `073-conv-mlp-and-head` | `precondition_failed` | — | — | — | — | — | — | — | — |
| `074-int4-wide-and-head` | `precondition_failed` | — | — | — | — | — | — | — | — |

All six measured slots report `dynamo compiled 2 graph(s)`, so none is a silent eager
fallback. Correctness passed everywhere: the conv **264/264 at 0.00000 nats** for the fifth
rental running, the head **0.9318 / 0.01674 nats** for the sixth, and `070`'s `sequence`
gate attributed its first divergence at position 10 to a reference top-2 gap of **0.0** —
a flip where the reference had no opinion at all.

**The identity slot did not calibrate.** 1.0236 at an IQR of 0.0260 is the largest offset
this project has recorded, against rental 54's 1.0002 ± 0.0001 eight days earlier. It is
reported as `inconclusive` and it is the reason the corrected column exists.

---

## 1. Inductor materialises the MLP's dequantised weight. It does not fuse it.

The dump (`results/diagnostics/inductor-output-code.txt`, 39116 lines, taken with
`--dump-install int4_mlp_torch_dequant`) holds this, per MLP block:

```
buf48 = reinterpret_tensor(buf24, (1, 2560, 9216), (23592960, 9216, 1), torch.float32)
buf51 = empty_strided_cuda((1, 2560, 9216), (23592960, 9216, 1), torch.float32)
triton_poi_fused___rshift____to_copy_add_bitwise_and_cat_mean_mm_mul_pow_rsqrt_sub_unsqueeze_view_24
    .run(buf46, buf47, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, buf48, buf51, 23592960)
buf49 = empty_strided_cuda((1, 9216, 16), (147456, 1, 9216), torch.float32)
triton_red_fused_mm_25.run(buf48, buf49, 147456, 160)
```

A **pointwise** kernel carrying `__rshift__`, `bitwise_and` and `sub` — the grouped nibble
unpack — writes **two complete `(2560, 9216)` fp32 tensors** at a grid of 23592960, one for
`gate_proj` and one for `up_proj`. The matmul is `triton_red_fused_mm_25`, a **separate**
kernel that reads `buf48` back. There are **34 such weight-sized fp32 allocations** in the
graph, recycled across sites, and **zero `extern_kernels.mm` calls** — so this is not a
fallback to cuBLAS, it is inductor choosing to materialise and then reduce.

**Do not read the kernel names as evidence of fusion.** `triton_poi_fused___rshift____..._mm_...`
carries `mm` in its name because inductor's naming follows the *origins* of the fused node,
and this pointwise kernel feeds an `mm`. Four kernels in this dump carry both the unpack and
`mm` in their names and none of them performs the matmul. The allocation and the grid size
are the evidence; the name is not.

**The price, and it is the branch the slot registered as unreadable-if-it-happens.**

| | MB/token |
|---|---:|
| MLP bf16 weights the reference streams | 4529.85 |
| packed int4 nibbles the candidate reads | 1132.46 |
| **fp32 dequantised weight written, then read back, 96 sites** | **18120** |
| candidate total | ~23300 |
| reference total | 8587.80 |
| **predicted ratio** | **~0.37** |

So `071`'s registered prediction — `win`, 1.25-1.55 — is **refuted**, and refuted by a
diagnostic that cost nothing and did not need the slot to run.

### Why the head fused and the MLP did not

`056` at the tied head produced one reduction kernel carrying the whole unpack, the `mm`,
the final RMSNorm and the residual add, with **no weight-sized buffer in the graph**. The
same source expression at MLP width produces a materialisation. The two differ in operand
size by 27x: the head's dequantised fp32 weight would be **2.54 GB** (248320 x 2560 x 4),
the MLP's is **94.4 MB**.

The ranked reading, stated as a hypothesis and not a result: **inductor materialises when it
can afford to, and the head's win was forced rather than chosen.** A 2.54 GB intermediate is
not something its memory planning will allocate, so at that one site it had to fuse the
unpack into the reduction's prologue; at 94.4 MB it allocates and picks a `mm` template that
cannot take a fused producer.

**That reframes this repository's central claim about the compiler.** `AGENT.md` §1.1 and
`HYPOTHESES.md` entry 1 both read `056` as "inductor fuses a grouped dequantisation into a
GEMV prologue", and generalised from it to 91.85% of per-token bytes. The generalisation is
wrong. What `056` showed is that inductor fuses it **at one site, where the alternative was
unallocatable** — which is a much narrower fact and does not reach the MLP at all.

### The cheap next experiment, and it is one line

`torch_dequant_gemv_int4` asks for an **fp32** matmul:

```python
return (flat.float() @ dense.float()).to(x.dtype)
```

That is what makes the materialised operand fp32 — 94.4 MB rather than 47.2 — and it is
plausibly also what selects the `mm` template over a reduction. Expressing it as a bf16
matmul with fp32 accumulation (which is what the Triton kernel does, and what the
correctness reference means) halves the materialisation and may remove it. **Two slots:
the expression as it stands and the bf16 variant, one variable apart, in one process** —
the shape that produced every finding this project holds. Neither needs a new kernel.

---

## 2. γ is a step, confirmed on a second card and a second GPU generation

| | tokens in the verify | γ | ratio |
|---|---:|---:|---:|
| ordinary decode | 1 | **1.000** by definition | — |
| `069` k=1 | 2 | **2.018** | 0.4956 |
| `070` k=2 | 3 | **2.124** | 0.4708 |

Fitting the two points: **10.6% per token**, intercept at one token **1.912** — and since a
one-token verify *is* ordinary decode at γ = 1.000, that leaves a **step of 91%** at the
seq=1 → seq>1 boundary.

| | step at the boundary | slope per token |
|---|---:|---:|
| rental 54, RTX 4090, 3 and 5 tokens | 23% | 4.4% |
| **rental 56, RTX 5090, 2 and 3 tokens** | **91%** | **10.6%** |

**Batch 010's conclusion survives independent confirmation and its alternative is refuted
on both cards.** A smooth per-token cost has to pass through (1 token, γ=1.000); through
`069` and `070` that model predicts 1.158 at two tokens against a measured **2.018**. The
registered prediction here was **0.786** — the step model's own number — and it is wrong by
59%. What was right is the *shape*: a step, not a slope, now measured with a two-token
verify that batch 010 never ran.

**And the step's magnitude moved 4x with the card, which is itself evidence about what it
is.** A traffic cost scales with bandwidth; a launch-and-dispatch cost scales with clock and
with fixed per-kernel overhead. This card is 0.34x on bandwidth and the step went up, not
down. The candidate column runs at **204 GB/s against the reference's 422** while moving
essentially the same weight bytes — the whole deficit is graph structure.

**Suspect 3 is dead.** Batch 010 §4 ranked `No valid triton configs ... Required: 110592
Hardware limit: 101376` on the verify's `k+1` shape as the 4090's narrower shared memory,
and said "any decision that turns on γ's exact value needs a 5090 first." This is a 5090
and it logs **the same 101376-byte limit** and the same empty autotune pool. The remaining
ranked suspect is `reference.py`'s `if seq_len > 1` mask branch, which is a fixed cost of
leaving seq=1 and therefore the right shape for a step.

---

## 3. The pair this repository ships, measured at last — and it is superadditive

`apply_champions` installs `inline_causal_conv` beside `int4_head_torch_dequant`, and no
rental had ever benchmarked that combination: rental 46's `058` composed the conv with the
Triton head that has since been retired.

| | ratio | corrected |
|---|---:|---:|
| `066` conv alone | 1.0778 | ~1.053 |
| `067` head alone | 0.9971 | ~0.974 |
| product of the two | 1.0747 | ~1.026 |
| **`068` both together** | **1.2179** | **~1.190** |

**The pair beats the product of its ingredients by 16%**, at a candidate bandwidth of
469 GB/s — the highest in the batch, on the slowest card in the project's history. The
registered reading was "068 against 066 and 067 separately", so the comparison was named
before the rental; what was not predicted is the direction. It is the opposite of `060`,
where two dispatch savings competed and the static cache was worth +1.5% alone and +0.0%
on top of the conv.

**Held back deliberately: this is a cross-slot comparison, and cross-slot is what this card
damages.** Within a slot the interleaved rounds divide the card out; across slots nothing
does, and the reference column moved 423 → 432 GB/s between `067` and `068`. The margin is
6x the identity band, so it is unlikely to be noise — but "two fusion barriers removed
together open a region neither opens alone" is a mechanism claim, and it needs a healthy
card and a dump before this file calls it one.

---

## 4. The card was the driver, and the record has said so since rental 40

| driver | rentals | reference |
|---|---|---:|
| 580.159.03 | 40, 42 | 1283, 1223 GB/s |
| 580.173.02 | 45, 46 | 1214, 1197 GB/s |
| 580.159.04 | 43 | **845 GB/s** |
| **610.43.02** | **56** | **441 GB/s** |

Same GPU model, same compute capability 12.0, same torch 2.11.0+cu128, same Triton 3.6.0,
**throttle_reasons 0x0** and a 400 W limit at 63 W draw. **Every driver seen twice lands
inside 7%. The two outliers are the two seen once.**

**SM clock is refuted as the explanation**: rental 43 had the highest clock of all seven
(2925 MHz) and the second-worst bandwidth, and this card's 2377 MHz is not 0.34x of 2827.

Blocker 18 — *"the card is an uncontrolled variable the size of the effect, and there is
nothing in the environment record to show it"* — has been open since rental 43. The field
was in the record all along; `card_baseline.py` even said "the same driver" in its own
docstring, which was false. Three changes:

* **`DF_MAX_CUDA` (13.0).** `cuda_max_good` is the *advertised* form of the driver branch,
  so this is the one place the decision can be made **before** the rental is paid for. The
  project had a floor and no ceiling, and the damage is at the top end: this host advertised
  **13.3**, every healthy one advertised 13.0. One data point at the ceiling, so it is a
  default with its evidence written beside it, not a law.
* **`ReferenceObservation.driver`**, populated for all seven observations, and the slot-0
  pre-flight now names what this host's driver has run before — including the case that
  matters once an observation is recorded: a driver whose *recorded* history is slow still
  warns, because recording it is what stops it reading as unknown.
* **The 4090 has a row** (rental 54, 848.6 GB/s), so it is no longer the example of an
  unrecorded model.

`DF_MIN_CUDA` also moved 12.8 → **12.9** before this rental, three rentals after
`GPU-ACCESS.md` concluded it should: a host advertising *exactly* 12.8 has failed with
`Error 804` every time one was taken.

---

## 5. The gate held its form and still declined for the card's reasons

Every precondition in this batch was a **margin between two slots** rather than an absolute
ratio — the fix for rental 46, where `061` declined because its floor asked whether the head
reached 1.02 *absolute*. The declines printed their arithmetic:

```
skipping 071: 067 - 000-identity measured 0.9971 - 1.0236 = -0.0265 against a floor of 0.0
```

That is the correct reading of the proposition the gate names, and it was **still the card
talking**. On a card achieving 0.34x, removing bytes buys nothing, so a bandwidth
hypothesis cannot clear a bandwidth gate — and the margin, though drift-proof, is not
regime-proof.

**The reusable form: a margin between two slots survives a card that drifts, and does not
survive a card that is outside the regime the hypothesis is about.** The instrument for that
is not a better precondition. It is refusing the host, which `DF_MAX_CUDA` now does, and
which costs $0.005 at the offer line instead of $0.77 at the writeup.

**And the decline was lucky.** `071` would have measured ~0.37. Three slots were spent on
nothing, and four were saved from measuring a catastrophe for the wrong reason.

---

## 6. What this batch changed

* **Inductor's grouped-dequant fusion is site-dependent and does not reach the MLP.** It
  materialises a 94.4 MB fp32 weight per site and runs the matmul separately; `071` would
  have returned ~0.37 against a registered 1.25-1.55. Entry 1's ladder is suspended, not
  closed, pending the bf16-matmul variant.
* **`056`'s win was narrower than this repository read it.** Not "the compiler fuses grouped
  dequantisation" but "it fused at the one site where materialising was unaffordable".
* **γ is a step, confirmed on a 5090 after a 4090**: 91% and 10.6%/token here against 23%
  and 4.4% there, with a two-token verify batch 010 never ran. The no-step model is refuted
  on both cards; the registered 0.786 was wrong by 59%.
* **Batch 010's suspect 3 is dead**: the empty `k+1` autotune pool is not the 4090's shared
  memory — a 5090 reports the same 101376-byte limit.
* **The shipped pair is measured: 1.2179**, 16% above the product of its parts, flagged as
  cross-slot and unconfirmed.
* **The card variable has a name**: the host driver, separating seven observations where
  clock, GPU model and toolchain do not. `DF_MAX_CUDA` refuses it at the offer line.
* **Blocker 21 is fixed and proven**: `fetch-weights` now watches bytes rather than stdout.
  The heartbeat showed **2.05 GB already on disk while the progress bar sat at `4/6`**,
  which is exactly what the old guard read as silence — and with Xet out of the path the
  9.32 GB landed in **~2 minutes** against rental 55's 40-minute stall.
