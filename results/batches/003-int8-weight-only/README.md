# Batch 003 — weight-only quantisation. Seven admissible ratios, and every one of them a loss.

> **Rental 37, 2026-09-16.** RTX 5090, instance 51253059, machine 137732, $0.4089/hr,
> **55.55 billed minutes, $0.3786**, destroyed cleanly. Preceded by rental 36, a deliberate
> abort at 2.08 minutes and $0.0142 (see "The abort" below). Session total **$0.3928**.
>
> `calibrated: true`. `graphs_compiled: 3` on **all seven slots**. This is the first batch
> in this project where every ratio is a real compiled-against-compiled comparison, and the
> first where no slot was voided. **Predictions scored 1 of 7 correct**, and the one that
> was right is the identity slot.

---

## The scorecard

| Slot | Ratio | IQR | Layer 1 | Layer 2 | Outcome | Predicted |
|---|---:|---:|---|---|---|---|
| `000-identity` | **1.0024** | 0.0190 | — | 5/5 exact | calibrated | `identity` ✅ |
| `009-gemv-bf16-control` | 0.2801 | 0.0106 | pass | 1/5 exact | `incorrect` | `inconclusive` ❌ |
| `010-int8-dequant-torch` | 0.9893 | 0.0212 | **fail** | 256/264 | `incorrect` | `loss` ❌ |
| `011-int8-mlp` | 0.4669 | 0.0254 | pass | 256/264 | `incorrect` | `win` ❌ |
| `012-int8-all-linear` | 0.1962 | 0.0042 | pass | 255/264 | `incorrect` | `win` ❌ |
| `013-int8-full` | 0.1878 | 0.0026 | pass | 256/264 | `incorrect` | `win` ❌ |
| `014-int4-full` | 0.1797 | 0.0040 | pass | 226/264 | `loss` | `win` ❌ |

**No kernel in this batch beat the compiler, and no kernel in this batch was wrong.** Those
are two separate statements and the rest of this file is about keeping them separate.

---

## 1. Why six slots came back `incorrect` — three causes, none of them a kernel

Every Triton kernel passed layer 1 at every probe, including the widened probe set added
after the abort: worst relative error across all of `int8_gemv`, `int4_gemv` and `bf16_gemv`
was **7.8e-3**, about one bf16 ULP. **There were no kernel bugs.** The six `incorrect`
verdicts came from three defects in the *gates*, and they are worth separating because they
generalise differently.

### 1a. `009` — an exact gate on an implementation that cannot be bit-exact

`009-gemv-bf16-control` computes the same mathematical function as the reference: bf16 in,
fp32 accumulate, bf16 out. I gated it `exact` on that basis. It matched **1 of 5 prompts**,
diverging at tokens 3, 9, 2 and 1 — while its layer-1 error was 7.5e-3, one bf16 ULP.

The category error: *computing the same function* and *producing the same bits* are not the
same property. My GEMV sums K in a different order from cuBLAS, so the last bf16 rounding
lands one ULP away, and this model's top-2 logit gaps are narrow enough that an argmax
flips. `AGENT.md` §7a already records this, and rental 35 already lost slots 001 and 002 to
exactly it. **I re-derived the trap rather than reading it.**

An exact gate is correct only for a kernel that is *bit-identical by construction* — which,
in this repository, means the identity champion and nothing else.

### 1b. `010`–`013` — bars set by intuition, below the resolution of the statistic

The four approximate slots all cleared their KL bar by a wide margin and all missed the
top-1 agreement bar. Converted from a fraction to what was actually counted:

| Slot | Matched | Flips | Bar allows | Mean KL | KL bar |
|---|---|---:|---|---:|---:|
| `010` | 256/264 | 8 | 5 | 0.001138 | 0.01 |
| `011` | 256/264 | 8 | 5 | 0.000820 | 0.01 |
| `012` | 255/264 | 9 | 5 | 0.001096 | 0.01 |
| `013` | 256/264 | 8 | **8** | 0.001216 | 0.02 |
| `014` | 226/264 | 38 | 40 | 0.091854 | 0.15 |

Two things are wrong here.

**The bars were guesses wearing a prediction's clothes.** I wrote 0.98 from general
knowledge that per-channel int8 is mild. It is mild — KL of 0.0011 nats is essentially
nothing — but on this checkpoint it still flips 3% of argmaxes, because the logit gaps are
narrow. Registering a number in advance makes it a prediction; it does not make it
*informed*. The right move was to derive the bar from something measurable, and the
measurable thing was available for free: `test_report_the_first_greedy_divergence` already
computes top-2 logit gaps for the oracle, and the same statistic predicts the flip rate.

**And the bars were finer than the statistic can resolve.** With n = 264 positions, top-1
agreement is quantised to 1/264 = 0.0038. `013` missed its bar by **0.000303 — eight
hundredths of a single token.** A threshold an order of magnitude below one sample is not a
gate, it is a coin toss with extra steps. The binomial 95% CI on 8/264 is roughly
[1.3%, 5.9%], so "≤ 2% flips" and "3% flips" are not distinguishable at this n at all.

The fix is structural, not a looser number: **gate on KL, which is continuous and has no
resolution floor, and report agreement alongside it with its interval.** If agreement is to
be gated, n has to grow by an order of magnitude first.

### 1c. `010` — the only layer-1 failure, and the control failed against its own reference

`010-int8-dequant-torch` is the sole layer-1 failure in the batch, and it is not a Triton
kernel. `Int8DequantLinear` computes `q.to(bf16) * scale.to(bf16)` and then `F.linear` —
it rounds the dequantised weight **to bf16** before the matmul, as any practical PyTorch
implementation would. The layer-1 reference dequantises in **fp32**. So the control was
being compared against arithmetic it never claimed to perform, and the check reported a
relative error of 0.45 for a computation that is doing exactly what it should.

I wrote one reference for two implementations with different dtype discipline, and `010`'s
own registered note asserted they would "match exactly". They match in distribution
(0.9697/0.001138 against `012`'s 0.9659/0.001096) and not at layer 1. **A shared reference
is only shared if the implementations share their rounding.**

---

## 2. Why the kernels that worked were *slower* — the kernel was never memory-bound

The ratios say the kernels lost. The interesting question is what they lost to, and the
answer is visible without a profiler once the bytes and the clock are put in one table.

| | MB/token | ms/token | Achieved | % of 1790 GB/s |
|---|---:|---:|---:|---:|
| `compiled` baseline (slot 0) | 9158 | 6.84 | **1308 GB/s** | **73%** |
| `compiled` baseline (slots 2-6) | 9158 | 7.63 | 1172 GB/s | 66% |
| `009` bf16 hand GEMV | 9158 | 26.90 | 332 GB/s | 19% |
| `012` int8 hand GEMV | 5588 | 38.68 | 141 GB/s | 8% |
| `013` int8 hand GEMV + head | 4952 | 40.63 | 119 GB/s | 7% |
| `014` int4 hand GEMV + head | 2850 | 42.49 | 65 GB/s | 4% |

**Read the last two columns downward.** As the kernel moves *fewer* bytes, it gets *slower*
— 26.90 → 38.68 → 42.49 ms/token while traffic falls 9158 → 5588 → 2850 MB/token — and its
achieved bandwidth collapses from 19% of peak to 4%. A kernel whose time rises as its
traffic falls is not bandwidth-bound. It is instruction-bound, and every byte removed was
replaced by arithmetic on the critical path.

That single observation explains the whole batch:

* **The 1.85× ceiling was never available to this kernel.** A bandwidth saving can only be
  collected by an implementation that is spending its time on bandwidth. Mine was spending
  it on issue.
* **int8 cost 1.438× the time of bf16** in the same kernel on the same sites — the
  `.to(tl.float32)` conversion of 4096 int8 values per K-iteration, added to a loop that had
  no spare issue slots.
* **int4 cost a further 1.046×** — the shifts and masks of the nibble unpack, same story.
* **The dose-response ladder inverted.** 49.5% of bytes → 0.4669, 77.9% → 0.1962. More
  sites was monotonically worse, which is the signature of "this kernel is slow and there is
  more of it", and the exact opposite of "fewer bytes is faster".

### The mechanism, in the source

```python
for k0 in range(0, K, BLOCK_K):
    ...
    acc += tl.sum(w * x[None, :], axis=1)   # <- a cross-lane reduction, every iteration
```

`tl.sum(..., axis=1)` is a **cross-lane reduction across BLOCK_K**, and it runs once per
K-iteration: **20 times** for K=2560, **72 times** for `down_proj`'s K=9216. Each one is
shuffles and shared-memory traffic that produce no bytes and hide no latency. The standard
form accumulates a full tile and reduces **once**:

```python
acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
for k0 in range(0, K, BLOCK_K):
    acc += w * x[None, :]        # pure FMA, no cross-lane traffic
out = tl.sum(acc, axis=1)        # one reduction for the whole K
```

Three further omissions compound it: no `tl.dot` (so the MMA pipeline is unused and the
loads are not scheduled against it), no split-K (so each program walks all of K serially
with nothing to overlap), and a hardcoded `num_warps=4` at every shape.

**None of this was visible on a CPU, and the control is the only reason it is visible now.**
Without `009`, the honest reading of `012` at 0.1962 would have been "weight-only
quantisation does not help on this model" — a conclusion that is false, expensive, and would
have closed the backlog's best hypothesis for the wrong reason.

---

## 3. The `010` surprise, and a claim this repository has been asserting for two weeks

`docs/HYPOTHESES.md` has said, as the load-bearing justification for hypothesis 1:

> given `dequant(W_int4, scales) @ x`, inductor materialises the full bf16 weight tensor
> into global memory and then calls into cuBLAS. That *adds* an 8.4 GB write on top of the
> read, making the quantised version **slower** than bf16.

The entry's own "Watch for" said to verify it before building on it. Nobody had. `010`
measured it: **0.9893, IQR 0.0212** — a margin from 1.00 of 0.0107, inside the noise band.
Weight-only int8 written in plain PyTorch and compiled costs **the same** as bf16. Not
slower. Not faster.

**And the materialisation story is refuted outright by a bandwidth budget**, with no
profiler and no `output_code` dump:

| | MB/token | required GB/s at the measured 7.68 ms/token |
|---|---:|---:|
| If inductor really materialised to DRAM | 19868 | **2525 — 141% of the card's peak** |
| DRAM traffic the clock actually permits | ≤ 14083 | ≤ 1790 |

A 5090 cannot move 2525 GB/s. Therefore inductor **does not** write a full bf16 weight to
DRAM and read it back — it either fuses the dequantisation into the matmul's prologue, or
the transient stays resident in the 96 MB L2 (a single 47 MB weight fits). The claim in the
backlog is wrong and is corrected in this pass.

What is *not* settled is why a 39% reduction in DRAM traffic bought 0% of time. If the
dequant is fused, `010` is moving 5588 MB/token at 710 GB/s — 40% of peak, against the
baseline's 73% on the same hardware — so the conversion is costing roughly half the
achievable bandwidth in stalls. That is a coherent story and it is **not evidence**.

**The thing that would settle it is free and I skipped it.** `docs/HYPOTHESES.md` says, in
bold, that dumping `TORCH_LOGS=output_code` before writing a kernel "is not optional". I
went straight from arithmetic to a rental. The rental produced a number whose mechanism I
still cannot name, on the one slot that was designed to test a mechanism.

---

## 4. What this batch did establish

Not everything here is a negative result.

* **The harness is calibrated and the batch is fully admissible.** `000-identity` = 1.0024,
  IQR 0.0190, 5/5 exact tokens.
* **Blocker 16 is proven on a GPU.** `recompile_limit_for(7)` raised dynamo's limit to 22
  and **every slot reports `graphs_compiled: 3`**. Rental 35's six void ratios cannot recur
  silently; the counter that would catch it has now been exercised on a real card.
* **The compiled reference is genuinely near the roofline: 1308 GB/s, 73% of a 5090's
  vendor peak, 6.84 ms/token against a 5.11 ms/token roofline.** This number did not exist
  before and it reframes the whole backlog — see §5.
* **Weight-only int8 accuracy on this checkpoint, measured:** per-channel int8 costs
  0.0011 nats of mean KL and flips 8 of 264 teacher-forced argmaxes. Group-128 int4 costs
  0.0919 nats and flips 38 of 264. Both are now facts rather than literature priors.
* **Quantising the tied LM head is not the risky site.** I argued it would be, because its
  error reaches the argmax unattenuated, and set its bars looser for that reason. Adding it
  moved KL by 0.00012 nats and agreement from 255/264 to 256/264. **That reasoning was
  wrong**, and the looser bars for `013` were unjustified.
* **The approximate gate works.** It produced stable, interpretable numbers, the kernel and
  torch paths agreed on them to within a token, and it distinguished int4 from int8 cleanly.
  Its *thresholds* were the problem, not its design.
* **The abort paid for itself in evidence, not in luck.** See below.

### The abort

Rental 36 was killed two minutes in, before the checkpoint download, at $0.0142, on noticing
that `check_distribution` fed ~150 rows through in one pass — above `GEMV_MAX_ROWS` — so
**layer 2 would have scored the dense fallback and never executed the kernel**, while layer
1's two hardcoded probes missed the 8-row launch branch (`k_proj`, `v_proj`, `in_proj_a`,
`in_proj_b`) that three slots install on.

Both were fixed before relaunching. In the event **no kernel was wrong**, so the widened
gates caught nothing — and that is the point worth recording: they converted "no evidence of
a bug" into "evidence of no bug", on eight sites per layer that would otherwise have been
untested. The 0.1962 ratio is now attributable to the kernel's speed and not to its
correctness, which is the only reason §2 can be written at all.

This is the fourth instance of the shape `AGENT.md` §8 keeps finding: **a gate whose subject
was never reached.** Blocker 7 fired before its subject existed; blocker 9 after its subject
had succeeded; blocker 15 had a justification nothing had exercised; this one would have
passed without the thing it gates ever running.

---

## 5. What to do next, and the bar it has to clear first

The ceiling arithmetic is unchanged and still correct: weights are 91.85% of per-token
bytes, and int8 halves them. What the batch adds is the **denominator** — the baseline is
already at 73% of peak bandwidth — and the **precondition**: only a kernel that is itself
bandwidth-bound can collect a bandwidth saving.

Concrete targets, from this rental's own numbers:

| To achieve | int8 GEMV must reach | mine reached |
|---|---:|---:|
| Merely **tie** the compiled baseline | 715 GB/s (40% of peak) | 141 GB/s (8%) |
| The predicted **1.64×** on layer projections | ~1170 GB/s (65%) | 141 GB/s (8%) |

**So the gating criterion for the next batch is not a quantisation result at all:**

> **`009-gemv-bf16-control` must reach ≥ 0.90 before any quantised slot is worth running.**

A bf16 GEMV moves exactly cuBLAS's bytes; if it cannot tie cuBLAS, nothing built on it can
win, and five slots of this rental were determined the moment `009` returned 0.2801. That is
the most expensive structural lesson here and it is fixable in the framework rather than by
discipline — see `docs/BATCHES.md` on conditional slots.

Ranked ideas, in `docs/HYPOTHESES.md` as entries 5-7:

1. **Free and first: `TORCH_LOGS=output_code` on both scoring columns, plus one profile.**
   It answers §3, it tells us where the baseline's remaining 27% goes, and it costs a step
   in an existing rental rather than a rental.
2. **Rewrite the GEMV: accumulate a tile and reduce once; `tl.dot` with M padded to 16 so
   the MMA pipeline is used; split-K for the narrow projections.** Flops are free here —
   arithmetic intensity at batch-1 decode is ~2 flop/byte against a machine balance near
   150 — so a 16× flop waste that buys a scheduled memory pipeline is a trade worth making.
3. **Then, and only then, fp8 rather than int8.** The 5090 is Blackwell: e4m3 converts in
   the MMA path rather than through ALU instructions, which removes precisely the cost that
   made int8 slower than bf16 here. Same 2× byte saving, without the conversion tax.

Everything else in the backlog still sits under 6.23% of bytes and cannot pay for a rental
on its own.
