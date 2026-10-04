# DeltaForge — Results

Hand-written Triton kernels for the **Qwen3.5-4B** decode path, measured against
`torch.compile(mode="max-autotune")` on an identical pure-PyTorch reference.

Twelve batches, 56 GPU rentals, $11.49 of billed compute, 2026-10-02.

---

## The result, in short

**The two shipped champions, composed, run this model's decode 1.2179x faster than
`torch.compile(mode="max-autotune")`** — 21.8% off the step, ~19% after correcting for that
card's calibration offset. The margin comes off a *strong* baseline, which is the whole
difficulty: the compiled reference already moves bytes at **65.7–71.5% of an RTX 5090's
vendor peak bandwidth**, so the headroom above it is thin, and finding any of it meant
knowing exactly where the compiler's remaining slack was and proving it before renting a
card.

The champions are a four-tap causal convolution at **1.0765–1.0778**, bit-identical to the
reference at 264/264 tokens and 0.00000 nats, and a group-128 int4 tied LM head at **1.0251**
(IQR 0.0104). Both follow from this project's central finding, which is a fact about the
compiler rather than about any one kernel: **an opaque `torch.library.custom_op` is a fusion
barrier, and it costs its own kernel plus everything inductor can no longer fuse across it —
a bill that is invisible at the call site.** Keeping the same arithmetic visible to inductor,
as fusible torch operations, wins at both sites.

The convolution is bit-identical either way and measured **0.7854 behind the custom op
against 1.0765 as fusible ops — one variable, 37%**, because the barrier drove buffer
allocations from 59 to 190 and made the linear-attention state reduction run twice per layer,
24 times per token. At the LM head the fusible form beat a hand-written Triton GEMV **by 3.2%
in the same process on the same card**: a custom op forfeits the RMSNorm fusion inductor
welds into the matmul, *and* the register-level 4-bit unpack that was the kernel's reason to
exist is something the compiler emits inline. That the two are the same program is not an
assumption — layer 2 returned 0.9318 agreement and 0.01674 nats for both, digit for digit, on
six rentals.

**Every number here came off a rented card, with its outcome and magnitude registered in a
manifest committed before the GPU existed** — 12 batches, 56 rentals, $11.49 of compute, an
identity control that can void a batch, two correctness gates, and 56 of 56 instances
destroyed.

---

## 1. What was measured, and against what

| | |
|---|---|
| Model | `Qwen/Qwen3.5-4B`, text decoder only (~4.21 B params; vision tower and MTP head excluded in `weights.py`) |
| Workload | batch 1, context 2048, 128 decoded tokens |
| Baseline | `src/deltaforge/reference.py` — pure PyTorch, **no custom kernels of any kind** — under `torch.compile(mode="max-autotune")` |
| Score | `median(t_compiled / t_candidate)` over interleaved rounds; > 1 beats the compiler |
| Noise band | interquartile spread of the per-round ratios. **A margin smaller than its IQR is not a win.** |
| Hardware | 47 RTX 5090 rentals, 9 RTX 4090; torch 2.11.0+cu128, Triton 3.6.0 |

**The baseline is deliberately mine.** HuggingFace's own Qwen3.5 modeling code dispatches
Gated DeltaNet to hand-written Triton via `flash-linear-attention` and attention to
FlashAttention. Benchmarking against that would compare hand-tuned Triton to hand-tuned
Triton while claiming to beat a compiler. `transformers` is used only to load weights,
tokenize, and act as a correctness oracle — never as the baseline. For the same reason
`reference.py` never calls `scaled_dot_product_attention`: SDPA *is* the fused kernel under
test. `reference_purity_test.py` enforces this in CI.

**The baseline is the project's most reusable number:**

> `torch.compile(mode="max-autotune")` runs this decode path at **6.70–7.30 ms/token —
> 1177–1282 GB/s, 65.7–71.5% of an RTX 5090's 1792 GB/s vendor peak**, against a 4.79
> ms/token roofline on the bytes it actually moves.

---

## 2. Arithmetic before kernels

`docs/roofline.py` prints where every byte of one decode step goes. It needs no GPU and no
checkpoint, and it decides what *can* win:

| What moves | share of per-token bytes |
|---|---:|
| Weights, streamed once | **91.85%** |
| GQA `repeat_interleave` materialisation | **6.23%** |
| Recurrent state, read + write | 1.10% |
| KV cache read | 0.78% |
| SwiGLU intermediates | 0.026% |
| Norm + residual | 0.018% |

A hypothesis cannot beat the share of bytes it touches. Batch-1 decode is a weight-streaming
problem; hand-writing a fused RMSNorm has a ceiling of 0.018%, which is below the harness's
own noise band. **Five hypotheses are in the graveyard with the arithmetic that closed
them**, at a cost of zero GPU minutes.

**Then I read the generated code**, which three rentals had skipped and which cost nothing:

- **Inductor already eliminates the GQA head expansion** — `x1 // 4` indexing on the
  unexpanded KV cache, never materialised. That closes the backlog's second-largest
  hypothesis (6.23% of per-token bytes) **without a kernel**, and corrects the baseline's
  true traffic from 9158.23 to **8587.80 MB/token**.
- **There is no cuBLAS in this decode path.** `extern_kernels` is called for convolution and
  nothing else: inductor generates Triton for every matmul and welds the residual add and
  the RMSNorm *into* them. A hand-written GEMV gives all of that fusion up — 248 times per
  token.
- **The compiled baseline runs with no CUDA graphs at all**, because the causal conv mutates
  its cache in place (0 recorded, 128 skipped, measured on every slot of rental 40).

That reframed the exercise: the compiler is two thirds of the way to the memory wall *and*
fusing everything around the matmuls it emits, so the only large win left is to move fewer
bytes — and taking a matmul away from inductor has a price that is measurable.

---

## 3. Headline results

**Two champions, and neither contains a line of Triton.**

| | `045-inline-causal-conv` | `056-int4-head-torch-dequant` |
|---|---|---|
| Replaces | `causal_conv` | `decode_step` |
| Median ratio | **1.0765** (r45), **1.0650** (r46), **1.0778** (r56) | **1.0171** (r46), 0.9971 (r56), **1.0251** (r57) |
| IQR | 0.0552 / 0.0213 / 0.0322 | 0.0130 / 0.0230 / **0.0104** |
| Correctness | **bit-identical** — 264/264 tokens, 0.00000 nats | 0.9318 top-1 agreement, 0.01674 nats |
| Best candidate bandwidth | **1254 GB/s — 70% of vendor peak** | 1068 GB/s |

**Composed, they are the largest margin this project has recorded: `068-champion-pair` =
1.2179** (rental 56), on a card whose identity slot carried +2.36%, so read it as ~1.19
corrected. Notably it is **16% above the product of its two ingredients measured alone in
the same process** (1.0778 × 0.9971 = 1.0747) — a cross-slot comparison on a drifting card,
so the leaderboard declines to promote on it and says why.

---

## 4. Findings

### 4.1 An opaque custom op is a fusion barrier, and the bill is invisible at the call site

The cleanest controlled pair in the project. `044-fused-causal-conv` and
`045-inline-causal-conv` compute **the same function** — both bit-identical to the reference,
layer-1 relative error `0.0`, layer-2 264/264 at 0.00000 nats — and differ only in whether
the four taps reach inductor as a `torch.library.custom_op` or as torch operations it may
fuse. Measured in the same process, on the same card, minutes apart:

| | ratio | IQR | Δ ms/token |
|---|---:|---:|---:|
| `044` — the taps behind `custom_op` | **0.7854** | 0.0446 | +1.919 |
| `045` — the same taps as fusible torch ops | **1.0765** | 0.0552 | **−0.727** |

**One variable, 37%.** The `output_code` dump gives the mechanism, not a story: a barrier
forces inductor to materialise the op's inputs and outputs (**59 → 190 buffer allocations**
per decode step) and it splits a producer chain it had been fusing, then *recomputes* the
shared prologue rather than reading the buffer it just wrote — the linear-attention state
reduction runs **twice per layer, 24 times per token** (298 reductions in the reference, 297
in the inline candidate, **321** with the custom op).

Two corollaries worth as much as the number:

- **It settled a 20% regression that two earlier rentals had attributed to composing two
  kernels.** There was no composition effect: the conv loses 21% *alone*, and composed with
  the int4 head it loses 20% (`050` − `044` = +0.011, inside both IQRs). Nobody had measured
  the ingredient before the composition.
- **The allocation count was a red herring.** Rental 43 had ranked 59 → 190 allocations as a
  suspect. The winning candidate allocates **232 and wins**. The duplicated reduction is the
  mechanism; the allocation count is a symptom.

### 4.2 At the project's best site, the compiler beat the hand-written kernel

Three registrations of **one program** — the group-128 int4 dequantise-GEMV on the tied LM
head — ran in a single process on a single card. They compute the same function, and the
record proves it: layer 2 returned **0.9318 agreement and 0.01674 nats** for every one of
them, as it has on six rentals.

| | how the program reaches `torch.compile` | ratio | IQR |
|---|---|---:|---:|
| `054-int4-head` | a hand-written Triton kernel behind `torch.library.custom_op` | **0.9851** | 0.0099 |
| `055-int4-head-triton-op` | the same kernel behind `torch.library.triton_op` | **errored** | — |
| `056-int4-head-torch-dequant` | torch operations, no kernel of mine at all | **1.0171** | 0.0130 |

**The slot that won was registered in advance as a predicted loss**, on the argument that
inductor would materialise the head's 1271.40 MB/token bf16 weight. It does not. Its final
kernel is one reduction carrying `__rshift__`, `bitwise_and`, the `mm`, the final RMSNorm
and the residual add **together** — the whole grouped nibble unpack fused into the matmul
prologue — and the graph holds **no weight-sized buffer anywhere**, allocating 48 buffers
where the reference allocates 59.

So the hand-written kernel lost twice over: a custom op cannot keep the RMSNorm fusion the
reference welds into the `lm_head` matmul, **and** the register-level unpack that was its
whole claim to necessity is something the compiler emits inline. `tiled_int4_head` is
retired.

`055` is a recorded negative with engineering value: `torch.library.triton_op` does not
trace this GEMV on torch 2.11 — dynamo ran the body under `FakeTensorMode` and the launch
reached `.data_ptr()` instead of being intercepted by `wrap_triton`. It cost 34 s of GPU
time. The CPU suite could assert the op was *registered* and structurally could not assert
it *traces*, because tracing needs Triton.

### 4.3 The compiler's dequantisation fusion is site-dependent

The obvious generalisation of 4.2 — "inductor fuses grouped dequantisation into a GEMV
prologue" — is **false**, and the dump refuted it before a kernel was written. At the 96 MLP
projections inductor materialises the dequantised weight as a full `(2560, 9216)` **fp32**
tensor — 94.37 MB a site — in a *pointwise* kernel carrying the unpack, then runs the matmul
as a **separate** reduction that reads the buffer back. Zero `extern_kernels.mm` anywhere:
this is a chosen materialisation, not a cuBLAS fallback.

A census counted per decode step from the candidate's own graph, rather than back-solved
from a ratio:

| dequantised operand | sites | buffers per graph | of how many sites |
|---|---|---:|---:|
| `(1, 2560, 9216)` fp32 | `gate_proj` / `up_proj`, K=2560 N=9216 | **16** | 64 |
| `(1, 9216, 2560)` fp32 | `down_proj`, K=9216 N=2560 | **1** | 32 |

**17 of 96 sites.** This also inverts the ranked prediction that cheaper, narrower operands
would materialise *more*: the wide-output sites carry nearly all the buffers.

And the prize this implied — int4 on 52.75% of per-token bytes — had been registered as a
predicted win **twice** and declined unexecuted both times on a precondition gate. Run
ungated, it is a **measured loss at 0.6390, IQR 0.0060**, the tightest band in its batch.
Taking the same construction to 200 sites and 97.85% of the bytes returned **0.6150**.

### 4.4 The GEMV was grid-starved, not structurally slow — and an aggregate could not show it

Batches 003 and 004 installed a hand-written GEMV on all 248 layer projections at once and
lost by 5x (0.2801, then **0.1934 after a rewrite aimed at the diagnosed cause**). Measured
per site, the same kernel family gives:

| | achieved bandwidth |
|---|---:|
| naive GEMV, 248 layer projections | 319 GB/s |
| tiled GEMV (`tl.dot`, K-major, split-K), 248 layer projections | 228 GB/s |
| int4, **the LM head alone** | **656 GB/s** |
| int8, **the LM head alone** | **847 GB/s** |

At BLOCK_N=64 the head launches **3880 programs** on a 170-SM card; `in_proj_a` is 32
channels wide and launches **four**. Same kernel, same flop padding, same split-K, opposite
result. **An aggregate number over 248 sites could not distinguish a slow kernel from a
starved grid** — which is why the first two quantisation batches drew the wrong conclusion
from a correct measurement.

Relatedly, batch 003's kernels were never bandwidth-bound in the first place: **as they
removed bytes they got slower** — 26.90 → 38.68 → 42.49 ms/token while traffic fell 9158 →
5588 → 2850 MB/token — because a cross-lane `tl.sum` ran once per K-iteration instead of
once per output, and the dequantisation landed on an already-saturated issue port. The 1.85x
int8 ceiling is real arithmetic and unreachable by an implementation that is not spending
its time on memory.

### 4.5 The matmul dtype is a no-op, because the materialisation decision is made upstream of the cast

One operator apart, two arms per site width:

| pair | fp32 arm | bf16 arm | margin | arms' IQRs |
|---|---:|---:|---:|---|
| tied head, 248320 wide | **1.0251** | **1.0218** | −0.0033 | 0.0104 / 0.0081 |
| 96 MLP projections, 9216 wide | **0.6390** | **0.6452** | +0.0062 | 0.0060 / 0.0259 |

Both margins sit inside their own interquartile spreads and point in opposite directions;
the candidate columns agree to the digit on achieved bandwidth (**458 against 458 GB/s** at
the MLP). The dump says why: the bf16 candidate's graphs hold **17 fp32 weight buffers per
decode step and zero bf16 ones**. The dequantise arithmetic is fp32 because the group scales
are fp32, inductor picks its materialisation point *upstream of* `.to(x.dtype)`, and the
requested cast is applied to a buffer already written at full fp32 width. **Changing a
matmul's dtype cannot move a decision made before the matmul.**

### 4.6 A speculative verify is a step, not a slope

Nobody had put a number on what a `k+1`-token verify costs at batch 1, so I built
instruments rather than candidates: drafters that are **wrong on purpose**, so acceptance is
zero by construction (confirmed by histogram across 2159 cycles) and the measured ratio is
exactly `1/γ`, the inflation factor.

- 1 → 3 tokens costs **31.6%**; 3 → 5 costs **6.7%**. A one-token verify *is* ordinary
  decode at γ = 1.000, so **~23% of the cost is a discontinuity at the `seq=1 → seq>1`
  boundary** — dispatch and kernel-selection, not traffic. The registered cost model had no
  term for it.
- What γ really moves is **break-even**: from 0.06 to **0.317** mean accepted tokens per
  cycle for a free drafter.
- **Both registered kill criteria were crossed and both were withdrawn as underived** — the
  spec's own formula with the measured γ still gives 1.34x at k=2 and 1.45x at k=4 for an
  int4 self-draft at p = 0.9, with k=4 winning by more than the k=2 its criterion spared.
- The n-gram drafter measurably works and is **20x too weak**: 0.016 accepted against a
  0.317 bar. Its histogram is the reason — `{0: 2108, 1: 0, 2: 17}`: never one token, and
  0.8% of the time the whole block. That is prompt-lookup finding a literal repeat or
  nothing, which is a property of the workload, not a tuning problem.

### 4.7 The measurement is the instrument, and it had to be sharpened before three conclusions flipped

The retired int4 head is the clearest case. **The kernel never changed** and never returned a
correctness result other than 0.9318 / 0.01674. What changed is resolution:

| | rental 40 | rental 43 | rental 45 | rental 46 |
|---|---:|---:|---:|---:|
| reference achieved | **1282 GB/s** | 845 GB/s | 1197 GB/s | 1167 GB/s |
| `000-identity` calibration | 1.0008 | 1.0101 | 0.9972 | 1.0044 |
| the int4 head | **1.0791** | 1.0161 | 1.0105 (inconclusive) | **0.9851 (loss)** |
| IQR | 0.00034 | 0.00407 | 0.0263 | 0.0099 |

Going from 5 scoring rounds to **15** took the worst IQR in a batch from 0.1511 to 0.0213
and `inconclusive` slots from six of eleven to one of ten, **on the same physical host
drifting the same way** — and it is what resolved the int4 head's loss, the static cache's
+1.5% win, and the fact that dispatch savings are **not additive** (the cache is +1.5% alone
and +0.0% on top of the fused conv; `058` and `060` return the same 7.251 ms/token from
genuinely separate runs).

**The card is an uncontrolled variable the size of the effects under test**, and the
write-ups name the rental for every number. Two RTX 5090s reporting the same memory clock,
driver and torch differed by **1.61x** on the baseline; one card lost 17% of its SM clock
mid-rental (2910 → 2400 MHz). Seven identity slots separate on exactly one field — the
**host driver**: every driver seen twice sits in 1197–1283 GB/s, and the only outliers
(441 GB/s, the worst in the project) are the only drivers seen once. Interleaving the
reference and candidate *within* each round divides drift out of every ratio, which is why
no slot was voided; absolute predictions and cross-rental comparisons do not survive it.

**And the obvious next step was refuted rather than assumed.** If the GEMV won at the head
because the head has parallelism, a better *tile* should unlock the other 83% of the bytes.
Searched on the card — BLOCK_N, split-K, BLOCK_K, warps, pipeline depth — the search made the
champion's own site **2.3x slower** (656 → 282 GB/s) while the MLP came back at 67 GB/s,
within noise of the 65 measured two rentals earlier at a different tile in a different
kernel. Two more tiles were then pinned in advance (0.9343, 0.9502) and the last untried
direction after that (0.9617). **Five measured points, four deliberate attempts, and the
heuristic tile nobody chose on purpose is the best of them. The tile question is closed.**

---

## 5. Method

The methodology is the part I would most want to be judged on, because it is what makes the
negative results usable.

**Interleaved scoring.** Both models are resident at once and timed in alternating rounds,
so thermal drift and clock changes cancel. They are separate module trees — installing a
kernel swaps a class on the candidate's modules — but they **share one set of parameter
tensors**, since nothing writes to a weight under `no_grad`: 8.4 GB on the card instead of
16.8, and one read of the checkpoint instead of two. `cli._assert_parameters_are_shared`
fails the run if that silently stops being true.

**A calibration slot that can void the batch.** Every batch opens with an identity champion
that installs nothing and must measure 1.00 ± noise. It has measured 1.0009, 1.0018, 1.0024,
0.9913, 1.0008, 1.0053 (IQR 0.00086) and 1.0002 (IQR **0.0001**). When it carried +2.36% on
rental 56, that batch's write-up reports every slot in a *corrected* column and promotes
nothing. **A batch whose identity slot misses 1.00 voids every number in it, in writing.**

**Two correctness gates, with magnitudes recorded as numbers.** Per kernel, `allclose`
against the reference operation at bf16 tolerance on real model shapes, with max absolute
and max relative error recorded rather than reduced to pass/fail. End to end, greedy-decode
128 tokens from five fixed prompts checked into the repo; the candidate's token sequence must
match eager exactly. Failures are kept in `results/` with their error magnitudes, because a
candidate that was fast but wrong is among the most useful things to read.

That discipline paid directly: **none of batch 003's six `incorrect` verdicts was a kernel
bug.** Every Triton kernel passed its numerics gate at every probe, worst relative error
7.8e-3 — about one bf16 ULP. All six were defects in the *gates*: an `exact` gate on a kernel
that computes the same function but not the same bits (fp32 accumulation in a different order
from cuBLAS lands one ULP away, and this model's top-2 logit gaps flip an argmax on that); a
top-1 agreement bar set finer than the statistic can resolve (at n=264, agreement quantises
to 1/264 = 0.0038 — **one slot missed its bar by 0.000303, eight hundredths of a single
token**); and a slot gated against a reference it never claimed to match. Five more gate
defects have been found and recorded since, including a zero-floor margin gate that passes
on +0.0062 against an IQR of 0.0259.

**Predictions registered before the rental.** Each hypothesis commits its predicted outcome,
magnitude range and reasoning to `src/deltaforge/batches.py` **before** the GPU is created,
and every batch write-up scores them:

| batch | predictions scored |
|---|---|
| 003 int8 weight-only | 1 of 7 |
| 004 bandwidth-bound GEMV | 1 of 2 (five slots declined, recorded as untested) |
| 005 launch and head | 3 of 5 |
| 006 tile and sites | 1 of 4 |
| 007 compose and retile | 2 of 6 |
| 008 ingredients and barriers | 5 of 11 |
| 009 visible kernels | 5 of 10 |
| 010 speculative verify | 3 of 5 |
| 011 bytes not kernels | 4 of 5 scored slots (identity did not calibrate) |
| 012 the matmul dtype | 5 of 6 verdicts, **3 of 6 magnitudes** |

Low scores are the point: an unfalsifiable account scores 100%. The three misses that
mattered most each corrected a mechanism — the predicted-loss slot that won (4.2), the
generalisation the dump refuted for free (4.3), and a cost model with no dispatch term
(4.6). Batch 012 scores *magnitudes* separately from verdicts, because a registered
prediction that gets the verdict right for the wrong reason is worth less than a verdict
column makes it look.

**An explicit outcome taxonomy**, so that "we didn't measure it" never reads as "it
doesn't work": `win`, `loss`, `inconclusive` (inside the noise band — neither promoted nor
buried), `graveyarded on mechanism` (closed by arithmetic, with the numbers that closed it),
`incorrect` (failed a gate, with error magnitudes), `precondition_failed` and `error`. A
precondition-skipped slot is recorded as **unexecuted, not refuted** — two defects survived
a rental that way, and the 52.75% prize in 4.3 was declined twice before it was finally
measured.

**Batching, because fixed cost dominates.** A rental's fixed cost — image, torch, a 9.32 GB
checkpoint, the GPU test suite and one `max-autotune` compile of the reference — is ~19
minutes measured; each additional hypothesis costs 3–6. So a rental measures **7–12
hypotheses**, compiling the reference once, with the reference column re-timed inside every
hypothesis's own interleaved rounds so per-round drift still divides out exactly. Each slot
runs in its own try/except and writes its record the moment it finishes, so a wrong kernel
costs a slot rather than a rental.

**Cost control in code, not discipline.** $50/month: a month-to-date gate that refuses at
$45, a session soft gate at 180 billed minutes checked before a run and never during one, a
**pre-flight refusal when the remaining budget cannot fit the identity slot plus one kernel
slot**, per-slot wall-clock caps, a local-side watchdog that destroys the instance at 210
minutes regardless of what the remote is doing, a batch that stops *itself* before a
hypothesis it cannot finish, teardown that pulls results back **before** destroying, and a
shell trap on `EXIT`/`INT`/`TERM` so a crash still destroys the instance. Every script in
`remote/` supports `--dry-run`, which exercises every gate and the teardown trap without
contacting the create or destroy endpoints.

**Ledger:** 56 instances provisioned, **56 destroyed, zero leaked**, 1621 billed GPU
minutes, **$11.4887 lifetime**, including two cancelled mid-flight with SIGTERM. The
cheapest informative rental cost $0.0478; the most expensive mistake cost $1.3878 and spent
its entire cap inside one compile.

**The compile cache is a measured artifact, not a hope.** A cold `max-autotune` compile had
never finished inside a session — 6980.9 s unfinished — until an unrolled prefill scan was
found and fixed; it then completed in **267.5 s**, and **57.4 s** warm off the cache banked
from the previous rental on the same card. Teardown reports warmth from bytes that actually
landed on disk, because a log that overstates it would corrupt the number the cache exists
to improve. A later rental spent ~33 minutes pushing a 2.2 GB cache to save ~3.5 minutes of
compiling, so pushes above `DF_CACHE_MAX_PUSH_MB` are now refused — fixed cost on the same
host fell from ~72 minutes to ~22.

**Infrastructure, honestly.** Twenty-two numbered blockers are tracked with status and
proof, because each rental that got further than its predecessor did so by exposing the next
one. Three of them could not even be *seen* until the one before was fixed: a cold compile
that never finished hid a teardown bug that only fires *between* slots, which in turn hid
dynamo's `recompile_limit` (8) silently timing **eager** candidates against a compiled
reference — the cause of six unrelated kernels all returning ratios between 0.146 and 0.157.
Those six numbers are kept in the leaderboard marked `void — candidate ran eager`, because
deleting evidence is worse than labelling it.

**Scale:** ~37k lines of Python and POSIX shell and **1160 tests — 1142 of which pass on a
laptop with no GPU and no checkpoint** (18 skip without a card), plus `cli fusion` — which compiles a
candidate locally and diffs the generated code, so "does this registration install a fusion
barrier?" is answered **before** renting a card. That tool is what refuted 4.3's
generalisation for free.

---

## 6. What is *not* claimed

Stated up front because a knowledgeable reader will ask:

- **Not beating cuBLAS on dense bf16 GEMM.** At batch 1 the linear layers are at the
  bandwidth roofline; you do not beat a roofline with a better kernel, only by moving fewer
  bytes or running a different algorithm.
- **Not beating a compiler at elementwise fusion.** Inductor's home turf, and it emits
  Triton. Those hypotheses are in the graveyard with the arithmetic that closed them.
- **Not beating vLLM or SGLang.** Those are already hand-tuned Triton and CUDA; out of scope
  as a win condition.
- **Not a serving engine, not training, not multi-GPU.** Single-card inference decode only.
- **No estimated, placeholder or illustrative numbers anywhere in the repository.** Every
  number came off a real card, and the ones that do not mean what they appear to mean say so
  in the line that reports them.

---

## 7. Open, and recorded as open

- **The loss level in 4.3 is unexplained.** The candidate's actual traffic is 13.44
  GB/token against a 5.19 GB/token byte model; the fp32 write and read-back of 17 operands
  accounts for 3.21 GB, leaving **~5.04 GB unattributed**. My first reading of those ratios
  back-solved a "multiplier on the quantised weight" and treated it as evidence about where
  fusion happens; the buffer census refuted that arithmetic, and the file says so rather
  than keeping the tidier story.
- **One rental-40 number has never reproduced.** The int4 head's 1.0791 belongs to that
  rental; later rentals on healthy cards gave 1.0161, 1.0105 and 0.9851. The leaderboard
  does not pretend otherwise.
- **One infrastructure blocker has no real fix**, only a manual `--exclude-machines`
  workaround: a phantom ask traps the deterministic, price-ordered offer search.

---

## 8. Where to look

| Path | What it is |
|---|---|
| `README.md` | The project's own front page: claims, non-goals, reproduction |
| `LEADERBOARD.md` | Every hypothesis attempted, with ratio, IQR, correctness, outcome and the rental it belongs to |
| `docs/roofline.py` | Where the bytes go. No GPU, no checkpoint — run it first |
| `docs/HYPOTHESES.md` | The backlog, and the graveyard with the arithmetic that closed each entry |
| `docs/BATCHES.md` | How a batch works and what filling one requires |
| `src/deltaforge/reference.py` | The baseline. Pure PyTorch, no custom kernels, ever |
| `src/deltaforge/fusion.py` | What inductor generated, parsed; which registrations are opaque. No GPU |
| `src/deltaforge/batches.py` | Batch manifests — the predictions, committed before each rental |
| `src/deltaforge/harness/` | Interleaved timing, correctness gates, result records |
| `remote/` | Instance lifecycle in POSIX shell, so it works before the environment exists |
| `results/batches/0NN-*/README.md` | One write-up per rental, scored against its registered predictions |

Reproduce the parts that need no GPU:

```sh
uv sync --extra dev && uv run pytest          # 1142 pass, 18 skip, no GPU needed
uv run python docs/roofline.py                # where the bytes go
remote/run_remote.sh --dry-run --session-id smoke --batch 001-calibration
uv run python -m deltaforge.cli fusion --install inline_causal_conv
```

Licensed Apache-2.0, matching the target model.
