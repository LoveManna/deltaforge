# Batch 009 — the compiler beat our kernel at the site we won on

> **Rental 46, 2026-09-23.** RTX 5090, instance 52273140, machine 140734 — **the same
> physical host as rental 45**, which makes this the first batch this project can compare
> to its predecessor with the card held fixed. $0.4896/hr, **79.00 billed minutes,
> $0.6446**, destroyed cleanly. Lifetime **$9.1429 across 45 instances, 45 destroyed,
> zero leaked.**
>
> `calibrated: true` — `000-identity` measured **1.0044 at an IQR of 0.0115**. Eight slots
> ran, one errored, two declined on registered floors.
>
> **Predictions scored 5 of 10, and the batch's headline is one of the misses.**
>
> **`056-int4-head-torch-dequant` — the int4 head with no kernel of ours in it at all —
> won at 1.0171, while `054-int4-head`, the hand-written Triton kernel that has been
> champion of `decode_step` since rental 40, lost at 0.9851.** Same weights, same
> function, same process, layer 2 identical to the digit. It was registered as a predicted
> **loss** on a bandwidth budget, and the budget's assumption was wrong: the dump shows
> inductor fusing the entire grouped nibble-unpack *and* the final RMSNorm into one
> reduction kernel, materialising nothing.
>
> **The champion of `decode_step` is now a program with no Triton in it.** Both champions
> this project ships were obtained by deleting a kernel.

---

## The scorecard

| Slot | Outcome | Ratio | IQR | ms/token cand | ms/token ref | Δ | cand GB/s | Layer 2 | Predicted |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `000-identity` | calibrated | **1.0044** | 0.0115 | 7.056 | 7.073 | −0.017 | 1217 | exact | `identity` ✅ |
| `053-inline-causal-conv` | **`win`** | **1.0650** | 0.0213 | **6.847** | 7.289 | **−0.442** | **1254** | 264/264, 0.0 | `win` ✅ |
| `054-int4-head` | **`loss`** | **0.9851** | 0.0099 | 7.448 | 7.356 | +0.092 | 1025 | 0.9318, 0.01674 | `win` ❌ |
| `055-int4-head-triton-op` | **`error`** | — | — | — | — | — | — | — | `win` ❌ |
| `056-int4-head-torch-dequant` | **`win`** | **1.0171** | 0.0130 | 7.386 | 7.536 | −0.150 | 1034 | 0.9318, 0.01674 | **`loss` ❌** |
| `057-static-decode-cache` | **`win`** | **1.0150** | 0.0123 | 7.395 | 7.489 | −0.094 | 1161 | 264/264, 0.0 | `win` ✅ |
| `058-conv-and-head` | **`win`** | **1.0339** | 0.0072 | 7.251 | 7.497 | −0.246 | 1053 | 0.9318, 0.01674 | `win` ✅ |
| `059-conv-and-head-triton-op` | `precondition_failed` | — | — | — | — | — | — | — | `win` (**untested**) |
| `060-conv-head-cache` | **`win`** | **1.0318** | 0.0084 | 7.251 | 7.501 | −0.250 | 1053 | 0.9318, 0.01674 | `win` ✅ |
| `061-int4-mlp-torch-dequant` | `precondition_failed` | — | — | — | — | — | — | — | `win` (**untested**) |

---

## 1. The result: our kernel lost to the compiler, at the one site we had ever won on

Three registrations of one program ran in one process on one card. They compute the same
function and the record says so: **layer 2 returned 0.9318 agreement and 0.01674 nats for
every one of them**, digit for digit, which is also what rentals 40, 43 and 45 recorded.

| | how the dequantise-GEMV reaches inductor | ratio | IQR | ms/token |
|---|---|---:|---:|---:|
| `054-int4-head` | a `torch.library.custom_op` (the champion since rental 40) | **0.9851** | 0.0099 | 7.448 |
| `055-int4-head-triton-op` | the same kernel behind `torch.library.triton_op` | **error** | — | — |
| `056-int4-head-torch-dequant` | torch operations, no kernel of ours at all | **1.0171** | 0.0130 | 7.386 |

**3.2% apart, and the sign is the finding.** Net of an identity slot at 1.0044, the
hand-written kernel costs about 1.9% and the compiler-generated one saves about 1.3%.

### The dump says why, and it is not the barrier alone

`--dump-install int4_head_torch_dequant` rendered the reference and this candidate into one
file. The reference's final kernel, and the candidate's:

```
REF   triton_red_fused__to_copy__unsafe_view_add_mean_mm_mul_pow_rsqrt_slice_t_view_43
CAND  triton_red_fused___rshift____to_copy__unsafe_view_add_bitwise_and_cat_mean_mm_mul_
      pow_rsqrt_slice_sub_unsqueeze_view_43
```

`__rshift__`, `bitwise_and`, `sub`, `unsqueeze` — **the entire grouped nibble unpack is
inside the same kernel as the `mm`, the `mean`/`pow`/`rsqrt` that is the final RMSNorm, and
the residual `add`.** One kernel, exactly where the reference has one kernel.

| per decode step | reference `[0/2]` | torch-dequant candidate `[0/5]` |
|---|---:|---:|
| `triton_poi_*` / `triton_red_*` / `triton_per_*` | 89 / 298 / 96 | **89 / 298 / 96** |
| `extern_kernels` | 24 | 24 |
| `empty_strided_cuda` allocations | 59 | **48** |
| custom-op dispatches | 0 | **0** |
| largest allocation | 248320 elements (the logits) | **248320 elements** |

**The registered prediction of a loss rested on a materialisation that does not happen.**
The rationale said: "if it instead materialises the bf16 weight — 1271.40 MB/token, against
a 96 MB L2 that cannot hold it — the candidate moves that plus the 317.85 MB of packed
nibbles it read to build it, and it cannot reach 1.0 however good the matmul is." There is
**no weight-sized buffer anywhere in the candidate graph**, and the candidate allocates
*eleven fewer* buffers than the reference. Inductor fuses a grouped dequantisation into a
248320-wide GEMV prologue, and this project had assumed for nine batches that it could not.

So the hand-written kernel loses for two compounding reasons, and only the first was
predicted:

1. **It is a fusion barrier.** The reference welds the final RMSNorm into the lm_head
   matmul; a custom op cannot, so that fusion is forfeited — the mechanism rental 45 priced
   at 37% on the causal conv.
2. **The thing it was written to do, inductor already does.** The nibble unpack in
   registers is the champion's whole claim to necessity, and the generated kernel does it
   in the same instruction stream, with the norm fused in as well.

**`docs/HYPOTHESES.md` entry 1 has been asking "why does halving the weight stream not show
up on the clock" since rental 37. This answers the other half: it does show up, when the
program is expressed so the compiler can fuse it.**

### `055` errored, and the error is a result

The `triton_op` registration — the same Triton kernel, made visible — never ran. Dynamo
refused it at trace time:

```
Dynamo failed to run FX node with fake tensors:
  call_function deltaforge.tiled_gemv_int4_visible.default(...)
  RuntimeError("Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor).
  If you're using torch.compile/export/fx, it is likely that we are erroneously tracing
  into a custom kernel.")
```

On torch 2.11 the op's fake implementation is obtained by running the body under
`FakeTensorMode`, and the Triton launch reached `.data_ptr()` rather than being intercepted
by `wrap_triton`. **The slot cost 34 seconds and it cost `059` as well**, whose precondition
correctly declined a composition of a kernel that does not compile.

**The CPU suite asserted the registration existed and could not assert that it traces**,
because tracing needs Triton and the test box has none. That is the
declined-slot/unexecuted-path failure in a new costume, and the cheap fix is named in §6.

---

## 2. There is no composition effect, and dispatch savings are not additive

| slot | ratio | IQR | ms/token cand |
|---|---:|---:|---:|
| `053` conv alone | **1.0650** | 0.0213 | 6.847 |
| `054` head alone | 0.9851 | 0.0099 | 7.448 |
| `057` cache alone | **1.0150** | 0.0123 | 7.395 |
| `058` conv + head | **1.0339** | 0.0072 | 7.251 |
| `060` conv + head + cache | **1.0318** | 0.0084 | **7.251** |

**`060` minus `058` is zero to three decimal places in ms/token**, and the two slots are
genuinely separate runs — their round-0 compiles took 186 s and 54 s and their individual
round timings differ throughout. So the static decode cache, worth **+1.5% on its own**,
is worth **nothing** once the fused causal conv is installed.

**That is the question `060` was registered to ask, and the answer is the one that was
worth asking for**: the conv's saving is a dispatch saving and the cache's is a dispatch
saving, and *they are competing for the same microseconds*. This project's launch
accounting is not additive, and the next composition that assumes it is will be wrong.

It also corrects batch 008. There, `052` (conv + head + cache) measured 1.0747 against
`051`'s (conv + head) 1.0443 and the cache appeared to add 3%. `051`'s IQR was **0.1511**.
At this rental's bands — 0.0072 and 0.0084 — the difference is −0.2%. **A conclusion that
survived one rental because its band was too wide to refute it did not survive a narrower
one**, which is the entire argument for §3.

`058` at 1.0339 is also readable against its parts for the first time: 1.0650 × 0.9851 =
1.0491 predicted against 1.0339 measured, so composing costs about 1.5% beyond the product
of the two. With the head now known to lose on its own, **the shipped pair is the conv
plus a head that subtracts from it.**

---

## 3. Fifteen scoring rounds instead of five, and it changed the answers

Batch 008 lost six of eleven slots to bands it could not resolve, on a card that downclocked
mid-rental. This batch ran **15 scoring rounds instead of 5** (`cli.BATCH_ROUNDS`).

| | rental 45 (batch 008) | **rental 46 (batch 009)** |
|---|---|---|
| scoring rounds | 5 | **15** |
| IQR range across slots | 0.0074 – **0.1511** | 0.0072 – **0.0213** |
| identity IQR | 0.0193 | **0.0115** |
| `inconclusive` slots | **6 of 11** | **1 of 10** (the identity) |

The same card downclocked again — SM 2910 → 2400 MHz, drifting between slots throughout —
and it no longer mattered. **Three conclusions this batch reached are ones batch 008 could
not have**: that the int4 head *loses* (0.9851 ± 0.0099, where rental 45 saw 1.0105 ±
0.0263 and could only call it inconclusive), that the static cache *wins* (1.0150 ± 0.0123,
resolved for the first time in three attempts), and that the cache adds nothing to the conv.

**The cost estimate in the manifest was wrong by 9x and the writeup should say so.** It
claimed "a round is ~2 s of wall clock for both columns", from the ~900 ms median each
column spends *inside the timed region*. Measured here, rounds are **~18 s apart**: the
2048-token prefill that `run_interleaved` excludes from every timed region is setup, and
setup is most of the wall clock. Ten extra rounds cost **~180 s a slot**, not 20 — about
25 minutes across this batch. It was worth it, and the arithmetic in `docs/BATCHES.md` is
now the measured one.

---

## 4. The two declined slots, and one floor that tested the wrong proposition

```
skipping 059-conv-and-head-triton-op: 055-int4-head-triton-op measured no ratio, needed 1.0
skipping 061-int4-mlp-torch-dequant:  056-int4-head-torch-dequant measured 1.0171, needed 1.02
```

`059` declined correctly: its ingredient did not compile.

**`061` declined for a bad reason, and the reason is instructive.** It is the MLP — 52.75%
of per-token bytes, a 1.6545x ceiling, the largest prize in the backlog — gated on the head
slot reaching **1.02**. The head slot returned **1.0171**, missing by 0.3%.

The floor's *stated* purpose was to check "has the compiler shown it can fuse a grouped
dequantisation into a GEMV prologue at all?" **The dump answers that with an unambiguous
yes**, and the floor asked an absolute-ratio question instead — one whose answer depends on
what fraction of the step the head happens to be and on how fast the card is that hour.
The right criterion was available and was not used: **`056` against `054`**, which is +3.2%
and nowhere near the band.

This is `AGENT.md` §8's "a gate is only a gate if it can resolve the thing it measures" in a
third costume. Batch 003's bar was finer than its statistic; batch 007's floors were fine;
this one was *well-resolved and about the wrong quantity*. **A precondition should name the
proposition its slot depends on, and where that proposition is a comparison, the floor
belongs on the comparison.** `batch.Precondition` can only express "slug ≥ float", which is
now a known limitation rather than an accident.

`061` is **unexecuted, not refuted**, and it is the first slot of the next batch.

---

## 5. Cost, and the guard that paid for itself immediately

| | |
|---|---|
| Instance | 52273140, RTX 5090, machine 140734 — **the same host as rental 45** |
| Rate | $0.4896/hr |
| Billed | **79.00 minutes, $0.6446** |
| Fixed cost before slot 0 | **~22 minutes** (rental 45: ~72) |
| Eight slots run | 364-514 s each |
| Two slots declined | 0 s |
| Lifetime | **$9.1429, 45 instances created, 45 destroyed, zero leaked** |

**`DF_CACHE_MAX_PUSH_MB` fired on its first rental and saved about 33 minutes.**

```
WARNING: not sending the 2236 MB compile cache up: over the 512 MB ceiling
(DF_CACHE_MAX_PUSH_MB). Rental 45 spent 33 billed minutes pushing 2.2 GB to save
~3.5 minutes of compiling. This rental compiles cold, deliberately.
```

Fixed cost fell from ~72 minutes to ~22 on the same host and the same image. The cold
compiles cost what they were budgeted to: round 0 of the first real candidate took 186 s.
**The guard traded 3.5 minutes of compiling for 33 minutes of uplink and the trade was
correct.**

Slots are more expensive than batch 008's — 364-514 s against 129-335 — and 15 rounds is
most of that. The net was still a cheaper rental with more resolution.

---

## 6. What is settled, and what the next batch asks

**Settled.**

1. **The compiler beats our hand-written kernel at the tied LM head**, 1.0171 against
   0.9851, same weights, same function, same process, layer 2 identical.
2. **Inductor fuses a grouped int4 dequantisation into a 248320-wide GEMV prologue, and
   the final RMSNorm with it.** Read out of generated code, not inferred: one reduction
   kernel carrying `__rshift__`, `bitwise_and`, `mm`, `mean`, `rsqrt`, and no weight-sized
   buffer anywhere.
3. **Dispatch savings do not add.** The static cache is +1.5% alone and +0.0% on top of the
   fused conv.
4. **Fifteen scoring rounds changes conclusions, not just error bars**, and it is what
   lets a 1.5% effect be called.
5. **`torch.library.triton_op` does not trace our GEMV on torch 2.11** — recorded with its
   exact error rather than as "it didn't work".

**Open, in the order they are worth a slot.**

1. **`061-int4-mlp-torch-dequant`, ungated.** 52.75% of per-token bytes, a 1.6545x ceiling,
   and the mechanism it depends on is now established from the generated code rather than
   hoped for. It should have run this rental.
2. **The shipped pair has never been measured.** Promoting `int4_head_torch_dequant` makes
   `apply_champions` install it beside `inline_causal_conv`, and **that composition has not
   been benchmarked** — `058` composed the conv with the *retired* kernel. First slot after
   the ingredients.
3. **Why does `058` fall 1.5% short of the product of its parts?** Both mechanisms are now
   fusion-friendly, so the obvious suspect — one barrier interfering with another — is
   gone.
4. **Make a new registration compile before the batch does.** `055` cost a slot to a
   trace-time failure that a five-second `torch.compile` of the installed module, in the
   dump step that already runs, would have caught.
5. **A published int4 kernel as an unscored column** is now a different question than it
   was: the thing to beat at this site may be inductor's own output.
