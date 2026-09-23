# Batch 008 — the barrier was the whole cost, and it was never the composition

> **Rental 45, 2026-09-23.** RTX 5090, instance 52251586, machine 140734, $0.4896/hr,
> **115.52 billed minutes, $0.9427**, destroyed cleanly. Two launches: rental 44 refused
> our ssh key six minutes in and cost $0.0500. Session total **121.75 minutes, $0.9927**.
> Lifetime **$8.4982 across 44 instances, 44 destroyed, zero leaked.**
>
> `calibrated: true` — `000-identity` measured **0.9972** at an IQR of **0.0193**, and the
> card pre-flight put the reference at **1197 GB/s**, 0.93x the best this project has
> recorded. **All eleven slots ran**; nothing was declined and nothing errored.
>
> **Predictions scored 5 of 11**, the worst scorecard this project has produced, and the
> two it got right are the two that matter.
>
> **`045-inline-causal-conv` won at 1.0765** — the largest margin ever measured here
> against a reference timed the same day — **and it is bit-identical to the reference.**
> Its control, the same arithmetic wrapped in an opaque custom op, measured **0.7854** in
> the same process. **One variable, 37%.**
>
> **And the conv regression was never about composition.** `044` alone lost 21%; `050`
> composed lost 20%. Rental 43 spent a writeup on an interaction that does not exist.

---

## The scorecard

| Slot | Outcome | Ratio | IQR | ms/token cand | ms/token ref | Δ | cand GB/s | Layer 2 | Predicted |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `000-identity` | calibrated | **0.9972** | 0.0193 | 7.173 | 7.172 | +0.001 | 1197 | exact | `identity` ✅ |
| `043-int4-head` | `inconclusive` | 1.0105 | 0.0263 | 7.609 | 7.664 | −0.055 | 1003 | 0.9318, 0.01674 nats | `win` ❌ |
| `044-fused-causal-conv` | **`loss`** | **0.7854** | 0.0446 | 9.421 | 7.502 | **+1.919** | 912 | **1.0000, 0.0 nats** | `win` ❌ |
| `045-inline-causal-conv` | **`win`** | **1.0765** | 0.0552 | **7.048** | 7.775 | **−0.727** | **1219** | **1.0000, 0.0 nats** | `win` ✅ |
| `046-static-decode-cache` | `inconclusive` | 1.0214 | 0.0406 | 7.269 | 7.434 | −0.165 | 1181 | 1.0000, 0.0 nats | `inconclusive` ✅ |
| `047-int4-head-narrow-tile` | `inconclusive` | **0.9617** | 0.0691 | 8.002 | 7.628 | +0.374 | 954 | 0.9318, 0.01674 nats | `loss` ❌ |
| `048-int8-head` | `inconclusive` | 0.9865 | 0.0342 | 7.485 | 7.475 | +0.010 | 1062 | 0.9811, 0.00040 nats | `win` ❌ |
| `049-fp8-head` | **`loss`** | 0.9562 | 0.0105 | 8.039 | 7.553 | +0.486 | 989 | 0.9659, 0.00158 nats | `win` ❌ |
| `050-int4-head-and-conv` | **`loss`** | **0.7963** | 0.0074 | 9.308 | 7.424 | +1.884 | 820 | 0.9318, 0.01674 nats | `loss` ✅ |
| `051-int4-head-and-inline-conv` | `inconclusive` | 1.0443 | **0.1511** | 7.373 | 7.569 | −0.196 | 1035 | 0.9318, 0.01674 nats | `win` ❌ |
| `052-int4-head-inline-conv-cache` | **`win`** | **1.0747** | 0.0127 | 7.351 | 7.920 | −0.569 | 1039 | 0.9318, 0.01674 nats | `win` ✅ |

`Δ` is candidate minus reference **within the slot**, which is the only comparison this
rental supports — see §6 on the clock.

---

## 1. The result: the same arithmetic, twice, 37% apart

`044` and `045` compute **the same function**, and the record proves it rather than
asserting it: both are **bit-identical to the reference** — layer 1 at a relative error of
exactly `0.0`, layer 2 at **264/264 agreement and 0.00000 nats**. Same taps, same fp32
accumulation, same round to bf16, same silu, same shifted history.

| | how the four taps reach inductor | ratio | ms/token | candidate GB/s |
|---|---|---:|---:|---:|
| `044-fused-causal-conv` | one `torch.library.custom_op` per layer | **0.7854** | 9.421 | 912 |
| `045-inline-causal-conv` | four torch multiplies inductor may fuse | **1.0765** | **7.048** | **1219** |

**1219 GB/s is the highest any column has achieved in this project** — 68% of a 5090's
1792 GB/s vendor peak, against the reference's 1104 in the same slot — and it was reached
by *deleting* a kernel rather than writing one.

### What the dump shows, and it is exact

`--dump-install tiled_int4_head,inline_causal_conv` rendered both columns into one file.
Per decode step:

| | reference `[0/2]` | candidate `[0/5]` |
|---|---:|---:|
| `extern_kernels.convolution` | **24** | **0** |
| `triton_poi_*` launches | 89 | **89** |
| `triton_per_*` launches | 96 | 96 |
| `triton_red_*` launches | 298 | **297** |
| total launches | **507** | **482** |
| `empty_strided_cuda` allocations | 59 | 232 |

**The conv cost nothing to absorb.** The pointwise count did not move: inductor folded the
four taps straight into the two kernels that were already there, and the cuDNN call simply
disappeared.

```
REF   triton_poi_fused__unsafe_view_cat_mm_transpose_5           x24   (builds the cat)
      extern_kernels.convolution                                  x24   (cuDNN)
      triton_poi_fused_copy_copy__slice_6                         x24   (advances the history)

CAND  triton_poi_fused__unsafe_view_cat_mm_slice_transpose_7      x24
      triton_poi_fused__unsafe_view_cat_copy__mm_slice_transpose_8 x24
      (no extern call anywhere in the graph)
```

Compare rental 43's Triton conv, which removed the same 24 cuDNN calls and **added 23
reductions** — the linear-attention state reduction recomputed twice per layer. Here the
reduction counts are 1, 1, 8, 8, 15, 15, 1 on **both** sides: the fusion groups are renamed
(`cat` becomes `select`) and **nothing is duplicated**.

### The allocation count was a red herring, and this rental says so

Rental 43 reported 59 → 190 allocations beside a 19% loss and ranked it as a suspect.
**This candidate allocates 232 and wins.** So allocator traffic was correlated with the
regression, not causal; the duplicated reduction was the mechanism, and only the
duplicated reduction. A writeup that had promoted the allocation count to a cause would
have been confidently wrong, and the only reason it is not in `AGENT.md` as a finding is
that batch 007 wrote "the magnitude is **not** closed" instead.

---

## 2. There was no composition effect. There never was.

Batch 007's central unexplained number was `039-int4-head-and-conv` at 0.8111, and its
writeup called it *"the largest unexplained number in the project"* and reasoned about how
two installers interact. **The interaction does not exist.**

| | rental | ratio |
|---|---|---:|
| `025-fused-causal-conv` alone | 40 | 1.0144 |
| `029` head + conv (tuned tile) | 42 | 0.7937 |
| `039` head + conv (champion's tile) | 43 | 0.8111 |
| **`044` conv alone** | **45** | **0.7854** |
| **`050` head + conv** | **45** | **0.7963** |

The conv loses **21% on its own**. Composed with the head it loses 20%. `050` minus `044`
is +0.011 in ratio, inside both slots' IQRs: **the head adds nothing to the regression and
subtracts nothing from it.** Three rentals of "why do these two not compose" were asking
the wrong question, and the answer cost one three-minute slot placed before the
composition instead of after it.

**This is the rule batch 007 wrote down after breaking it, paying off the first time it was
enforced** — `batches_test.test_every_ingredient_of_every_008_composition_is_measured_alone_first`
now asserts it structurally rather than leaving it to whoever fills the next batch.

### The one thing this leaves open

`025` measured **1.0144** on rental 40 and two later measurements say ~0.79-0.81. The
kernel has not changed. The honest statement is that **rental 40's number is contradicted
and this rental cannot say why** — it is a recorded measurement, not a retracted one, and
the leaderboard now carries both with their rentals named. What is no longer in doubt is
the *current* answer: on a card running the reference at 1145 GB/s, the custom op costs
1.92 ms/token.

---

## 3. The champion does not reproduce, on a third card and a healthy one

`043-int4-head` is `022` unchanged, for the third time.

| | rental 40 | rental 43 | **rental 45** |
|---|---:|---:|---:|
| reference achieved | 1282 GB/s | 845 GB/s | **1197 GB/s** |
| `000-identity` | 1.0008 (IQR 0.00034) | 1.0101 (IQR 0.01093) | **0.9972 (IQR 0.0193)** |
| the int4 head | **1.0791** | 1.0161 | **1.0105** |
| ms/token removed | 0.493 | 0.097 | **0.055** |

Rental 43's 1.0161 could be blamed on an 800 GB/s card. **This one cannot.** The reference
ran at 1197 GB/s, the identity slot carried a *negative* 0.28% offset, and the head still
returned 1.0105 — `inconclusive`, because 0.0105 does not clear its own 0.0263 IQR.

**Two of three cards now put this kernel at ~1%, and the 7.91% belongs to rental 40.** The
byte arithmetic is untouched — 1271.40 MB/token becomes 317.85 plus 19.87 of fp32 group
scales, a 1.1249x ceiling — and the kernel is provably the same one: layer 2 came back at
**0.9318 and 0.01674 nats**, digit for digit what rentals 40 and 43 recorded. What is not
reproducible is the *value of the site*.

---

## 4. Three encodings at one site, and the ordering is the finding

Both 8-bit head kernels ran correctly for the first time. `023` shipped on rental 40 with a
dropped per-channel scale (layer-1 relative error **4511**) and `024` never compiled at
all; both fixes had never executed on a card until now.

| | ratio | layer 2 | candidate GB/s |
|---|---:|---|---:|
| `043` int4, group-128 | **1.0105** | 0.9318, 0.01674 nats | 1003 |
| `048` int8, per-channel | 0.9865 | **0.9811, 0.00040 nats** | 1062 |
| `049` fp8, e4m3 | **0.9562** | 0.9659, 0.00158 nats | 989 |

**int4 > int8 > fp8**, and two registered mechanisms die on that ordering.

* **int4 ahead of int8 confirms the regime**: at a site with parallelism to spare the
  kernel is bandwidth-bound, so halving the bytes beats avoiding the nibble unpack. That
  was the reading rental 40 proposed from byte rates alone; it is now a timing.
* **fp8 behind int8 refutes entry 7 outright.** The whole claim was that e4m3 converts
  *inside* the MMA pipeline on sm_120 where int8→fp32 is an ALU instruction on the
  critical path, so fp8 should be the cheaper of two encodings moving identical bytes. It
  is **3.1% slower**, at the tightest IQR in the batch (0.0105). Same bytes, same site,
  same tile, same kernel structure — the conversion tax runs the other way on this card.

The accuracy predictions were good: int8 at 0.00040 nats and fp8 at 0.00158 land exactly
between batch 003's measured int8 (0.0011) and int4 (0.0919) points, which is the third
consecutive batch where **a bar derived from something this repository measured** passed a
working kernel that a prior-derived bar would have failed.

---

## 5. The tile question is closed

| the head at group-128 int4 | `[BLOCK_N, BLOCK_K, SPLIT_K, warps, stages]` | rental | ratio |
|---|---|---|---:|
| heuristic | `[64, 64, 1, 4, 3]` | 40 / 43 / **45** | 1.0791 / 1.0161 / **1.0105** |
| searched by micro-benchmark | `[256, …]` | 42 | 0.9920 |
| wide, pinned | `[128, 64, 1, 8, 3]` | 43 | 0.9343 |
| deep, pinned | `[64, 64, 1, 4, 5]` | 43 | 0.9502 |
| **narrow, pinned** | `[32, 64, 1, 4, 3]` | **45** | **0.9617** |

**Five points measured in the decode step, and the heuristic nobody chose on purpose is
the best of them.** `047` was registered as a predicted `loss` with a consequence attached,
and it lost: 0.9617, +0.374 ms/token, 954 GB/s against the heuristic's 1003. The
wave-count theory — thinner programs, more of them — was the last mechanism standing after
rental 43 killed register pressure and latency, and it is dead too.

`docs/HYPOTHESES.md` entry 9 now stops spending slots on tiles. The registered next
instrument is a published int4 kernel (Marlin, machete) as an unscored column, which
answers whether *any* hand-written kernel is fast on this shape.

The pin landed exactly as registered and did not leak: `047` recorded
`{'int4 248320 2560': [32, 64, 1, 4, 3]}` and `050`, `051` and `052` — the next slots to
install an int4 head — all recorded `{}`. The stale key visible in `048` and `049` is the
*int4* entry sitting beside their own `int8`/`fp8` keys, which nothing reads.

`047` also recorded `graphs_compiled: 0` and, as batch 007 predicted, it is the healthy
case: its first round took 1.5 s where a compiling candidate's takes 48-195 s. The warning
now prints that evidence beside the counter instead of leaving it to a writeup.

---

## 6. The card downclocked mid-rental, and it is why six slots are inconclusive

| slot | SM clock at capture |
|---|---:|
| `000` – `044` | 2910 MHz |
| `045` | 2850 MHz |
| `046` onward | **2400 MHz** |

The memory clock held at 13801 MHz throughout. The reference column moved with the SM
clock: **7.17 ms/token in slot 0 and 7.92 in slot 10**, a 10% drift *inside one rental*.

Two consequences, and the design handles one of them:

* **Ratios survive.** Every slot times its own reference in the same interleaved rounds,
  so the drift divides out — which is exactly the property `_slot_metadata` records in
  every result because a sceptical reader asks about it.
* **Resolution does not.** IQRs here are 0.0074 to **0.1511**, against rental 40's
  0.00034. Six slots landed `inconclusive` not because their effects are zero but because
  the band is wide: `043` at 1.0105 ± 0.0263 and `046` at 1.0214 ± 0.0406 are both real
  numbers the rental could not resolve.

`051` is the extreme case and it is worth reading carefully. Its **candidate** rounds are
tight — 921, 926, 928, 944, 949, 985 ms — while its **reference** rounds include 1110 and
1179 ms outliers, which is what produced round ratios of 0.9753 and 1.2425 and an IQR of
0.1511. `052` ran the same candidate plus the static cache three minutes later, caught a
clean reference, and returned **1.0747 at an IQR of 0.0127**. So the pair is consistent and
`051`'s verdict is an artefact of the column it was divided by, not of the kernel.

**The instrument that would fix this is more scoring rounds when the IQR is wide**, not a
faster card — and the card pre-flight added this batch would not have caught it, because
the card started fine and degraded. A pre-flight tests the card you were given; it cannot
test the card you will still have in forty minutes.

---

## 7. The predictions, including the four that were wrong

5 of 11. The honest reading is that **the mechanisms were predicted better than the
magnitudes**, and one mechanism was predicted exactly backwards.

| Slot | Predicted | Measured | What the miss says |
|---|---|---|---|
| `043-int4-head` | win | inconclusive | The prediction named 1.02-1.09 and said "below 1.05 is the interesting outcome". It landed at 1.0105 and the interesting outcome is what happened. |
| `044-fused-causal-conv` | win | **loss 0.7854** | Predicted 1.005-1.02 from `025`'s rental-40 number. Wrong by 27%, and it is the batch's most valuable miss: it is what proves the regression is not the composition. |
| `047-int4-head-narrow-tile` | loss | inconclusive | 0.9617 is a loss by any reading; the classifier calls it inconclusive because 0.0383 does not clear a 0.0691 IQR. **The prediction was right and the resolution was not.** |
| `048-int8-head` | win | inconclusive | 0.9865 ± 0.0342. The registered claim was that it wins by less than int4 — it did not win, and the *ordering* claim it was there to test survives. |
| `049-fp8-head` | win | **loss 0.9562** | The mechanism is refuted, not the magnitude. See §4. |
| `051-int4-head-and-inline-conv` | win | inconclusive | 1.0443 ± 0.1511; §6. `052` is the same candidate with a clean reference. |

Two predictions that were right are the two that were registered as falsifiable
alternatives rather than as hopes: `045` **win** and `050` **loss**, which together are the
experiment.

---

## 8. Cost, and the 33 minutes that bought three

| | |
|---|---|
| Rental 44 | instance 52250686, machine 148117 — refused the account ssh key at minute 6 | **6.23 min, $0.0500** |
| Rental 45 | instance 52251586, RTX 5090, machine 140734, $0.4896/hr | **115.52 min, $0.9427** |
| Fixed cost before slot 0 | **~72 minutes** | |
| Eleven slots | 129-324 s each, **~39 minutes total** | |
| Session | 121.75 minutes against the 180-minute gate | **$0.9927** |
| Lifetime | **$8.4982, 44 instances created, 44 destroyed, zero leaked** | |

**72 minutes of fixed cost to buy 39 minutes of measurement is the worst ratio this
project has recorded**, and most of it has a name: the compile-cache push took **~33
minutes** for **2.2 GB**, at about 1.05 MB/s of home uplink, to save a warm compile worth
**268 s cold → 57 s warm ≈ 3.5 minutes**. It then failed to come home — the teardown pull
timed out at `DF_CACHE_PULL_TIMEOUT` (exit 124) — so the rental paid for the upload and
banked nothing.

The cache is a *good* idea that has outgrown its transport. `run_remote.sh` now refuses to
push a cache above `DF_CACHE_MAX_PUSH_MB` (default 512) and says why, because a push that
costs ten times the compile it saves is not an optimisation. The real fix is pruning the
cache or pushing it concurrently with the 9.32 GB checkpoint fetch, and that is registered
in `docs/BATCHES.md` rather than done here.

---

## 9. What is settled, and what the next batch should ask

**Settled.**

1. **A custom op costs what inductor can no longer fuse across it, and here that is 21% of
   the whole decode step** — measured against the identical arithmetic expressed as torch
   operations, both bit-identical to the reference, in the same process.
2. **The conv regression is not a composition effect.** `044` alone and `050` composed are
   within each other's IQRs.
3. **The allocation count is not the mechanism.** 232 allocations won; 190 lost.
4. **int4 > int8 > fp8 at the tied head**, so the site is bandwidth-bound and entry 7's
   MMA-pipeline argument is refuted.
5. **The tile is not what holds the head below the baseline's byte rate.** Five points,
   four deliberate attempts, the heuristic wins.
6. **The int4 head's 7.91% belongs to rental 40.** Two later cards, one of them fast, put
   it at ~1%.

**Open, in the order they are worth a slot.**

1. **Where else is this project paying the barrier?** `tiled_gemv_int4` — the champion of
   `decode_step` — is *itself* a `torch.library.custom_op`, and it appears in the candidate
   graph as four dispatches inductor cannot fuse across. The same experiment that produced
   `045` applies: an int4 head expressed so that inductor schedules it, or registered
   through `torch.library.triton_op` rather than `custom_op`. **That is the single highest
   value hypothesis this project now holds**, because the head is 14.80% of the bytes and
   the mechanism is measured rather than hoped.
2. **More scoring rounds when the IQR is wide.** Six inconclusive slots on a rental where
   the card lost 17% of its SM clock is a resolution problem with a cheap fix.
3. **A published int4 kernel as an unscored column**, now that the tile question is closed.
4. **Prune or parallelise the compile cache push**, which cost 28% of this rental.
