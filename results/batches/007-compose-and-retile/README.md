# Batch 007 — the champion is card-dependent, and the conv regression is not the tile

> **Rental 43, 2026-09-20.** RTX 5090, instance 51815165, machine 9105, $0.4622/hr,
> **42.67 billed minutes, $0.3287**, destroyed cleanly. One launch, one instance, no
> refusals. Session total **42.67 minutes, $0.3287**. Lifetime **$7.5055 across 42
> instances, 42 destroyed, zero leaked.**
>
> `calibrated: true` — `000-identity` measured **1.0101** at an IQR of **0.01093**, which
> is the largest identity offset and by far the widest identity band this project has
> recorded. Six slots ran, three declined on their registered floors.
>
> **Predictions scored 2 of 6.** Every kernel slot predicted `win`; one returned one, at
> a fifth of its predicted magnitude and inside the harness's own offset.
>
> **The champion does not reproduce.** `022-int4-head` measured **1.0791** on rental 40
> and **1.0161** here — the same kernel, the same site, the same heuristic tile, the
> tuner off. Net of the identity offset it saves **nothing**.
>
> **And `029`'s 0.7937 was not the tile.** Re-run at the champion's tile it returned
> **0.8111**. The `output_code` dump names the mechanism: the fused conv is a fusion
> barrier, and the composed graph recomputes the linear-attention state reduction
> **twice per layer, 24 times per token**, on identical inputs at an identical grid.

---

## The scorecard

| Slot | Outcome | Ratio | IQR | ms/token Δ vs its own reference | Tile in `launch_shapes` | Layer 2 | Predicted |
|---|---|---:|---:|---:|---|---|---|
| `000-identity` | calibrated | **1.0101** | 0.01093 | −0.102 | — | exact | `identity` ✅ |
| `035-int4-head` | **`win`** | **1.0161** | 0.00407 | −0.097 | *(none — heuristic)* | 0.9318, 0.01674 nats | `win` ✅ |
| `036-int4-head-static-cache` | `inconclusive` | 1.0094 | 0.01334 | −0.107 | *(none — heuristic)* | 0.9318, 0.01674 nats | `win` ❌ |
| `037-int4-head-wide-tile` | `loss` | **0.9343** | 0.01627 | +0.728 | `[128, 64, 1, 8, 3]` | 0.9318, 0.01674 nats | `win` ❌ |
| `038-int4-head-deep-pipe` | `loss` | **0.9502** | 0.01356 | +0.523 | `[64, 64, 1, 4, 5]` | 0.9318, 0.01674 nats | `win` ❌ |
| `039-int4-head-and-conv` | `loss` | **0.8111** | 0.01883 | +2.580 | *(none — heuristic)* | 0.9318, 0.01674 nats | `win` ❌ |
| `040-int4-head-conv-cache` | `precondition_failed` | — | — | — | — | — | `win` (**untested**) |
| `041-wide-tile-and-cache` | `precondition_failed` | — | — | — | — | — | `win` (**untested**) |
| `042-wide-tile-conv-cache` | `precondition_failed` | — | — | — | — | — | `win` (**untested**) |

Baseline on this card: **1374 ms per 128 tokens = 10.73 ms/token, 8587.80 MB/token,
800 GB/s — 44.6% of an RTX 5090's 1792 GB/s vendor peak.** Rental 40 ran the same
reference at **6.70 ms/token and 1282 GB/s**. Every prediction in this batch was derived
from rental 40's number and every one of them was wrong by roughly that factor.

---

## 1. The card, first, because everything else is read through it

This rental's RTX 5090 ran the reference **1.61x slower** than rental 40's, and the usual
explanations do not hold:

| | rental 40 | rental 42 | **rental 43** |
|---|---:|---:|---:|
| reference ms/token | **6.70** | 7.17 | **10.73** |
| reference achieved | **1282 GB/s** | 1198 GB/s | **800 GB/s** |
| memory clock | 13801 MHz | 13801 MHz | **13801 MHz** |
| SM clock | 2827 MHz | 2902 MHz | **2955 MHz** |
| driver | 580.159.03 | 580.159.03 | 580.159.04 |
| torch / triton | 2.11.0+cu128 / 3.6.0 | 2.11.0+cu128 / 3.6.0 | 2.11.0+cu128 / 3.6.0 |
| host platform | `5.15.0-181-generic` | `5.15.0-185-generic` | **`6.10.0-hiveos`** |

**Identical memory clock and a higher SM clock, on the same software, 1.61x slower.** The
clock figures are observed at capture time and not locked, so they are weak evidence — but
they are the evidence that rules out the obvious reading, which is that we rented a
downclocked card. The one column that differs is the host: machine 9105 runs **HiveOS**, a
mining distribution, and it is also the lowest-reliability host this project has accepted
(0.9808 against a 0.98 filter).

**This is not a diagnosis and the writeup must not pretend it is.** A persistent power
limit is the obvious suspect on a mining host and this batch captured no power telemetry to
test it with. What *is* established: two cards that report the same memory clock and the
same driver can differ by 1.6x on this workload, so **the card is an uncontrolled variable
of the same order as the kernels being measured**, and every absolute prediction this
project makes is conditioned on a number it does not control.

**The cheap instrument already exists.** `000-identity` reports the reference column's
achieved bandwidth before any kernel slot runs — 845 GB/s here against 1198-1282 on the two
prior rentals. A pre-flight that compares that figure against what this GPU model has
recorded before, and says so loudly, costs nothing and would have framed every number below
correctly from minute 27 instead of from the writeup. Whether it should *abort* is a
separate question and the answer is probably no: a slow card still produces valid
within-slot ratios, and this rental's most valuable finding came from one.

---

## 2. The champion does not reproduce, and that is the batch's result

`035-int4-head` is `022-int4-head`: the same kernel, the same single site, the same
heuristic tile, `DF_TILE_TUNE` off, `launch_shapes` empty as it must be. It is the slot
`AGENT.md` §6 has demanded since rental 42 carried a number across sessions.

| | rental 40 (`022`) | **rental 43 (`035`)** |
|---|---:|---:|
| median ratio | **1.0791** | **1.0161** |
| IQR | 0.00034 | 0.00407 |
| identity that day | 1.0008 | **1.0101** |
| reference ms/token | 6.70 | 10.73 |
| ms/token removed | **0.493** | **0.097** |

And the last row is the one that matters, because the identity slot removed **0.102
ms/token** by installing nothing at all:

| slot | ms/token Δ | what it installed |
|---|---:|---|
| `000-identity` | **−0.102** | nothing |
| `035-int4-head` | **−0.097** | int4 on the tied LM head |
| `036-int4-head-static-cache` | **−0.107** | that, plus 64 static cache tensors |

**Three slots, three different candidates, one number.** The int4 head's saving on this
card is **0.00 ± 0.11 ms/token** — the identity slot's own IQR is ±0.0109 in ratio, which
is ±0.11 ms/token, and the spread between the three deltas is 0.010. `035`'s `win` verdict
is correct as the classifier defines it (1.0161 clears 1.0 by more than its own 0.0041 IQR)
and it is **not a measurement of the kernel**: it is the systematic advantage the candidate
column carries in every slot, which the identity champion exists to expose and did.

The ceiling arithmetic is unchanged and still right — 1271.40 MB/token becomes 317.85 plus
19.87 of fp32 group scales, a **1.1220x** ceiling. What changed is that on a card running
at 800 GB/s the step is no longer spending its time where those bytes are. The candidate
column achieved **717 GB/s** against the reference's 799 on 11.1% fewer bytes.

**What this does not license.** `022` is still the champion: it is the best-measured
candidate this project has, it won on both cards it has been benchmarked on, and nothing
here beat it. But its headline **1.0791 is a property of rental 40's card as much as of the
kernel**, and `LEADERBOARD.md` now says so in the champion block rather than in a footnote.

---

## 3. `036` — the composition that was supposed to be the champion

`036-int4-head-static-cache` is batch 005's `026`, declined on rental 40 and never run
since. Two mechanisms as far apart as this project has: bytes inside one matmul, and
inductor's per-call alignment check outside every kernel. Predicted **1.08-1.11** from
0.493 + 0.094 ms off 6.70.

It measured **1.0094, `inconclusive`** — *below* the head alone, and below identity.

The static cache contributed 722 GB/s against `035`'s 717 on the candidate column, which is
0.7% and inside the slot's 0.0133 IQR. **Both halves of this composition measured zero on
this card**, so the slot says nothing about whether they compose. It is not evidence
against composition; it is a composition of two things that each measured nothing, and the
honest record of it is that the experiment did not execute on hardware that could resolve
it.

`034` won at 1.0196 on rental 42 by removing the alignment check from 64 tensors across 128
decode steps. That is a *dispatch* saving, denominated in microseconds of CPU work, and it
should have been **more** visible on a slower card, not less. That it was not is the one
result in this batch that the card explanation does not obviously cover, and it is worth a
line in the next batch rather than a paragraph of speculation here.

---

## 4. The tile: four points, and the one nobody chose is still the best

Both pins landed exactly as registered and neither leaked:

```
035-int4-head              launch_shapes {}                                    ← heuristic
036-int4-head-static-cache launch_shapes {}                                    ← heuristic
037-int4-head-wide-tile    launch_shapes {'int4 248320 2560': [128,64,1,8,3]}  ← the pin
038-int4-head-deep-pipe    launch_shapes {'int4 248320 2560': [64,64,1,4,5]}   ← the pin
039-int4-head-and-conv     launch_shapes {}                                    ← cleared again
```

`039` ran *after* both pinned slots and came back on the heuristic, so `_install_head`'s
clear of its `_TUNED` key held on the box and not only in `test_a_pinned_tile_does_not_leak_into_the_next_slot`.
**This was the one failure the batch could have hidden**, and the field that would have
hidden it is the field that proves it did not.

| the head, group-128 int4 | tile `[BLOCK_N, BLOCK_K, SPLIT_K, warps, stages]` | rental | ratio |
|---|---|---|---:|
| heuristic | `[64, 64, 1, 4, 3]` | 40 | **1.0791** |
| heuristic | `[64, 64, 1, 4, 3]` | 43 | **1.0161** |
| searched on the card | `[256, …, 1, …]` | 42 | 0.9920 |
| **wide, pinned** | `[128, 64, 1, 8, 3]` | 43 | **0.9343** |
| **deep, pinned** | `[64, 64, 1, 4, 5]` | 43 | **0.9502** |

`037` cost **+0.83 ms/token** and `038` **+0.63**, both measured against the heuristic in
the same process twenty minutes apart. The registered reading of `037 < 1.0` was *"the
pressure suspect is wrong and the grid is what matters, which points the next batch at
BLOCK_N=32 rather than at another wide tile."* That reading stands, and `038` strengthens
it: deepening the pipeline from 3 stages to 5 — the one axis rental 42's search never
varied alone, and the axis that would pay if the kernel were latency-bound — also loses.

**Three deliberate attempts to beat a tile nobody chose on purpose, by three different
mechanisms, all worse.** Two of them here were pinned in advance and ranked by the ratio
the whole decode step returns, which is the one selection method rental 42 did not
discredit. `docs/HYPOTHESES.md` entry 9 now records that the heuristic tile is the best of
four measured points and that the remaining untried direction is *thinner*, not wider.

### `graphs_compiled: 0` on `037` and `038` is a false positive

Both pinned slots recorded `dynamo compiled 0 graph(s)` and tripped the WARNING that says
the ratio compares eager against compiled. **It does not, and the counter is what is
wrong.**

| slot | candidate round 0 | candidate steady state |
|---|---:|---:|
| `035-int4-head` | **70437 ms** (compiling) | 1371 ms |
| `036-int4-head-static-cache` | **63639 ms** (compiling) | 1352 ms |
| `037-int4-head-wide-tile` | **1501 ms** (nothing to compile) | 1448-1485 ms |
| `038-int4-head-deep-pipe` | **1452 ms** (nothing to compile) | 1432-1478 ms |

`graphs_compiled` is a delta on dynamo's `unique_graphs`, so a **guard-passing cache hit
scores zero**. The tile is not in the traced graph — it is written into `_TUNED` at install
time and read as a launch parameter — so `037`'s graph is identical to `035`'s and dynamo
correctly reused it. Rental 35's real fallback ran candidates 6.8x slow at ratios of
0.146-0.157; these ran within 7% of the reference and passed layer 2 at **0.9318 /
0.01674 nats**, which only a working int4 head produces.

The counter is still worth having — it caught rental 35 — but **`0` means "no new graph",
not "no compile"**, and after a slot that compiled an identical graph, `0` is the healthy
value. The warning needs to distinguish the two, and the round-0 timing it already records
is enough to do it.

---

## 5. `039` — the largest unexplained number, explained structurally

`029` returned **0.7937** on rental 42 with the tuned BLOCK_N=256 tile. `039` is the same
pair with the champion's tile restored, and it returned **0.8111**. Two points, 1.7% apart,
on different cards, with the tile variable removed.

**The tile was not the cause. The composition is.** The conv saves 0.093 ms/token alone
(rental 40) and costs **+2.68 ms/token** composed with the int4 head, net of identity.

`--dump-install tiled_int4_head,fused_causal_conv` rendered `TORCH_LOGS=output_code` for
exactly this pair — the first time the dump has been pointed at a real composition — and
the reference and candidate graphs are both in the same file, so they diff directly.

**What the conv did, as designed:**

| per decode step | reference `[0/2]` | candidate `[0/5]` |
|---|---:|---:|
| `extern_kernels` calls | 24 | **0** |
| `triton_poi_*` launches | 89 | **41** |
| `triton_per_*` launches | 96 | 96 |
| custom-op dispatches | 0 | 25 (24 conv + 1 head) |
| total launches | **507** | **483** |

Every cuDNN call is gone and 48 pointwise launches with them. The `cat` disappeared exactly
where it should: `_to_copy__unsafe_view_add_cat_mean_mm_mul_pow_rsqrt_t_transpose_view`
(15 calls) became `_to_copy__unsafe_view_add_fused_causal_conv_step_mean_mm_mul_pow_rsqrt_t_transpose_view`.
**The kernel does what it claims, and the composition still launches 24 fewer kernels than
the baseline it is 19% slower than.**

**What it also did:**

| per decode step | reference `[0/2]` | candidate `[0/5]` |
|---|---:|---:|
| `triton_red_*` launches | 298 | **321** |
| `empty_strided_cuda` allocations | **59** | **190** |
| allocated bytes | 3.67 MB | 5.96 MB |

The 23 extra reductions are one duplicated kernel per layer, and the two call sites are
identical in their inputs and their grid:

```
REF  combined: ..._clone_copy__div_exp_..._sigmoid_silu_softplus_..._sum_..._9.run(
                   arg13_1, arg10_1, buf7, arg12_1, buf5, buf6,  buf10, buf12, arg13_1, buf13,  4096, 128)

CAND prologue: ..._clone_exp_..._softplus_..._sum_..._4.run(
                   arg13_1, arg10_1, buf7, arg12_1, buf5, buf6,  buf8,                          4096, 128)
CAND full:     ..._clone_copy__div_exp_..._sigmoid_softplus_..._sum_..._6.run(
                   arg13_1, arg10_1, buf7, arg12_1, buf5, buf6,  buf8, buf9, buf11, arg13_1, buf12, 4096, 128)
```

`arg13_1` is `(1, 32, 128, 128)` — the linear-attention recurrent state, 524288 elements.
The reference reduces it **once per layer**. The composition materialises the prologue into
`buf8` **and then recomputes the same reduction inline anyway**, on the same six inputs at
the same 4096x128 grid, 24 times per token.

**The mechanism is a fusion barrier.** `torch.ops.deltaforge.fused_causal_conv_step` is an
opaque custom op, so inductor cannot fuse across it: it must materialise the op's inputs
and outputs into real buffers, which is the 59 → 190 allocations, and it splits a producer
chain it previously fused — then recomputes the shared prologue rather than reading the
buffer it just wrote. **A custom op does not cost what its own kernel costs. It costs that
plus whatever inductor can no longer fuse around it**, and that bill is paid in the
*neighbouring* kernels, where nobody thinks to look.

### What this does *not* establish, and the next instrument

**The structure is named; the magnitude is not closed.** 24 extra reductions of a 2 MB
state and 131 extra allocator calls come to roughly 0.1-0.7 ms/token on this card by any
arithmetic this repository can do from a dump. The measurement is **+2.68**. So the dump
has narrowed the question by a lot and not answered it, and the writeup says so rather than
rounding a plausible mechanism up to a sufficient one.

Two cheap next steps, in order:

1. **Dump the conv installed alone** — `--dump-install fused_causal_conv`, now a one-line
   change. If the duplicated reduction is present there too, the barrier is the conv's and
   has nothing to do with the head, and `025`'s 1.0144 won *despite* it. If it is absent,
   the duplication is a property of the *pair* and the two installers interact through
   dynamo's tracing in a way nothing here predicts.
2. **A profile.** The dump counts launches; it cannot price them. This is the first
   question in the project that a launch census cannot answer.

### The design mistake this batch made, having been written to fix it

Batch 007 exists because rental 42 carried `022`'s number across sessions. It fixed that
for the head — `035` is the whole reason the rest of this writeup can be trusted — and then
**made the identical mistake with the conv**: it carried `025`'s 1.0144 from rental 40 and
never re-measured the conv alone on this card. So `039` cannot separate "the composition is
bad" from "the conv is bad on an 800 GB/s card", and on a rental where *every* mechanism
went flat, that is not a remote possibility.

**A composition slot is worth nothing without its ingredients measured in the same
process.** `035` was in this batch and `025` was not, for no better reason than that the
head's number was the one under suspicion. Both were equally one rental old.

---

## 6. The three declined slots are unexecuted, not refuted

```
skipping 040-int4-head-conv-cache: 039 measured 0.8111, needed 1.05
skipping 041-wide-tile-and-cache:  037 measured 0.9343, needed 1.08
skipping 042-wide-tile-conv-cache: 041 measured no ratio, needed 1.1
```

Every floor was registered before the rental and every one fired correctly: each declined
slot was a strictly worse version of a measurement the batch already held. They cost
nothing and they proved nothing. `040` — all three wins at once, the batch's arithmetic
maximum — **has now been registered twice and run zero times**, and its status is
unchanged from before this rental.

---

## 7. Cost

| | |
|---|---|
| Instance | 51815165, RTX 5090, machine 9105, HiveOS host |
| Rate | $0.4622/hr |
| Billed | **42.67 minutes, $0.3287** |
| Fixed cost before slot 0 | ~27 minutes (image, torch, 9.32 GB checkpoint, GPU suite, composition dump) |
| Slots | 6 ran at 176-270 s each; 3 declined at zero cost |
| Session | 42.67 minutes against the 180-minute gate |
| Lifetime | **$7.5055, 42 instances created, 42 destroyed, zero leaked** |

The 75-95 minute, $0.65-0.85 estimate in the run plan was **high by about half**, because
three slots declined and because two of the six reused a compiled graph instead of paying
~200 s to build one. `docs/BATCHES.md` gets both numbers.
