# Batch 006 — the tile was not the problem, and the promise pays without the graph

> **Rental 42, 2026-09-20.** RTX 5090, instance 51656940, machine 144419, $0.5163/hr,
> **50.82 billed minutes, $0.4373**, destroyed cleanly. A 4090 died first on CUDA
> `Error 804` for a further 5.13 minutes and $0.0337; two more launches created nothing.
> Session total **55.95 minutes, $0.4710**. Lifetime **$7.1768 across 41 instances, 41
> destroyed, zero leaked.**
>
> `calibrated: true` — `000-identity` measured **1.0053** at an IQR of **0.00086**.
> `graphs_compiled: 3` on every slot that ran. Five slots ran, three declined.
>
> **Predictions scored 1 of 4.** Every slot predicted `win`; one was.
>
> **`034-static-cache-cudagraphs` won at 1.0196, IQR 0.00149 — and recorded
> `cudagraph_nodes: 0`.** It is a win whose stated mechanism never fired, which is only
> legible because the counter was there. The batch's central hypothesis — that the
> hand-written GEMV was losing on a tile nobody had measured — is **refuted**, twice, by
> measurement.

---

## The scorecard

| Slot | Outcome | Ratio | IQR | Tile chosen | Layer 2 | Predicted |
|---|---|---:|---:|---|---|---|
| `000-identity` | calibrated | **1.0053** | 0.00086 | — | exact | `identity` ✅ |
| `028-int4-head-tuned` | `loss` | 0.9920 | 0.00481 | BLOCK_N **256**, SPLIT_K 1 | 0.9318, 0.01674 nats | `win` ❌ |
| `029-head-and-conv` | `loss` | **0.7937** | 0.01706 | inherited from 028 | 0.9318, 0.01674 nats | `win` ❌ |
| `030-int4-mlp` | `loss` | **0.3545** | 0.00063 | BLOCK_N 128, SPLIT_K 4 / 16 | 0.8977, **0.04868 nats** | `win` ❌ |
| `031-int4-mlp-and-head` | `precondition_failed` | — | — | — | — | `win` (untested) |
| `032-int4-wide-and-head` | `precondition_failed` | — | — | — | — | `win` (untested) |
| `033-int4-wide-head-and-conv` | `precondition_failed` | — | — | — | — | `win` (untested) |
| `034-static-cache-cudagraphs` | **`win`** | **1.0196** | 0.00149 | — | 264/264, **0.0 nats** | `win` ✅ |

Baseline on this card: **917.9 ms per 128 tokens = 7.17 ms/token, 8587.80 MB/token,
1198 GB/s — 66.9% of an RTX 5090's 1792 GB/s vendor peak.** Rental 40's card ran the same
reference at 1282 GB/s, so this is a slower physical 5090; ratios are within-slot and
absolute times are provenance.

---

## 1. The batch's premise was wrong, and the measurement that says so is 2.3x

The whole batch rested on one inference from rental 40: the GEMV won on the head because
the head launches 3880 programs, therefore the heuristic tile — which targets 256, *one
wave* — was leaving performance on every other site. `tune_launch_shape` timed every
candidate tile on the card instead of deriving one, with the heuristic always the first
candidate, so **the floor was supposed to be the champion.**

It was not. `028` installs the identical kernel on the identical site as `022-int4-head`
and returned **0.9920 where `022` returned 1.0791.**

| the head, group-128 int4 | tile | programs | in-situ achieved |
|---|---|---:|---:|
| `022`, rental 40, heuristic | BLOCK_N 64, SPLIT_K 1 | 3880 | **656 GB/s** |
| `028`, rental 42, tuner's pick | BLOCK_N **256**, SPLIT_K 1 | 970 | **282 GB/s** |

*(In-situ site time is backed out the way rental 40 backed out its own: the candidate
differs from the baseline at one site, so the site's candidate time is
`t_head_baseline + (t_candidate - t_baseline)`, with `t_head_baseline` the head's bytes at
the baseline's model-wide rate. 1.0639 + 0.064 = 1.128 ms for 317.85 MB.)*

**So the tuner chose a tile 2.3x slower than the one it was meant to improve on.** Not
because the search space was wrong — the heuristic's tile is in it — but because the thing
it measured is not the thing that matters.

### Why an isolated micro-benchmark ranked the tiles backwards

The tuner times one site at a time, in a loop, with `triton.testing.do_bench` flushing L2
between iterations, on an otherwise idle card. It reported **0.2 ms for the head's
327 MB — 1639 GB/s, higher than the whole compiled model achieves (1198)** — and then
chose the tile that maximised that number.

In the decode step the same kernel achieves 282 GB/s. A 5.6x gap between a micro-benchmark
and the same code in place is not noise; the micro-benchmark is measuring a different
machine state. The card boosts through a tight loop of one kernel; in the real step that
kernel is one of 508 launches, arriving with a cache and a clock shaped by the 507 around
it. **A tile ranked on the first does not rank on the second**, and the direction is
consistent with what got picked: fewer, fatter programs look better when launch and
scheduling overhead dominate, which is exactly what an idle-card loop over a single
kernel exposes.

**What is now measured, and it is a fact about method rather than about Triton:
per-site micro-benchmarking does not select tiles for this workload.** `DF_TILE_TUNE` now
defaults to off, so nothing inherits a selector one rental has discredited; the search
space, the rounds and the plausibility guard stay in the tree for a tuner that measures
the decode step instead.

---

## 2. The 83% behind the layer projections is not reachable by tiling, and now there is a number

`030-int4-mlp` put group-128 int4 on the 96 MLP projections — **52.75% of everything the
compiled column moves, a 1.6545x ceiling** — and returned **0.3545**.

| | ms/token | MLP bytes | MLP achieved |
|---|---:|---:|---:|
| compiled baseline | 7.18 | 4529.85 MB | ~1198 GB/s |
| `030` candidate | **20.27** | 1132.47 MB | **67 GB/s** |

It ties at 331 GB/s. It reached 67.

That number is the one worth keeping, because it is the third independent measurement of
the same thing and the first at a tile chosen by search:

| int4, group-128 | sites | achieved |
|---|---|---:|
| `014` (batch 003, rental 37) | all 248 layer projections | 65 GB/s |
| `030` (this rental) | 96 MLP projections, searched tile | **67 GB/s** |
| `022` (rental 40) | the tied LM head alone | **656 GB/s** |

Three rentals, two kernel structures, two tile-selection methods, and the layer
projections sit at 65-67 GB/s every time while the head sits an order of magnitude
above. **Whatever separates the head from the MLP, the tile is not it** — batch 005's
"grid-starved" reading survives as a description of the head and fails as an explanation
of the MLP, because 288 and 320 program instances did not behave like 40 and 80.

The candidate was **correct**: top-1 0.8977 against a 0.8333 bar, mean KL **0.04868 nats**
against 0.10. The bar was derived before the rental from `014` minus `022` scaled by the
MLP's share of layer weight bytes, predicted **~0.048**, and measured 0.04868. The gate
method works; the kernel does not.

### What the preconditions bought

`031`, `032` and `033` declined on `030 >= 1.00`. With the MLP at 0.3545, all three would
have re-measured a 67 GB/s kernel over more sites — which is precisely the five slots
batch 003 spent before preconditions existed. **Three slots and roughly 15 billed minutes
saved, and nothing informative lost.** Their predictions are unscored because they tested
nothing.

---

## 3. The composition that cost 20%, which nothing had ever run

`029` is `028` plus `025-fused-causal-conv`, the kernel that won on its own at 1.0144 on
rental 40. Composed, the pair returned **0.7937**.

| | ms/token | Δ against this run's baseline |
|---|---:|---:|
| baseline | 7.164 | — |
| `028` — head alone | 7.248 | +0.064 |
| `029` — head + conv | **9.079** | **+1.915** |

The head accounts for 0.064 of that. **The fused causal conv, which saved 0.093 ms/token
by itself on rental 40, added ~1.85 ms/token here.** It is the same kernel, unmodified
since rental 40.

This rental cannot say why, and the honest position is that it is unexplained. What it
*can* say is that it is a **composition** effect and not a property of either kernel:
both were measured alone, one on this very rental. `027-conv-head-static-cache` was
supposed to test this pair on rental 40 and its precondition declined it, so **this is the
first execution of the two together** — the fourth time a declined slot has turned out to
be an unexecuted path rather than a verified one.

Ranked suspects for the next rental, cheapest first: the conv installer patches
`GatedDeltaNet` while the head installer replaces the root class, and a root-class swap is
a new dynamo code object — so the conv's custom op may be landing in a differently-fused
graph than it did alone. `graphs_compiled: 3` on both slots, so it is not a compile
fallback. `TORCH_LOGS=output_code` on the composed candidate names it in one step and
costs a slot.

---

## 4. `034` won, and its mechanism did not fire

**`cudagraph_nodes: 0`, `cudagraph_skips: 127`** — the same counters rental 40's `021`
recorded, and this time beside a **1.0196 win at an IQR of 0.00149**, on a candidate that
is bit-identical to the reference (264/264 agreement, **0.0 nats, 0.0 absolute error**).
Layer 1 confirms the install: **64 of 64 decode-cache tensors marked static.**

So the slot is a win *and* entry 8's hypothesis is still untested. Those are different
claims and the record can hold both only because the counter is in it.

**What the 2.0% actually is.** `mark_static_address` does two things. It is the promise
the cudagraph mutation check wants — which was refused again — and it puts every marked
tensor into `static_input_idxs`, which inductor's launch path reads for a *second* purpose:
`get_input_idxs_to_check` skips the per-call alignment test for a static input, and
`copy_misaligned_inputs` skips the copy behind it. Sixty-four cache tensors, on every one
of 128 decode steps, stop being checked. Achieved bandwidth rose 1198 → **1222 GB/s on
identical bytes**, which is what removing work from the dispatch path looks like when the
kernels are untouched.

**And this session refuted batch 005's two ranked suspects for free, on a CPU.** Running
the tiny config under `TORCH_LOGS=cudagraph_static_inputs` prints

```
Adding static input pos 5 for source L['cache'].layers[0].conv
Adding static input pos 12 for source L['cache'].layers[0].recurrent
Adding static input pos 25 for source L['cache'].layers[1].keys
```

— so the mark *does* reach `static_input_indices`, through a plain Python object, and
`_extract_tensor_dict` *does* stamp it. Neither suspect survives, and neither needed a GPU
to kill.

**What is left, and the log now names it.** The captured skip reasons show the decode
region refused for `mutated inputs (64 instances)` while cudagraph trees are recorded
around it for other graphs:

```
Recording cudagraph tree for symint key (2048, 2048)
skipping cudagraphs due to mutated inputs (64 instances). Found from : .../reference.py, line 715, in forward
Recording cudagraph tree for symint key 2049
Recording cudagraph tree for symint key 2050
```

Two things to read there. The count is still **64** with all 64 tensors marked, so the
marking is not reaching *this* check on torch 2.11 (the box) even though it reaches the
index list on 2.14 (the laptop) — a version difference is now the leading suspect and it
is cheap to test. And `symint key 2049, 2050, ...` confirms the other half of entry 8's
warning: `cache.seq_len` reaches the graph as an int, `cudagraphify_impl` keys its cache
on every int input, and a 128-token decode therefore wants 128 recordings.

---

## 5. Costs, measured

| | Rental 42 |
|---|---|
| Whole rental, provisioning to destroy | **50.82 minutes, $0.4373** at $0.5163/hr |
| Fixed cost before slot 0 | **27.12 minutes** — of which the 9.32 GB checkpoint was 1m27s |
| Slot 0 — identity | **179 s** (bench 162 s) |
| Slots 1-4 | **178-362 s** |
| Of which the approximate correctness gate | 10.3-16.4 s |
| Of which install-time tile tuning | **12.1 s** (one shape) to **14.8 s** (two shapes) |
| A candidate that replaces the root class | **+198 s** in its first warmup round |
| Declined slots | **0 s** — three `precondition_failed` |
| Peak memory | 8.39 GiB allocated, 16.61 GiB reserved, of 31.36 |

**The fixed cost doubled against rental 40's ~13 minutes, and sending the compile cache
*up* is most of it.** Rental 40 fixed the sync that shipped 1.4 GB of cache inside the
9.5 MB repo; this rental still pushes that cache deliberately, to the path the run reads.
It bought a warm reference compile — `compile_candidate_compiled` rounds to 0.0 s on every
slot — and cost more than it saved on a rental this short.

**A candidate that replaces the root model class now costs ~198 s, not 72-218 s.** Rental
40 recorded the range; this rental puts `029` and `030` at 197.7 s and 199.2 s, which is
the recompile and not the tuner (the tuner is separately timed inside `candidate_build`).

---

## 6. Getting a card cost three launches

| | |
|---|---|
| Launch 1 | RTX 4090, machine 27290, advertised `cuda_max_good` **exactly 12.8** — died on `Error 804: forward compatibility was attempted on non supported HW`. **5.13 minutes, $0.0337.** |
| Launch 2 | `DF_MIN_CUDA=12.9`, machine excluded. Vast returned **HTTP 429** on the 4090 search; no offers; exit 4, nothing created. |
| Launch 3 | Same filters, no 429, and genuinely **no 4090 offer at 12.9** under $0.45. Exit 4, nothing created. |
| Launch 4 | `--max-rate 0.65` — five RTX 5090 offers, took one at **$0.5163/hr, CUDA 13.0, 32607 MB**. |

**Blocker 11 is confirmed, not merely recurring: a host advertising exactly the floor is
the one that fails.** Rental 33 was 12.8 and failed; rental 34 was 13.0 and was fine;
this 4090 was 12.8 and failed. Three for three. Raising `DF_MIN_CUDA` to 12.9 is the right
filter and it empties the 4090 pool at $0.45, so the two knobs are coupled: **tighten the
CUDA floor and the rate ceiling has to move with it.**

An exit-4 refusal costs nothing and is not a fault. It is also not always a market
transient — launch 2's was a 429 and launch 3's was a real empty pool, and only the log
line distinguishes them.

---

## 7. What is settled, and what the next rental should do

**Settled:**

* **The tile is not what separates the head from the layer projections.** A searched tile
  made the head **2.3x slower** (656 → 282 GB/s) and left the MLP at **67 GB/s**, within
  noise of batch 003's 65 at a completely different tile and kernel structure.
* **Per-site micro-benchmarking does not select tiles for this workload.** 1639 GB/s in
  isolation against 282 GB/s in place, on the same kernel and the same site.
* **`mark_static_address` is worth ~2.0% on the launch path with no CUDA graph recorded**,
  bit-identically, at an IQR of 0.0015.
* **Batch 005's two suspects for `021` are dead**, killed on a CPU with a log flag.
* **Composing the fused conv with the tiled head costs ~1.85 ms/token**, where the conv
  alone saved 0.093. Unexplained, and the first time the pair has run.

**Not settled:** what CUDA graphs are worth here — entry 8 is now **0 for 2** on slots
that were built to test it and never recorded a node. The next attempt must establish that
a graph is recorded *before* it reports a ratio.

**The next rental, in order:**

1. **Re-measure the champion.** `022-int4-head` was not benchmarked this rental — `028`
   replaced it with a different tile — so the leaderboard's 1.0791 has not been re-verified
   on a second card. That is a gap this batch created and it is one slot.
2. **`TORCH_LOGS=cudagraph_static_inputs` on the rented box**, in the dump step, which
   costs nothing and settles whether torch 2.11 populates `static_input_idxs` from
   `mark_static_address` the way 2.14 does. If it does not, the fix is a torch bump, not a
   kernel.
3. **`034` composed with `022`**, which is `026` from batch 005 and has still never run:
   2.0% on the dispatch path and 7.9% on the head are disjoint and both are measured.
4. **`029`'s 20% regression, with `TORCH_LOGS=output_code` on the composed candidate.**
   Two kernels that win alone and lose 20% together is the largest unexplained number this
   project now holds.
5. **Stop attacking the layer projections with this kernel family.** Three rentals, two
   structures, two tile-selection methods, 65-67 GB/s every time. The next idea there has
   to be a different algorithm — a published int4 kernel as an unscored column, which
   `docs/HYPOTHESES.md` has been asking for since it was written, would say whether the
   shape is hard or our kernel is.
