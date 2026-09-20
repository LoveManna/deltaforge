# Run batch 007 — the champion re-measured, its tile pinned, its wins composed

**You are picking up DeltaForge to spend one rental.** Read `AGENT.md` first; this file is
the plan for the batch that is already registered in `src/deltaforge/batches.py` as
`007-compose-and-retile`. Nothing here needs a GPU until section 4.

Everything below is committed before the rental. **Do not edit a prediction, a bar or a
precondition after the box is up** — that is the one thing this project is trying to prove
it does not do.

---

## 1. Why this batch, in five lines

* The leaderboard's champion — `022-int4-head`, **1.0791** — was measured **once, on one
  card, on 2026-09-19**, and rental 42 did not re-measure it. `035` fixes that and every
  other slot is read against it.
* Three wins exist on mechanisms that share nothing: the int4 head (1.0791), the fused
  causal conv (1.0144), the static decode cache (1.0196). **Two of them have never been
  composed, and the one pair that was composed lost 20%.** Four slots here compose them.
* On this one site the **tile is worth 2.3x** (656 GB/s against 282), and both tiles ever
  measured in the decode step were chosen by something other than the decode step. Two
  slots pin a tile in the manifest and let the ratio rank it.
* Nothing here attacks the layer projections. Three rentals, two kernel structures and two
  tile-selection methods have put them at **65-67 GB/s**, and `docs/HYPOTHESES.md` entry 6
  now says the next idea there has to be a different algorithm.
* **Every slot can beat 1.0791**, and `test_every_007_slot_can_beat_the_incumbent_champion`
  asserts it. The head at 4 bits carries a 1.1249x ceiling; the conv and the cache add
  measured savings on top.

## 2. The slots, what each predicts, and what each outcome means

Order is load-bearing and is not sorted. Identity first; the champion's re-measurement
next because everything is read against it; the two ungated compositions after that so a
champion is banked early; the gated ones last.

| # | Slot | Installs | Predicted | Reads as |
|---|---|---|---:|---|
| 0 | `000-identity` | — | 1.00 ± noise | A miss **voids the whole batch**. Note the *sign*: rental 42 carried +0.53%. |
| 1 | `035-int4-head` | `tiled_int4_head` | **1.06-1.10** | The champion on a second card. Below 1.05 means 1.0791 was card-dependent and the leaderboard has to say so. |
| 2 | `036-int4-head-static-cache` | head + static cache | **1.08-1.11** | Batch 005's `026`, never run. 0.493 + 0.094 ms off 6.70 = 1.0960 naive. |
| 3 | `037-int4-head-wide-tile` | `tiled_int4_head_wide` | **1.09-1.12** | BLOCK_N 128 at 8 warps. ~1.079 = the tile is not what binds the head; **< 1.0 = register pressure is the wrong suspect and BLOCK_N 32 is the next point, not another wide one.** |
| 4 | `038-int4-head-deep-pipe` | `tiled_int4_head_deep` | **1.08-1.12** | `num_stages` 5. Read against 037: both winning means the axes are independent and the next batch pins their combination. |
| 5 | `039-int4-head-and-conv` | head + fused conv | **1.08-1.11** | `029` with the tuned tile taken out. ~1.096 = the 20% belonged to the tile; ~0.79 = it belongs to the composition, and this rental's dump names it. |
| 6 | `040-int4-head-conv-cache` | all three | **1.10-1.14** | Batch 005's `027`, never run. Gated on 039 ≥ 1.05. |
| 7 | `041-wide-tile-and-cache` | wide head + cache | **1.10-1.15** | Gated on 037 ≥ 1.08 — below that this is a worse `036`. |
| 8 | `042-wide-tile-conv-cache` | wide head + conv + cache | **1.12-1.17** | Gated on 041 ≥ 1.10. **042 − 041 below the conv's 0.093 ms means two dispatch wins are competing for the same microseconds.** |

The arithmetic behind every range, on rental 40's card (6.70 ms/token, 1282 GB/s):

| | ms/token removed | source |
|---|---:|---|
| int4 head | 0.493 | measured, `022` |
| fused causal conv | 0.093 | measured, `025` |
| static decode cache | 0.094 | measured, `034`, net of that rental's +0.53% offset |
| head at 900 GB/s instead of 656 | 0.639 | arithmetic: 317.85 MB ÷ rate |
| head at 1200 GB/s | 0.727 | arithmetic; the 1.1249x ceiling needs 1282 |

## 3. Before you rent — 20 minutes, no GPU, no money

1. `uv run pytest` green, `uv run ruff check . && uv run ruff format --check .` clean.
2. `uv run python docs/roofline.py` — sanity, and it costs nothing.
3. **Confirm the tile tuner is off.** `DF_TILE_TUNE` defaults to `0` and rental 42 is why.
   If anything in the environment sets it, the pinned slots stop measuring what they say:
   `tune_launch_shape` returns early on a key that is already in `_TUNED`, so a pin
   survives, but `035` would inherit a *searched* tile and re-measure `028`.
4. Read `results/batches/006-tile-and-sites/README.md` §1 and §3. §1 is why the tuner is
   off; §3 is the number slot 5 exists to explain.

## 4. The run

```sh
SESSION="batch007-$(date -u +%Y%m%dT%H%M%SZ)"
LOG="/tmp/$SESSION.log"

remote/run_remote.sh --dry-run --session-id smoke --batch 007-compose-and-retile   # spends nothing

remote/launch.sh --log "$LOG" --session-id "$SESSION" --batch 007-compose-and-retile \
  --dump-install tiled_int4_head,fused_causal_conv \
  --max-rate 0.65
tail -n +1 -F "$LOG"                      # +1, never -n 0. Rental 29 cost a day to this.
remote/launch.sh --status --log "$LOG"    # 0 finished, 1 failed or killed, 2 running
```

`--dump-install` is new and it is the cheapest thing in this run. The `output_code` dump
renders the registry's champions, which is **one** kernel; naming the pair makes it render
the composition that lost 20%, in the step that was going to run the dump anyway. The file
lands in `results/diagnostics/inductor-output-code.txt`.

**Market notes, from rental 42.** `DF_MIN_CUDA=12.9` — a host advertising *exactly* 12.8
has failed on CUDA `Error 804` three times out of three. That filter empties the 4090 pool
under $0.45, so the rate ceiling has to move with it: `--max-rate 0.65` found five RTX 5090
offers. An exit-4 "no offers" costs nothing and is not a fault; check the log line to see
whether it was an empty pool or an HTTP 429 before retrying.

**Expect 75-95 minutes and $0.65-0.85** at ~$0.52/hr: ~27 minutes fixed, then nine slots.
Rental 42's slots ran 178-362 s and **every kernel slot here will sit at the top of that
range**, because all eight replace the root model class — a new dynamo code object, which
rental 42 timed at **~198 s inside the first warmup round** (discarded, so no ratio is
affected). That recompile is not in `COLD_PHASE_ESTIMATES` and it is the largest single
cost in this batch. The session gate is 180 minutes, the batch stops itself before a slot
it cannot finish, and the three gated slots cost nothing when a floor is missed.

## 5. What to watch while it runs

| Watch | Because |
|---|---|
| `000-identity` ratio **and sign** | A miss voids everything. +0.53% on rental 42 was a third of what `034` appeared to win. |
| `graphs_compiled` on every slot | `0` means the ratio beside it is eager against compiled, not a comparison. Nine slots need the recompile limit raised, and `recompile_limit_for` does it. |
| `launch_shapes` in each slot record | It lists only tiles that were *written* — the heuristic is never written — so the four unpinned slots (`035`, `036`, `039`, `040`) must show **`int4 248320 2560` absent**, and `037`/`041`/`042` must show it at `[128, 64, 1, 8, 3]`, `038` at `[64, 64, 1, 4, 5]`. **A pin leaking into an unpinned slot is the one failure this batch could hide**; `_install_head` clears the key and `test_a_pinned_tile_does_not_leak_into_the_next_slot` is the CPU guard, so this is the field that proves it held on the box. |
| Layer-1 checks on composed slots | They name *which* install is missing if a composition silently drops one. |
| `precondition_failed` rows | Expected and healthy if a floor is missed. A declined slot is an **unexecuted** path, not a refuted one — say so in the writeup. |

## 6. Known biases, so you do not read them as findings

* **The byte model ignores quantisation scales.** `decode_bytes_per_token` scales a
  region's bf16 bytes by `bits/16`, so the int4 head is counted as 317.85 MB/token when the
  kernel really reads **337.72** — 317.85 of packed nibbles plus **19.87 MB of fp32 group
  scales** (20 groups × 248320 × 4 B). Ratios are unaffected; the *achieved bandwidth*
  reported for a quantised candidate is ~6% low at the head site and 0.26% low model-wide.
  The champion's "656 GB/s" is ~697 GB/s of real traffic. The bias is conservative, and the
  arithmetic above is what to fix it with if a slot turns on it.
* **Cross-slot ratios share one card and one thermal history.** Comparing `037` with `035`
  is the intended reading and it is sound; comparing either with rental 40 is not, which is
  exactly why `035` exists.

## 7. Writing it up (this is part of the session, not a follow-up)

`AGENT.md` §6.1 has the table. The minimum: `results/batches/007-compose-and-retile/README.md`
with the scorecard and the *causes*, a row per slot in `LEADERBOARD.md`, the champion block
updated (or explicitly re-confirmed at its new number), `docs/HYPOTHESES.md` entry 9
amended with whatever the tile slots say, and `ledger/spend.jsonl` reconciled.

Three things this batch can settle that the docs are currently waiting on:

1. **Whether 1.0791 reproduces.** The leaderboard says in bold that it has not been.
2. **Whether the conv composes.** `029`'s 0.7937 is the largest unexplained number the
   project holds, and slot 5 plus the dump is the whole experiment.
3. **Whether the head's tile has anything left in it.** Two in-place points, 2.3x apart,
   and this batch adds two more.

## 8. What was deliberately left out, and the arithmetic that left it

Do not spend a slot re-deriving these.

* **int8 or fp8 on the head (`023`, `024`).** Their business is unfinished — `023` had a
  dropped scale, `024` would not compile, both fixed — but 8 bits at that site has a
  ceiling of **1.0799**, which is 0.0008 above the incumbent. It cannot win. It is a
  diagnostic about the unpack tax, and this batch is not spending a slot on one.
* **BLOCK_N=32 on the head.** The opposite theory to `037`'s: if the head is short of waves
  rather than of registers, thinner programs win. It is the right next point **if `037`
  loses**, and the wrong one to run beside it, because the two pins would cost two slots to
  answer one question the first can answer alone.
* **An FMA accumulator instead of `tl.dot`** (entry 6 has asked for this since it was
  written). The arithmetic now argues against it at *this* site: the padded `tl.dot` is
  issuing ~39.5 TFLOP/s against a card that does ~400, so the MMA pipe is at 10% and the
  16x waste is not what binds. An FMA form moves that work onto the ALU pipe the nibble
  unpack is already using, and needs a `(BLOCK_K, BLOCK_N)` fp32 accumulator — 16 KB per
  program — where `tl.dot`'s is `(16, BLOCK_N)`. It remains the right experiment on a site
  where the grid, not the tile, is the problem.
* **Entry 8 — CUDA graphs, ceiling ~1.26x, the largest unattacked number in the project.**
  It is 0 for 2 and both attempts recorded `cudagraph_nodes: 0`. What is left is *not* a
  kernel, and neither option fits inside a slot:
  * **The buffer-resident cache.** Inductor's check exempts an input that is a parameter
    or buffer — "parameters and buffers carry that promise structurally", as
    `static_cache.py` already quotes. Our 64 cache tensors reach the graph through
    `L['cache'].layers[i].conv`, a plain Python object, so they are lifted as *inputs*
    whatever `mark_static_address` says. Making them buffers means patching the layer
    classes to read their own state, which is a real build (~2 module classes plus the
    root) and is testable on a CPU with a counting backend: **the claim to check first is
    that the conv history stops appearing as a graph placeholder.**
  * **The torch version.** The box runs 2.11 from the cu128 index and the laptop runs 2.14,
    where the mark demonstrably reaches `static_input_indices`. A bump means a different
    CUDA index and a driver floor, which couples it to blocker 11 and to the rate ceiling.
    It changes both scoring columns equally, so it is legitimate — but it changes the
    baseline, invalidates the warm compile cache, and is a **whole-rental decision, not a
    slot**. Do not do it inside a batch that is also measuring kernels.
* **The layer projections.** 65-67 GB/s across three rentals. Entry 6's remaining ask is a
  published int4 kernel (Marlin, machete) as an **unscored column**, to say whether the
  shape is hard or our kernel is.
