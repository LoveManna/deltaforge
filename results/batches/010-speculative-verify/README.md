# Batch 010 — a verify costs 32%, and almost none of it is the extra tokens

> **The headline is a number nobody had, and it is much worse than the spec predicted.**
> A three-token verify costs **γ(2) = 1.316** and a five-token verify **γ(4) = 1.404**,
> against a traffic model that budgeted 1.06 and 1.11. The two points together say the cost
> is **not per-token**: entering the multi-token path at all costs ~23%, and each extra
> token after that costs ~4.4%.
>
> **And the batch's own kill criteria do not follow from the batch's own cost model.** Both
> were crossed, yet substituting the measured γ back into §1's formula still shows 1.34x at
> k=2 and 1.45x at k=4 for an int4 self-draft at p = 0.9. What the measured γ really does is
> move break-even acceptance: from ~0.06 to **0.317 mean accepted tokens per cycle** for a
> free drafter. The n-gram drafter delivers **0.016**. Short by 20x, on the text workload
> built to give it its best chance.

Rental 54, RTX 4090, 2026-09-30. Five slots, all measured, nothing voided.
Session cost across eight rentals: **$0.811**, of which seven rentals and $0.540 bought no
measurement at all — see §5.

---

## The scorecard

| Slot | Outcome | Ratio | IQR | ms/token cand | ms/token ref | Δ | γ = 1/ratio | mean accepted | Predicted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `000-identity` | calibrated | **1.0002** | 0.0001 | 10.118 | 10.120 | −0.002 | — | — | `identity` ✅ |
| `062-verify-inflation-k4` | **`loss`** | **0.7120** | 0.0055 | 14.217 | 10.123 | +4.094 | **1.404** | **0.000** | `loss` ✅ (0.87–0.95 ❌) |
| `063-verify-inflation-k2` | **`loss`** | **0.7596** | 0.0052 | 13.328 | 10.124 | +3.204 | **1.316** | **0.000** | `loss` ✅ (0.93–0.97 ❌) |
| `064-spec-ngram-k2` | **`loss`** | **0.7958** | 0.0036 | 12.725 | 10.125 | +2.600 | 1.257 | **0.016** | `inconclusive` ❌ |
| `065-spec-ngram-k4` | **`loss`** | **0.7344** | 0.0062 | 13.786 | 10.124 | +3.662 | 1.362 | **0.016** | `loss` ✅ |

The identity slot returned **1.0002 at an IQR of 0.0001** — the tightest calibration this
project has recorded. Every ratio here is admissible, and all five slots report
`dynamo compiled 2 graph(s)`, so none is a silent eager fallback.

**This is an RTX 4090, and the harness said so before any candidate ran:** *"This project
has never recorded this GPU model before, so there is nothing to compare it against and no
prediction here was derived from it."* The reference ran at 10.12 ms/token and 849 GB/s.
Every ratio below is against that same card in the same interleaved rounds; no number here
may be compared with rental 46's 7.05 ms/token.

---

## 1. γ is a step, not a slope

| | tokens in the verify | γ | what the spec's traffic model said |
|---|---:|---:|---:|
| ordinary decode | 1 | **1.000** by definition | — |
| `063` k=2 | 3 | **1.316** | 1.06 |
| `062` k=4 | 5 | **1.404** | 1.11 |

Fit γ = a + b·(k+1) to the two measured points: **b = 4.40% per token**, **a = 1.184**.

1 → 3 tokens costs **31.6%**. 3 → 5 tokens costs **6.7%**. The extrapolation to a one-token
verify gives 1.23, but a one-token verify *is* ordinary decode, where γ = 1.000 by
definition. **So there is a discontinuity of roughly 23% at the seq=1 → seq>1 boundary
itself**, and it is the dominant term in every number in this batch.

The spec has no term for it. §1 reasoned that "the weights do not move: 8411.51 MB either
way", priced the KV, state and intermediate traffic at 176.29 MB/token (+8.2% of bytes at
k=4), added 2.9% for state versioning, and named dispatch as "the open half" without
predicting it. **The open half is the whole answer.** The measured per-token slope, 4.4%, is
about twice the 2.1% the traffic model implies, which is consistent with the scan's extra
launches. The 23% step is something else, and §4 ranks the suspects.

**`062`'s acceptance histogram is what makes this a measurement of γ and not of a drafter.**
2159 cycles, every one accepting zero drafts:

```
{"block_size": 4, "cycles": 2159, "mean_accepted": 0.0,
 "histogram": {"0": 2159, "1": 0, "2": 0, "3": 0, "4": 0}}
```

Acceptance is zero by construction *and* by measurement, so 1/0.7120 is γ(4) and nothing
else. This is what `034-static-cache-cudagraphs` could not say about its own 2.0%.

---

## 2. The registered kill criteria are wrong, and they are wrong in the unsafe direction

Registered in the spec, before any measurement:

* γ(4) > 1.25 → k = 4 is dead.
* γ(2) > 1.15 → the whole hypothesis is dead at this model's shape.

Both are crossed. **Neither conclusion follows from §1's own formula.** Substituting the
measured γ into `speedup = (1 + Σpⁱ) / (k·d/v₁ + γ)` with the spec's own `d/v₁ = 0.355`:

| drafter | k | p = 0.9 | p = 0.8 | p = 0.7 |
|---|---:|---:|---:|---:|
| int4 self-draft, measured γ | 2 | **1.34x** | 1.21x | 1.09x |
| int4 self-draft, measured γ | 4 | **1.45x** | 1.22x | 1.04x |

At p = 0.9 the hypothesis still wins on these γ values — and **k = 4 wins by more than
k = 2**, though k = 4 is the one its criterion declares dead. The criteria were evidently
set by eye rather than derived, and a session that applied them mechanically would have
closed the largest open entry in `HYPOTHESES.md` on arithmetic that does not support it.

What the measured γ actually does is move break-even. Break-even needs tokens per cycle to
exceed the cost multiplier:

| drafter | break-even mean accepted, predicted γ | break-even, measured γ |
|---|---:|---:|
| free (`d = 0`), k=2 | 0.06 | **0.317** |
| int4 self-draft, k=2 | 0.51 (as p) | **0.63** (as p) |

That is a serious narrowing. It is not a refutation, and this file does not record one.

**The reusable lesson:** a kill criterion is a prediction too, and it has to be *derived*
from the cost model and checked against it, not chosen because it looks strict. Four
hypotheses were registered against bars that the spec's own §1 contradicts.

---

## 3. The n-gram drafter: never 1, sometimes 2

```
064 k=2  {"cycles": 2125, "mean_accepted": 0.016, "histogram": {"0": 2108, "1": 0, "2": 17}}
065 k=4  {"cycles": 2125, "mean_accepted": 0.016, "histogram": {"0": 2108, "1": 0, "2": 17, "3": 0, "4": 0}}
```

2108 cycles accepted nothing. **Zero cycles accepted exactly one.** 17 cycles — 0.8% —
accepted the full two-token block. Prompt-lookup on this text either finds a literal repeat
and gets the whole block, or finds nothing.

The arithmetic closes: 2108×1 + 17×3 = 2159 emitted tokens, the same total `062` and `063`
emitted in 2159 single-token cycles.

**`065` explains itself from its own histogram.** At k=4 the counts at 3 and 4 are both
zero: extending the block bought *no* additional acceptance while costing 6.7% more γ. That
is why `065` (0.7344) sits below `064` (0.7958), and it is what its registered rationale
predicted — the one substantive prediction in this batch that was right for the stated
mechanism.

**`064`'s prediction was wrong, and in an interesting way.** It was registered as
`inconclusive`, "lands within noise of 063". It landed **4.8% above** `063` against an
identity IQR of 0.0001 — far outside noise. The drafter measurably works. It is simply 20x
too weak, and this was the text workload, added for precisely this slot.

**One thing this batch cannot explain.** Both n-gram ratios beat `(1 + mean_accepted)/γ`:
0.7958 against 0.7717 predicted, and 0.7344 against 0.7233. That is 1.5–3%, several times
the IQR, in the direction of the *more expensive* drafter being faster than its own
acceptance accounts for. Recorded as open; nothing here measures it.

---

## 4. What to measure next, if anything

The 23% step at seq=1 → seq>1 is now the whole hypothesis, and it is a claim about the
decode graph rather than about drafting. Ranked suspects, none of them tested:

1. **The full-attention mask.** `reference.py`'s `if seq_len > 1` branch builds a
   position-aware causal mask that the seq=1 path skips entirely. 8 of 32 layers.
2. **The rollback layer's unrolled scan.** `k+1` sequential `recurrent_gated_delta_rule`
   launches per layer against the reference's 1 — 72 launches at k=2, 120 at k=4, across 24
   layers. This scales with k, so it cannot be the step, but it is in the slope.
3. **Autotune config starvation on this card.** Every candidate compile logged
   `No valid triton configs ... triton_mm Required: 110592 Hardware limit: 101376` on the
   verify's `k+1` shape — a shape the reference column never compiles. A 5090 has more
   shared memory per SM and would offer a wider pool, so **γ measured here is plausibly
   inflated, and by an unknown amount.** Any decision that turns on γ's exact value needs a
   5090 first.

Slot `a` was designed to be worth a rental on its own terms, and it was: it produced the
one number that governs everything else. The cheapest next step is not a drafter. It is
**one 5090 rental of `062` and `063` alone**, to learn whether 23% is this model's decode
graph or this card's shared-memory limit.

---

## 5. Seven rentals bought nothing, and the reason was one line of ssh

| rental | card | died at | after | cost |
|---|---|---|---|---|
| 47 | 5090 | `fetch-weights` | 24.00 min | $0.164 |
| 48 | 5090 | `pip install torch` | 3.75 min | $0.027 |
| 49 | 5090 | image pull, `docker login failed` | 13.23 min | $0.096 |
| 50 | 5090 | slot 0's benchmark | 32.18 min | $0.241 |
| 51 | 5090 | cancelled: host advertised CUDA exactly 12.8 | 0.70 min | $0.005 |
| 52 | 4090 | container refused the account ssh key | 1.27 min | $0.008 |
| 53 | 4090 | **three slots OOM'd**; identity calibrated at 1.0001 | 20.40 min | $0.119 |
| 54 | 4090 | **all five slots measured** | 25.08 min | $0.152 |

**47, 48 and 50 were one bug.** All three died on `Connection ... closed by remote host` at
exit 255, on three different machines, three different gateways and three different stages.
`ssh host "command"` made the batch's stdout *be* the ssh channel, so a paid hour lived
exactly as long as one TCP connection. Keepalives were tried first and rental 50 refuted
them alone: it carried them and still lost the session 32 minutes in. `remote/step.sh` now
runs each step detached under `setsid` and polls its log, so a drop costs a reconnect.
Rentals 30 and 39 were lost to the same thing.

**53 is the rental that paid for itself by failing.** Three slots OOM'd at **23.03 GiB on
k=4 and 23.03 GiB on k=2** — identical to two decimals, which is the measurement that named
the cause: the per-step recording never scaled with the block at all. It recorded a cloned
state and conv window per token for *any* forward, so the benchmark's 2048-token prefill
built 2049 versions per layer and ran 2048 sequential scan launches to do it. That is
`002-compile-cost` again, in a new kernel. **A 32 GB 5090 might well have absorbed it and
hidden the defect**; the 4090 fallback that the raised CUDA floor forced is what found it.

Zero leaked instances across all eight, verified against the vast API after each.

---

## 6. What this batch changed

* **γ exists as a number for the first time**: 1.316 at k=2, 1.404 at k=4, with acceptance
  provably zero, on a card whose identity slot calibrated at 1.0002 ± 0.0001.
* **The cost is a step at the multi-token boundary**, not a per-token tax — which relocates
  the hypothesis from "how good is the drafter" to "why does seq>1 cost 23%".
* **The `sequence` correctness policy earned itself.** It ran against real bf16 divergences
  five times and attributed every one: gaps of 0.0 and 0.125 against a 0.3 ceiling. The
  `approximate` policy this spec originally registered would have returned 264/264 for all
  of them, including through the OOM-causing defect.
* **The 0.3 divergence ceiling earned itself too.** A divergence at gap 0.125 passed; the
  `0.02` this project once shipped would have failed it unconditionally.
* **Four kill criteria are withdrawn** as underived — see §2.
* **The n-gram drafter is refuted on text**, at 0.016 against a 0.317 bar, with a histogram
  that says why: it finds literal repeats or nothing.
