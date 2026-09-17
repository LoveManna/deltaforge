# Bandwidth-Bound GEMV — Implementation Plan (batch 004)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce DeltaForge's first champion — a hand-written Triton kernel whose median
ratio against `torch.compile(mode="max-autotune")` exceeds 1.0 by more than the IQR — by
making the decode GEMV memory-bound first and quantising it second.

**Architecture:** Batch 003 established that the ceiling arithmetic is sound and the kernel
could not reach it: it was instruction-bound, so removing bytes made it slower. This plan
rewrites the GEMV around a tensor-core `tl.dot` accumulator with a K-major weight layout and
split-K, which removes the per-iteration cross-lane reduction that cost batch 003 its
bandwidth; then layers fp8/int8 on top of a kernel that can actually collect the saving. It
also fixes the three gate defects that failed six working kernels, and adds the two harness
capabilities whose absence made batch 003 more expensive than it needed to be — per-column
achieved bandwidth, and slots that can decline to run.

**Tech Stack:** Python 3.12, PyTorch (CUDA CDN build on the rented box, CPU index locally),
Triton, `torch.library.custom_op`, pytest, ruff, uv. Target `Qwen/Qwen3.5-4B` at batch-1
decode on an RTX 5090 (sm_120, 1790 GB/s, 96 MB L2, 170 SMs).

## Global Constraints

- **No fabricated, estimated, placeholder or illustrative numbers anywhere.** If a run
  produced no measurement, the record says so. (`AGENT.md` §6)
- **Every prediction is registered in `src/deltaforge/batches.py` and committed before the
  rental it is tested on.** A prediction written after seeing the number is not a prediction.
- **`uv run pytest` must pass on a clean CPU-only checkout, with no CUDA present.** GPU
  tests carry the `gpu` marker and skip on a real condition.
- **`uv run ruff check . && uv run ruff format --check .` must pass.**
- **`reference.py` is never modified to accommodate a kernel.** `reference_purity_test.py`
  enforces this by AST inspection.
- **Installers key by kernel name, not by replaced operation** — `model.INSTALLERS` and
  `kernels.CHECK_BUILDERS` both. Several kernels replace the same operation.
- **Candidates share the reference's parameter tensors.** Quantised copies are registered as
  **buffers**; `cli._assert_parameters_are_shared` walks `named_parameters` and would reject
  a parameter with no counterpart.
- **Session budget: 180 cumulative billed minutes; month-to-date gate $45** (currently
  $6.157 lifetime). `remote/run_remote.sh` enforces both.
- Line length 110. Tests colocated as `<module>_test.py`; `testpaths` is `src` and `remote`.

---

## The number this plan is aimed at

From rental 37's own records, the compiled baseline moves 9158 MB/token in 7.64 ms —
**1171 GB/s effective, 73% of the card's 1790 GB/s peak.** Define `f` as the candidate
kernel's bytes-per-second as a fraction of that baseline rate. Then:

| `f` | GB/s | bf16 control | int8, layer projections | int8/fp8, + LM head |
|---:|---:|---:|---:|---:|
| **0.284** | **332** | **0.333** | 0.620 | 0.581 | ← batch 003, measured |
| 0.50 | 585 | 0.562 | 1.000 | **1.000** |
| 0.61 | 715 | 0.646 | 1.15 | **1.20** |
| 0.73 | 850 | 0.761 | 1.31 | **1.40** |
| 1.00 | 1171 | 1.000 | 1.64 | **1.85** |

Three consequences that shape everything below.

**The gate is much lower than I claimed at the end of batch 003.** I wrote "the bf16 control
must reach ≥ 0.90". That is wrong and would have cancelled a winning batch: quantisation
halves the bytes, so int8 **ties** the baseline at `f` = 0.50, which shows up as a bf16
control of only **0.562**. The real gate is **bf16 control ≥ 0.56**, and ≥ 0.75 buys ~1.4×.

**The required improvement is 1.8×, not 4×.** Batch 003 sat at `f` = 0.284. Reaching 0.50 is
a 1.76× improvement in kernel efficiency, and reaching 0.73 is 2.6×. Removing 20-72
cross-lane reductions per output from an inner loop is comfortably in that range.

**There is one in-batch signal that says whether the rewrite worked, and it is not the
ratio.** A memory-bound kernel achieves *the same GB/s* at bf16 and at int8, while moving
half the bytes. An instruction-bound one achieves less at int8, because the conversion
competes for issue slots. Batch 003 measured 332 → 141 GB/s and that single collapse is the
whole story. Task 1 makes the harness report it directly.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/deltaforge/harness/bench.py` | **Modify.** `BenchConfig` gains `bytes_per_token`; `BenchResult` gains `achieved_gbps` per column. |
| `src/deltaforge/harness/bytes_model.py` | **Create.** Bytes moved per decode token for a given weight encoding. Pure arithmetic, no torch. Shared by the roofline script and the bench. |
| `src/deltaforge/harness/bytes_model_test.py` | **Create.** |
| `src/deltaforge/batch.py` | **Modify.** `Hypothesis` gains `requires`; `BATCH_OUTCOMES` gains `precondition_failed`. |
| `src/deltaforge/batch_run.py` | **Modify.** `run_batch` evaluates `requires` against finished slots. |
| `src/deltaforge/kernels/tiled_gemv.py` | **Create.** The rewritten kernel: K-major layout, `tl.dot`, split-K. bf16 / fp8 / int8 / int4 variants. |
| `src/deltaforge/kernels/tiled_gemv_test.py` | **Create.** Quantisers, packing, launch geometry, and CPU emulation of the kernel's indexing. |
| `src/deltaforge/kernels/__init__.py` | **Modify.** Register the new kernels and their check builders. |
| `src/deltaforge/model.py` | **Modify.** One installer per new kernel. |
| `src/deltaforge/batches.py` | **Modify.** `BATCH_004`, with predictions and preconditions registered. |
| `remote/run_remote.sh` | **Modify.** An `output_code` dump step before the batch. |

`quantised_linear.py` stays exactly as it is. Its kernels are registered, retired, and
measured; they are the control that batch 004's numbers are read against, and deleting a
measured loss would throw away the comparison that makes the new number mean something.

---

### Task 1: Report achieved bandwidth per column

Batch 003's central finding required computing GB/s by hand from two files after the rental.
A ratio says a kernel lost; **GB/s says whether it lost on bytes or on instructions**, and
that is the difference between "quantisation does not work here" and "this kernel is not
memory-bound yet".

**Files:**
- Create: `src/deltaforge/harness/bytes_model.py`
- Create: `src/deltaforge/harness/bytes_model_test.py`
- Modify: `src/deltaforge/harness/bench.py`
- Modify: `src/deltaforge/harness/bench_test.py`

**Interfaces:**
- Produces: `decode_bytes_per_token(config, *, weight_bits: dict[str, int], context_length: int) -> float`
  returning MB/token; `BenchResult.achieved_gbps: dict[str, float]`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test for the bytes model**

```python
# src/deltaforge/harness/bytes_model_test.py
from ..config import qwen3_5_4b_config
from .bytes_model import decode_bytes_per_token


def test_all_bf16_reproduces_the_roofline_scripts_total():
    """The number docs/roofline.py prints, from the same arithmetic in an importable place."""
    mb = decode_bytes_per_token(qwen3_5_4b_config(), weight_bits={}, context_length=2048)
    assert abs(mb - 9158.23) < 1.0


def test_halving_the_layer_projections_removes_half_their_bytes():
    config = qwen3_5_4b_config()
    full = decode_bytes_per_token(config, weight_bits={}, context_length=2048)
    int8 = decode_bytes_per_token(config, weight_bits={"layers": 8}, context_length=2048)
    # Layer projections are 7140 MB/token of the 8411 MB of weights.
    assert abs((full - int8) - 3570.0) < 5.0


def test_quantising_the_head_is_worth_its_own_share():
    config = qwen3_5_4b_config()
    layers = decode_bytes_per_token(config, weight_bits={"layers": 8}, context_length=2048)
    both = decode_bytes_per_token(config, weight_bits={"layers": 8, "head": 8}, context_length=2048)
    assert abs((layers - both) - 635.5) < 5.0


def test_an_unknown_region_is_refused_rather_than_ignored():
    """A typo in a manifest must not silently score against bf16 bytes."""
    import pytest

    with pytest.raises(ValueError, match="unknown weight region"):
        decode_bytes_per_token(qwen3_5_4b_config(), weight_bits={"mpl": 8}, context_length=2048)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest src/deltaforge/harness/bytes_model_test.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'deltaforge.harness.bytes_model'`

- [ ] **Step 3: Write `bytes_model.py`**

Lift the per-region weight arithmetic out of `docs/roofline.py` into importable form. Regions
are `"mlp"`, `"linear_attn"`, `"full_attn"`, `"head"`, plus the alias `"layers"` meaning the
first three. `weight_bits` maps a region to its stored bit width; absent means 16.

```python
"""Bytes moved per decode token, as a function of how the weights are stored.

`docs/roofline.py` has always computed this and printed it. It is importable now because
the benchmark needs it: a ratio says a kernel lost, and only bytes-over-time says whether it
lost on bandwidth or on instruction issue. Batch 003 turned on exactly that distinction and
had to derive it by hand from two files after the rental was over.

No torch. This is arithmetic over a `ModelConfig` and it is tested on a CPU.
"""
```

Signature and the region split:

```python
WEIGHT_REGIONS = ("mlp", "linear_attn", "full_attn", "head")
_LAYER_REGIONS = ("mlp", "linear_attn", "full_attn")


def decode_bytes_per_token(config, *, weight_bits=None, context_length=2048) -> float:
    bits = _expand(weight_bits or {})
    total = 0.0
    for region, bf16_mb in _weight_megabytes(config).items():
        total += bf16_mb * bits[region] / 16.0
    return total + _non_weight_megabytes(config, context_length)
```

`_expand` resolves the `"layers"` alias, defaults every region to 16, and raises
`ValueError(f"unknown weight region {name!r}; known: {...}")` on anything else.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `uv run pytest src/deltaforge/harness/bytes_model_test.py -v`
Expected: 4 passed.

- [ ] **Step 5: Make `docs/roofline.py` import it rather than duplicate it**

Two copies of this arithmetic drifting apart would mean the printed ceiling and the recorded
bandwidth disagree, and nothing would notice. Run
`uv run python docs/roofline.py` and confirm the printed total is still `9158.23`.

- [ ] **Step 6: Write the failing test for per-column bandwidth**

```python
# in src/deltaforge/harness/bench_test.py
def test_a_column_reports_the_bandwidth_it_achieved():
    """The number that says whether a kernel is memory-bound or issue-bound.

    Batch 003's kernels all lost; only GB/s distinguished "moved fewer bytes and still lost"
    from "never reached the bus". A candidate that achieves the same GB/s as the reference
    while moving half the bytes is bandwidth-bound and winning; one whose GB/s falls as its
    bytes fall is issue-bound, which is a different problem with a different fix.
    """
    config = BenchConfig(rounds=3, warmup_rounds=0, bytes_per_token={"compiled": 9158.23,
                                                                    "candidate_compiled": 5588.0})
    result = run_interleaved(
        {"compiled": lambda: None, "candidate_compiled": lambda: None},
        config,
        timer=_FakeTimer({"compiled": 7.64 * 128, "candidate_compiled": 7.64 * 128}),
    )

    # Same wall clock, 39% fewer bytes -> 39% lower achieved bandwidth.
    assert result.achieved_gbps["compiled"] == pytest.approx(1171, rel=0.01)
    assert result.achieved_gbps["candidate_compiled"] == pytest.approx(714, rel=0.01)


def test_bandwidth_is_absent_rather_than_guessed_when_bytes_are_not_supplied():
    """A column with no byte model must report nothing, not zero — zero is a measurement."""
    result = run_interleaved({"compiled": lambda: None}, BenchConfig(rounds=3, warmup_rounds=0),
                             timer=_FakeTimer({"compiled": 1.0}))
    assert result.achieved_gbps == {}
```

Match `_FakeTimer` to whatever `bench_test.py` already uses; reuse the existing fixture
rather than adding a second one.

- [ ] **Step 7: Run it and watch it fail**

Run: `uv run pytest src/deltaforge/harness/bench_test.py -k bandwidth -v`
Expected: FAIL, `TypeError: BenchConfig.__init__() got an unexpected keyword argument 'bytes_per_token'`

- [ ] **Step 8: Implement it**

`BenchConfig` gains `bytes_per_token: dict[str, float] = field(default_factory=dict)`.
`BenchResult` gains `achieved_gbps: dict[str, float] = field(default_factory=dict)`, computed
from each column's **median** ms and its MB/token, divided by `workload["decode_tokens"]`.
Carry it into `to_dict()` so it lands in the slot record.

- [ ] **Step 9: Run the tests and make sure they pass**

Run: `uv run pytest src/deltaforge/harness -q`
Expected: all pass.

- [ ] **Step 10: Wire it into the batch runner and print it**

In `batch_run.BatchRunner.run_slot`, build `bytes_per_token` for the two scoring columns from
`self.hypothesis.weight_bits` (Task 5 adds that field; until then pass `{}` for the candidate
and the bf16 total for the reference), and log one line per slot:

```python
self.log(
    f"[batch] {hypothesis.slug}: achieved "
    + ", ".join(f"{k} {v:.0f} GB/s" for k, v in result.achieved_gbps.items())
)
```

- [ ] **Step 11: Commit**

```bash
git add src/deltaforge/harness/bytes_model.py src/deltaforge/harness/bytes_model_test.py \
        src/deltaforge/harness/bench.py src/deltaforge/harness/bench_test.py \
        src/deltaforge/batch_run.py docs/roofline.py
git commit -m "Report the bandwidth a column achieved, not only the ratio it scored"
```

---

### Task 2: Slots that can decline to run

Batch 003 spent five of seven slots on variants whose outcome was fully determined the moment
slot 1 returned 0.2801. The ordering rule already encodes the intent that early slots inform
later ones; the runner cannot act on it.

**Files:**
- Modify: `src/deltaforge/batch.py`
- Modify: `src/deltaforge/batch_test.py`
- Modify: `src/deltaforge/batch_run.py`
- Modify: `src/deltaforge/batch_run_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Hypothesis.requires: Precondition | None`, where
  `Precondition = namedtuple("Precondition", "slug floor reason")`; the outcome string
  `"precondition_failed"`.

- [ ] **Step 1: Write the failing tests**

```python
# in src/deltaforge/batch_test.py
def test_a_slot_can_require_an_earlier_slots_ratio():
    p = Precondition(slug="015-gemv-bf16", floor=0.56, reason="quantisation cannot tie below this")
    hypothesis = _hypothesis(slug="017-fp8", requires=p)
    assert hypothesis.requires.floor == 0.56


def test_a_precondition_naming_a_slot_that_is_not_earlier_in_the_batch_is_refused():
    """A forward reference would silently never fire, which is worse than not having one."""
    with pytest.raises(ValueError, match="must name an earlier slot"):
        Batch(batch_id="x", hypotheses=(
            _hypothesis(slug="a", requires=Precondition(slug="b", floor=0.5, reason="r")),
            _hypothesis(slug="b"),
        ))


def test_precondition_holds_when_the_named_slot_cleared_the_floor():
    assert precondition_holds(Precondition("a", 0.56, "r"), {"a": 0.60}) is True


def test_precondition_fails_when_it_did_not():
    assert precondition_holds(Precondition("a", 0.56, "r"), {"a": 0.28}) is False


def test_a_precondition_on_a_slot_that_errored_fails_closed():
    """No ratio is not the same as a good ratio. Failing open would run the batch anyway."""
    assert precondition_holds(Precondition("a", 0.56, "r"), {"a": None}) is False
```

```python
# in src/deltaforge/batch_run_test.py
def test_a_failed_precondition_records_why_rather_than_running_the_slot():
    """`precondition_failed` is distinct from `not_run` on purpose.

    `not_run` means the clock arrived first; `starved` means the rental scored nothing;
    this means the batch decided the slot could not tell us anything. Flattening them would
    erase the evidence for whether the gate was set correctly.
    """
    batch = Batch(batch_id="t", hypotheses=(
        _slot("015-gemv-bf16"),
        _slot("017-fp8", requires=Precondition("015-gemv-bf16", 0.56, "kernel not memory-bound")),
    ))
    runner = FakeRunner({"015-gemv-bf16": 0.28})

    results, _, _ = run_batch(runner, batch, budget=SlotBudget(deadline_epoch=time.time() + 3600))

    assert [r.outcome for r in results] == ["loss", "precondition_failed"]
    assert "kernel not memory-bound" in results[1].error
    assert runner.slots_run == ["015-gemv-bf16"]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/deltaforge/batch_test.py src/deltaforge/batch_run_test.py -k precondition -v`
Expected: FAIL, `ImportError: cannot import name 'Precondition'`

- [ ] **Step 3: Implement in `batch.py`**

```python
class Precondition(NamedTuple):
    """A floor an earlier slot's ratio must clear for this slot to be worth running.

    Batch 003 is why. Its slot 1 measured a hand-written GEMV against cuBLAS on identical
    bytes and returned 0.2801, which settled every quantisation slot behind it: a kernel at
    28% of the baseline's byte rate cannot collect a byte saving. Five slots then re-measured
    that fact at different bit widths. The batch could not stop, and the ordering rule that
    put the informative slot first had no way to act on what it found.
    """

    slug: str
    floor: float
    reason: str


def precondition_holds(precondition, ratios_by_slug) -> bool:
    """Fails closed. A slot that errored has no ratio, and no ratio is not a good one."""
    if precondition is None:
        return True
    ratio = ratios_by_slug.get(precondition.slug)
    return ratio is not None and ratio >= precondition.floor
```

`Batch.__post_init__` validates that every `requires.slug` appears strictly earlier in
`hypotheses`. Add `"precondition_failed"` to `BATCH_OUTCOMES`.

- [ ] **Step 4: Implement in `batch_run.run_batch`**

Before `budget.can_start()`, check the precondition against the ratios of finished slots;
on failure append a `SlotResult(outcome="precondition_failed", error=...)` carrying the
reason and the observed ratio, and `continue` — do **not** break, because a later slot may
have a different precondition or none.

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `uv run pytest src/deltaforge/batch_test.py src/deltaforge/batch_run_test.py -q`

- [ ] **Step 6: Make `score_predictions` ignore a slot that never ran**

A precondition-skipped slot tested no prediction. Counting it wrong understates the record
exactly as counting it right would flatter it — the same rule `error` and `not_run` already
follow. Add a test asserting `scored` excludes it.

- [ ] **Step 7: Commit**

```bash
git add src/deltaforge/batch.py src/deltaforge/batch_test.py \
        src/deltaforge/batch_run.py src/deltaforge/batch_run_test.py
git commit -m "Let a batch decline a slot whose answer an earlier slot already gave"
```

---

### Task 3: Fix the three gate defects

Six of batch 003's seven slots came back `incorrect` and not one was a kernel bug.

**Files:**
- Modify: `src/deltaforge/batch.py`
- Modify: `src/deltaforge/batch_test.py`
- Modify: `src/deltaforge/harness/correctness.py`
- Modify: `src/deltaforge/harness/correctness_test.py`

**Interfaces:**
- Consumes: `Hypothesis.correctness`, `top1_threshold`, `kl_threshold` (already exist).
- Produces: `DistributionCheck.top1_interval: tuple[float, float]`; `Hypothesis` validation
  that refuses `correctness="exact"` for any non-identity hypothesis.

- [ ] **Step 1: Write the failing tests**

```python
# in src/deltaforge/batch_test.py
def test_only_the_identity_champion_may_be_gated_exactly():
    """009 was gated `exact` because it computes the same function as the reference. It does;
    it does not compute the same *bits*. fp32 accumulation in a different order from cuBLAS
    lands one bf16 ULP away, and one ULP flips an argmax on this model — it matched 1 of 5
    prompts. Bit-identity is a property of the implementation, and only installing nothing
    has it. This is the third time the project has paid for the distinction."""
    with pytest.raises(ValueError, match="only the identity champion is bit-identical"):
        _hypothesis(slug="015-gemv-bf16", kernels=("tiled_gemv_bf16",), correctness="exact")


def test_a_top1_bar_finer_than_the_sample_can_resolve_is_refused():
    """013 missed its bar by 0.000303 at n=264, where agreement quantises to 1/264 = 0.0038.
    A threshold an order of magnitude below one sample is not a decision procedure."""
    with pytest.raises(ValueError, match="finer than one sample"):
        _hypothesis(correctness="approximate", top1_threshold=0.97, kl_threshold=0.02,
                    correctness_positions=264)
```

```python
# in src/deltaforge/harness/correctness_test.py
def test_agreement_is_reported_with_the_interval_it_is_known_to():
    """8 flips in 264 is 3.0% [1.3%, 5.9%] at 95%. Reporting 0.9697 alone invites a reader
    to compare it against a bar it cannot be distinguished from."""
    result = check_distribution(model, model, [[1, 2, 3]], top1_threshold=0.9,
                                kl_threshold=1.0, max_new_tokens=5)
    low, high = result.top1_interval
    assert 0.0 <= low <= result.top1_agreement <= high <= 1.0


def test_kl_is_the_gate_and_agreement_is_reported_beside_it(model):
    """KL is continuous and has no resolution floor; agreement quantises to 1/n. Batch 003
    passed every KL bar by an order of magnitude and failed four slots on agreement."""
    perturbed = _slightly_perturbed(model)
    result = check_distribution(perturbed_pair(model, perturbed), top1_threshold=0.99,
                                kl_threshold=1.0, max_new_tokens=8)
    assert result.mean_kl <= result.kl_threshold
    assert result.top1_agreement < result.top1_threshold
    assert result.passed, "a slot inside its KL bar must not fail on a sub-resolution agreement bar"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/deltaforge/batch_test.py src/deltaforge/harness/correctness_test.py -k "exact or resolution or interval or beside" -v`
Expected: FAIL — the validations and the field do not exist.

- [ ] **Step 3: Implement**

In `Hypothesis.__post_init__`: refuse `correctness == "exact"` unless `is_identity`; add
`correctness_positions: int | None` and refuse a `top1_threshold` whose distance from any
achievable `k/n` is below `1/n`.

In `DistributionCheck`: add `top1_interval`, a 95% Wilson score interval —

```python
def _wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson rather than normal-approximation: at 8/264 the normal interval runs negative.

    This is the interval a reader needs beside an agreement figure. 8 flips in 264 is
    3.0% [1.3%, 5.9%], and batch 003 compared that against a 2% bar as though the two were
    distinguishable.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
```

and
make `passed` require the **KL** bar and require the top-1 bar only when it lies *outside*
the interval — so a sub-resolution miss cannot fail a slot. Record both in `to_dict()`.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `uv run pytest src/deltaforge -q`

- [ ] **Step 5: Re-derive batch 003's verdicts as a regression test**

```python
def test_batch_003s_slots_would_now_pass_on_their_measured_numbers():
    """The gates, not the kernels, produced five of batch 003's six `incorrect` verdicts.
    Feeding the recorded numbers back through the fixed gates must show that."""
    for agreement, kl, n in ((256/264, 0.001138, 264), (256/264, 0.000820, 264),
                             (255/264, 0.001096, 264), (256/264, 0.001216, 264)):
        check = DistributionCheck(top1_agreement=agreement, mean_kl=kl, max_kl=0.02,
                                  num_positions=n, top1_threshold=0.98, kl_threshold=0.01)
        assert check.passed
```

- [ ] **Step 6: Fix the layer-1 reference dtype mismatch**

`quantised_linear.int8_dequant_correctness_checks` compares `Int8DequantLinear` — which
rounds the dequantised weight to bf16 — against an fp32 reference. Give it its own reference
that rounds the same way, and leave the fp32 reference to the Triton kernels. Assert in a
test that the two references differ, so the distinction cannot be silently collapsed again.

- [ ] **Step 7: Commit**

```bash
git add src/deltaforge/batch.py src/deltaforge/batch_test.py \
        src/deltaforge/harness/correctness.py src/deltaforge/harness/correctness_test.py \
        src/deltaforge/kernels/quantised_linear.py src/deltaforge/kernels/quantised_linear_test.py
git commit -m "Gate on the statistic that can resolve the claim, and stop calling approximate kernels wrong"
```

---

### Task 4: The tiled GEMV — bf16 first, and nothing else until it holds

**Files:**
- Create: `src/deltaforge/kernels/tiled_gemv.py`
- Create: `src/deltaforge/kernels/tiled_gemv_test.py`
- Modify: `src/deltaforge/kernels/__init__.py`
- Modify: `src/deltaforge/model.py`

**Interfaces:**
- Consumes: `_triton.HAS_TRITON`, `require_cuda`, `harness.correctness.check_kernel`.
- Produces: `tiled_gemv_bf16(x, w_k_major) -> Tensor` (custom op `deltaforge::tiled_gemv_bf16`),
  `to_k_major(weight) -> Tensor`, `install_tiled_bf16(model, entry)`,
  `TiledBf16Linear`, `tiled_bf16_correctness_checks`.

**Three changes, and each has a reason batch 003 measured:**

1. **K-major weight storage.** Store `W` as `[K, N]`, transposed once at install. A GEMV
   program owns output columns `[n0, n0+BLOCK_N)`; in `[K, N]` the slice it needs at each `k`
   is contiguous, so consecutive threads read consecutive addresses. It is also the layout
   `tl.dot` wants, with no transpose in the loop.
2. **`tl.dot` with M padded to 16.** The MMA accumulator carries the partial sums across
   K-iterations, so there is **no cross-lane reduction in the loop at all** — that is the 20-
   to 72-times-per-output cost that took batch 003 from 1171 to 332 GB/s. Put `x` in row 0
   and zeros in rows 1-15; `tl.sum(acc, axis=0)` then extracts the answer with a single
   reduction at the end. The 16× flop waste is free: arithmetic intensity here is ~2
   flop/byte against a machine balance near 150.
3. **Split-K.** `in_proj_a` and `in_proj_b` are 32 output channels wide and cannot fill the
   card by output channel at any tile size. Partials go to a `[SPLIT_K, M, N]` fp32 buffer
   and a second kernel reduces them — deterministic, unlike `atomic_add`.

- [ ] **Step 1: Write the failing test for the layout transform**

```python
def test_k_major_storage_is_the_transpose_and_is_contiguous():
    weight = torch.randn(2560, 9216)  # [N, K] as nn.Linear stores it

    k_major = to_k_major(weight)

    assert k_major.shape == (9216, 2560)
    assert k_major.is_contiguous()
    assert torch.equal(k_major, weight.t().contiguous())
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest src/deltaforge/kernels/tiled_gemv_test.py -k k_major -v`
Expected: FAIL, `ImportError: cannot import name 'to_k_major'`

- [ ] **Step 3: Implement `to_k_major` and the launch geometry**

```python
def to_k_major(weight: Tensor) -> Tensor:
    """`[N, K]` -> a contiguous `[K, N]`. Done once at install; free at decode time."""
    return weight.detach().t().contiguous()


def _launch_shape(n: int, k: int) -> tuple[int, int, int, int]:
    """`(BLOCK_N, BLOCK_K, SPLIT_K, num_warps)`.

    `tl.dot` needs BLOCK_N >= 16 and BLOCK_K >= 16, so unlike batch 003's kernel the tile
    cannot be narrowed to buy parallelism. Split-K buys it instead, which is the right
    instrument anyway: `in_proj_a` is 32 wide and no tile choice reaches a full card.
    """
    block_n = 32 if n <= 4096 else 64
    block_k = 64
    programs = -(-n // block_n)
    # Never split further than there are K-blocks to split: a program with no work still
    # costs a partial buffer row and a pass over it in the reduction kernel.
    split_k = max(1, min(8, -(-256 // max(programs, 1)), k // block_k))
    return block_n, block_k, split_k, 4
```

- [ ] **Step 4: Write the CPU emulation test for the kernel's indexing**

Triton is absent locally, but the indexing is executable and is where this kernel is most
likely to be wrong — split-K partial boundaries and the row-0 extraction. Follow the kernel
body statement for statement, as `quantised_linear_test.py` does, and include a mutation test
that proves the emulation can fail.

```python
def _emulate(x, w_k_major, block_n, block_k, split_k):
    k, n = w_k_major.shape
    partials = torch.zeros(split_k, x.shape[0], n, dtype=torch.float32)
    chunk = -(-k // split_k)
    for pid_k in range(split_k):
        lo, hi = pid_k * chunk, min((pid_k + 1) * chunk, k)
        for n0 in range(0, n, block_n):
            cols = torch.arange(n0, min(n0 + block_n, n))
            acc = torch.zeros(16, len(cols), dtype=torch.float32)
            for k0 in range(lo, hi, block_k):
                rows = torch.arange(k0, min(k0 + block_k, hi))
                x16 = torch.zeros(16, len(rows), dtype=torch.float32)
                x16[0] = x[0, rows].float()
                acc += x16 @ w_k_major[rows][:, cols].float()     # tl.dot accumulate
            partials[pid_k, :, cols] = acc.sum(dim=0)             # one reduction, row 0
    return partials.sum(dim=0)


@pytest.mark.parametrize(("n", "k", "split_k"), [(64, 256, 1), (64, 256, 4), (48, 192, 3)])
def test_the_split_k_emulation_reproduces_the_dense_product(n, k, split_k):
    torch.manual_seed(0)
    weight = torch.randn(n, k)
    x = torch.randn(1, k)

    emulated = _emulate(x, to_k_major(weight), block_n=32, block_k=64, split_k=split_k)
    dense = torch.nn.functional.linear(x.float(), weight.float())

    assert torch.allclose(emulated, dense, rtol=1e-5, atol=1e-4)


def test_a_split_k_that_drops_the_tail_chunk_is_caught():
    """The emulation only proves something if it can fail. This is the mutation: a `hi` that
    uses the chunk size rather than clamping to K silently loses the last partial rows."""
    torch.manual_seed(0)
    weight, x = torch.randn(32, 200), torch.randn(1, 200)
    w = to_k_major(weight)
    chunk = -(-200 // 3)
    wrong = sum(
        (x[:, p * chunk:(p + 1) * chunk].float() @ w[p * chunk:(p + 1) * chunk].float())
        for p in range(2)  # the bug: two chunks instead of three
    )
    assert not torch.allclose(wrong, torch.nn.functional.linear(x.float(), weight.float()),
                              rtol=1e-5, atol=1e-4)
```

- [ ] **Step 5: Run the emulation tests and make them pass**

Run: `uv run pytest src/deltaforge/kernels/tiled_gemv_test.py -q`

- [ ] **Step 6: Write the Triton kernel and the custom op**

```python
@triton.jit
def _tiled_gemv_bf16_kernel(X, W, PARTIALS, M, N, K, stride_wk, stride_pk, stride_pm,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n, pid_k, pid_m = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)

    chunk = tl.cdiv(K, SPLIT_K)
    k_lo, k_hi = pid_k * chunk, tl.minimum((pid_k + 1) * chunk, K)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_hi
        xv = tl.load(X + pid_m * K + offs_k, mask=mask_k, other=0.0)
        # x in row 0, zeros elsewhere: the MMA then carries the partial sums and the only
        # cross-lane reduction in this kernel is the tl.sum after the loop.
        xt = tl.where(rows[:, None] == 0, xv[None, :], 0.0).to(X.dtype.element_ty)
        w = tl.load(W + offs_k[:, None] * stride_wk + offs_n[None, :],
                    mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(xt, w, acc)

    tl.store(PARTIALS + pid_k * stride_pk + pid_m * stride_pm + offs_n,
             tl.sum(acc, axis=0), mask=offs_n < N)
```

A second `_reduce_partials_kernel` sums `[SPLIT_K, M, N]` fp32 down to `[M, N]` and casts to
the output dtype. Wrap both in `@torch.library.custom_op("deltaforge::tiled_gemv_bf16", ...)`
with a `register_fake`, mirroring `quantised_linear.bf16_gemv` — including the
`GEMV_MAX_ROWS` dense fallback for the untimed prefill, and the module docstring explaining
why that is not the silent fallback `require_cuda` forbids.

- [ ] **Step 7: Register the kernel, its installer and its checks**

`TiledBf16Linear(nn.Linear)` holding a `w_k_major` **buffer** and using `self.weight` for
nothing at decode time. Register in `kernels/__init__.py` as `tiled_gemv_bf16` replacing
`decode_step`; installer in `model.py`; check builder `tiled_bf16_correctness_checks`, reusing
`quantised_linear._probe_weights(model, sites="layers")` so every launch branch is probed.
The name is prefixed because `quantised_linear` already exports `bf16_correctness_checks` for
a different kernel, and `CHECK_BUILDERS` keys by kernel name for exactly this reason.

- [ ] **Step 8: Run the whole suite and the linters**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
```

- [ ] **Step 9: Commit**

```bash
git add src/deltaforge/kernels/tiled_gemv.py src/deltaforge/kernels/tiled_gemv_test.py \
        src/deltaforge/kernels/__init__.py src/deltaforge/model.py
git commit -m "A GEMV whose inner loop has no cross-lane reduction in it"
```

---

### Task 5: fp8 and int8 on the tiled kernel

**Files:**
- Modify: `src/deltaforge/kernels/tiled_gemv.py`
- Modify: `src/deltaforge/kernels/tiled_gemv_test.py`
- Modify: `src/deltaforge/kernels/__init__.py`, `src/deltaforge/model.py`
- Modify: `src/deltaforge/batch.py` (add `Hypothesis.weight_bits`)

**Interfaces:**
- Consumes: `to_k_major`, `_launch_shape` from Task 4.
- Produces: `quantise_fp8_per_channel(weight) -> QuantisedWeight`,
  `tiled_gemv_fp8(x, w_k_major, scale)`, `tiled_gemv_int8(x, w_k_major, scale)`,
  `Hypothesis.weight_bits: dict[str, int]` feeding Task 1's byte model.

**Why fp8 before int8 this time.** Batch 003 measured the conversion tax precisely: int8 cost
**1.438× the time of bf16** in the same kernel on the same sites, because `int8 -> fp32` is an
ALU instruction on the critical path. On sm_120, e4m3 converts in the MMA pipeline. Both go in
the batch, adjacent, so the comparison isolates exactly that.

- [ ] **Step 1: Write the failing test for fp8 quantisation**

```python
def test_fp8_e4m3_round_trip_keeps_the_per_channel_scale_meaningful():
    weight = torch.randn(16, 256)
    weight[3] *= 100.0

    q = quantise_fp8_per_channel(weight)

    assert q.qweight.dtype is torch.float8_e4m3fn
    assert q.scale.shape == (16,)
    restored = q.qweight.to(torch.float32) * q.scale.unsqueeze(1)
    # e4m3 carries 3 mantissa bits: ~6% worst-case relative error per element.
    rel = ((restored - weight).abs() / weight.abs().clamp(min=1e-6)).max()
    assert rel < 0.07


def test_fp8_is_coarser_than_int8_and_finer_than_int4():
    """The accuracy ordering the batch-004 gates are derived from, asserted rather than assumed."""
    weight = torch.randn(32, 512)
    err = lambda r: (r - weight).abs().mean()
    assert err(_restore_int8(weight)) < err(_restore_fp8(weight)) < err(_restore_int4(weight))
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/deltaforge/kernels/tiled_gemv_test.py -k fp8 -v`
Expected: FAIL, `ImportError: cannot import name 'quantise_fp8_per_channel'`

- [ ] **Step 3: Implement the quantisers and the kernel variants**

Per-channel symmetric, scale = `absmax / 448.0` for e4m3 (448 is e4m3's max finite value),
chunked over rows exactly as `quantised_linear` does — the tied head is 248320×2560 and a
2.5 GB fp32 temporary would OOM the gate. The kernel differs from Task 4's in one line:

```python
w = tl.load(...).to(tl.bfloat16)     # fp8 -> bf16, in the MMA pipeline on sm_120
acc = tl.dot(xt, w, acc)
```

- [ ] **Step 4: Add `Hypothesis.weight_bits` and feed Task 1's byte model**

```python
weight_bits: dict[str, int] = field(default_factory=dict)
```

Validate every key against `bytes_model.WEIGHT_REGIONS` in `__post_init__`, so a manifest typo
fails on a laptop rather than scoring a candidate against the wrong byte count on a rented box.

- [ ] **Step 5: Run the suite and the linters, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
git commit -am "fp8 and int8 on a kernel that can collect the saving"
```

---

### Task 6: Batch 004, with every prediction and precondition registered

**Files:**
- Modify: `src/deltaforge/batches.py`
- Modify: `src/deltaforge/batches_test.py`

| # | Slot | Sites | `weight_bits` | Predicted | Requires |
|---|---|---|---|---|---|
| 0 | `000-identity` | — | — | `identity` | — |
| 1 | `015-tiled-gemv-bf16` | layer projections | — | `inconclusive` (0.75-0.95) | — |
| 2 | `016-fp8-all-linear` | layer projections | `{"layers": 8}` | **`win`** (1.15-1.30) | `015 ≥ 0.56` |
| 3 | `017-fp8-full` | + tied head | `{"layers": 8, "head": 8}` | **`win`** (1.25-1.45) | `015 ≥ 0.56` |
| 4 | `018-fp8-mlp` | MLP only | `{"mlp": 8}` | `win` (1.08-1.18) | `015 ≥ 0.56` |
| 5 | `019-int8-all-linear` | layer projections | `{"layers": 8}` | `inconclusive` (1.05-1.25) | `015 ≥ 0.56` |
| 6 | `020-int4-full` | + tied head | `{"layers": 4, "head": 4}` | `win` (1.4-1.9) | `016 ≥ 1.05` |

Seven slots, which is also what `batches_test.test_a_batch_holds_seven_to_twelve_hypotheses_unless_it_is_calibrating`
requires of a non-calibration batch — `BATCH_004.is_calibration` stays `False`.

Slot 1 predicts `inconclusive` deliberately: it moves exactly cuBLAS's bytes, so tying is the
honest expectation and **its job is to be the divisor**, not to win.

Slots 4, 2 and 3 are a dose-response ladder at 49.5%, 77.9% and 91.8% of per-token bytes —
the same device that made batch 003 readable, where a larger share buying a larger win is
what distinguishes "the mechanism works" from "something else moved". Slot 5 is the direct
re-run of batch 003's `012`, same sites and same quantisation with only the kernel structure
changed: **0.1962 against whatever it now returns is the value of Task 4, isolated.** Slot 6's
precondition is on `016` rather than `015`, because int4's extra unpack is only worth trying
once fp8 has shown the pipeline is clear.

- [ ] **Step 1: Write the manifest tests first**

```python
def test_batch_004_gates_every_quantised_slot_on_the_bf16_control():
    """Batch 003 spent five slots on variants its first slot had already settled."""
    for hyp in BATCH_004:
        if hyp.weight_bits:
            assert hyp.requires is not None, f"{hyp.slug!r} would run regardless of the control"
            assert hyp.requires.floor >= 0.56


def test_the_bf16_control_is_not_predicted_to_win():
    control = BATCH_004.get("015-tiled-gemv-bf16")
    assert control.prediction == "inconclusive"
    assert control.weight_bits == {}


def test_every_004_slot_is_gated_approximately_except_the_identity():
    for hyp in BATCH_004:
        assert (hyp.correctness == "exact") == hyp.is_identity


def test_004_byte_shares_agree_with_the_byte_model():
    """A manifest's byte_share and its weight_bits must describe the same candidate.

    `byte_share` in this repo is the share of per-token bytes a hypothesis *attacks*, not the
    share it saves — 012 carried 0.779 for the layer projections it quantised, not the 0.390
    it removed. Deriving it from the same arithmetic the bench now uses stops the two drifting.
    """
    from .harness.bytes_model import decode_bytes_per_token
    config = qwen3_5_4b_config()
    full = decode_bytes_per_token(config, weight_bits={}, context_length=2048)
    for hyp in BATCH_004:
        if not hyp.weight_bits:
            continue
        # Zero-width stand-in: bits=0 removes the attacked regions entirely, so the
        # difference from bf16 is exactly the bytes those regions contribute.
        without = decode_bytes_per_token(
            config, weight_bits=dict.fromkeys(hyp.weight_bits, 0), context_length=2048
        )
        assert abs(hyp.byte_share - (full - without) / full) < 0.01
```

- [ ] **Step 2: Run them and watch them fail; then write `BATCH_004`**

Each hypothesis carries a mechanism, a rationale over 80 characters, a prediction registered
before the rental, `weight_bits`, and thresholds **derived from batch 003's measurements**:
int8 measured 0.001138 nats and 8/264 flips, int4 measured 0.091854 and 38/264. Set fp8's KL
bar at 0.03 — between them, nearer int8 — and state in the rationale that it is interpolated
from two measured points rather than taken from priors, which is how batch 003's bars went
wrong.

- [ ] **Step 3: Dry-run the money machinery, which spends nothing**

```bash
remote/run_remote.sh --dry-run --session-id smoke --batch 004-bandwidth-bound-gemv
```

- [ ] **Step 4: Commit before renting**

```bash
git commit -am "Register batch 004's predictions and preconditions, before the rental"
```

---

### Task 7: Dump what the compiler emits, inside the rental

`docs/HYPOTHESES.md` says in bold that reading inductor's generated code "is not optional".
Batch 003 skipped it and came back with `010` at 0.9893 — a 39% cut in DRAM traffic for 0% of
time — whose mechanism is still unnamed.

**Files:**
- Modify: `remote/run_remote.sh`
- Modify: `remote/scripts_test.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_run_dumps_inductor_output_code_before_the_batch():
    """Not a nicety. The one open question from rental 37 is what inductor emits for a
    quantised linear, and it costs a step rather than a rental."""
    script = (REPO / "remote/run_remote.sh").read_text()
    assert "TORCH_LOGS=output_code" in script
    assert script.index("TORCH_LOGS=output_code") < script.index("cli batch")


def test_the_dump_is_pulled_home_with_the_results():
    script = (REPO / "remote/run_remote.sh").read_text()
    assert "inductor-output-code" in script
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest remote/scripts_test.py -k output_code -v`

- [ ] **Step 3: Add the step**

A short `TORCH_LOGS=output_code python -m deltaforge.cli bench --rounds 1 --warmup-rounds 0`
run over one decoder layer, teed to `/workspace/deltaforge/results/diagnostics/
inductor-output-code.txt`, which the existing results rsync already brings home. Guard it with
`|| true`: a diagnostic that fails must not take the rental with it.

- [ ] **Step 4: Run the tests, dry-run, commit**

```bash
uv run pytest remote/ -q
remote/run_remote.sh --dry-run --session-id smoke --batch 004-bandwidth-bound-gemv
git commit -am "Dump the code we are trying to beat, in the rental that tries to beat it"
```

---

### Task 8: Run it, and stop at the honest point

- [ ] **Step 1: Confirm the tree is committed and green**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && git status --short
```

- [ ] **Step 2: Launch through `launch.sh`, never a hand-rolled `nohup`**

```bash
SESSION="batch004-$(date -u +%Y%m%dT%H%M%SZ)"
LOG="/tmp/$SESSION.log"
remote/launch.sh --log "$LOG" --session-id "$SESSION" --batch 004-bandwidth-bound-gemv
tail -n +1 -F "$LOG"          # +1, never -n 0
```

`--status` plus a live instance is what says a rental is up; "provisioning..." prints before
the offer search.

- [ ] **Step 3: Read slot 1 before anything else**

`015-tiled-gemv-bf16`'s achieved GB/s is the result of Tasks 4-5, whatever the ratios behind
it say:

| Slot 1 reports | What it means | What to do |
|---|---|---|
| ≥ 820 GB/s (ratio ≥ 0.75) | memory-bound; quantisation should win | let the batch run |
| 585-820 GB/s (0.56-0.75) | memory-bound enough to tie or win narrowly | let it run, expect a thin margin |
| < 585 GB/s (< 0.56) | still issue-bound | the preconditions fire; **do not** loosen them |

If the preconditions fire, the rental ends having answered the only question that mattered,
at four slots' cost rather than seven. That is the batch working, not the batch failing.

- [ ] **Step 4: Check `graphs_compiled` and the identity slot before believing any number**

`calibrated: true`, and a non-zero `graphs_compiled` on every slot. A `0` means that ratio is
not a comparison — blocker 16's signature.

- [ ] **Step 5: Write the batch up, and update the docs in the same pass**

`results/batches/004-bandwidth-bound-gemv/README.md`, then `LEADERBOARD.md`,
`docs/HYPOTHESES.md`, `docs/BATCHES.md`, `docs/GPU-ACCESS.md`, `AGENT.md` §8, `README.md` —
`AGENT.md` §6.1 says when each is worth touching. **A claim this batch disproves gets fixed
in the same pass**, including anything in this plan.

- [ ] **Step 6: Promote, or do not**

Promotion requires the median ratio to exceed 1.0 by **more than the IQR of the scoring
rounds**. Batch 003's IQRs were 0.0026-0.0254, so a 1.15× would clear comfortably and a 1.01×
would not. A margin inside the noise band is `inconclusive`: it neither promotes nor
graveyards. Recording a noise-band result as a win is how a leaderboard becomes fiction.

---

## What would falsify this plan

Stated in advance, so that a null result is readable rather than re-litigated:

- **Slot 1 lands below 0.56 after the rewrite.** Then the problem is not the reduction
  structure, and the remaining suspects are the `tl.dot` row-0 extraction, occupancy, or a
  split-K reduction that costs more than it buys. The `output_code` dump from Task 7 and one
  profile would be the next step — not another quantisation batch.
- **Slot 1 clears 0.75 and the fp8 slots still lose.** Then the conversion is not free on this
  card even in the MMA path, and the honest conclusion is that weight-only quantisation is not
  reachable in Triton on sm_120 — at which point the comparison against Marlin/machete that
  `docs/HYPOTHESES.md` has been asking for becomes the priority, as an unscored column.
- **`018-int8-all-linear` returns ~0.1962 again.** Then Task 4 changed nothing measurable and
  the emulation tests were testing the wrong property.
- **Everything wins but the margins sit inside the IQR.** Then the batch bought a direction
  and not a champion, and the next rental is `--rounds` raised, not a new kernel.

## Cost

Seven slots on a warm compile cache, from rental 37's measured 293-428 s per slot plus ~19
minutes fixed: **roughly 55-70 minutes, $0.40-0.50**, inside the 180-minute session gate with
a wide margin. Preconditions firing make it cheaper, not more expensive. Month-to-date is
$6.157 against a $45 gate.
