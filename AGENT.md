# Start here

**This is the only file you need to read before starting.** Everything else is linked from
it, and nothing else is required reading until you need the detail it holds.

---

## 1. What this project is trying to prove

> A hand-written Triton kernel can beat what `torch.compile(mode="max-autotune")`
> generates, on the decode path of an open-weights LLM — and we can say **where**, **by how
> much**, and **why**, in advance.

The target is `Qwen/Qwen3.5-4B`, benchmarked at batch-1 decode. Every session rents one GPU
for under an hour, tests one hypothesis, records the result win or lose, and destroys the
instance.

**The last clause is the whole project.** "My kernel beat the compiler" is the premise the
entire inference-serving industry is built on — vLLM, SGLang and TensorRT-LLM are all
hand-written kernels — so re-proving it is not a finding. The finding is a correct,
mechanistic account of *where the compiler wins and where it cannot*, registered before the
measurement and then confirmed by it.

So: **state your prediction and your mechanism before you write code.** Being right in
advance is the result. A hypothesis you predicted would fail, that failed for the reason you
gave, is worth more than an unexplained win.

## 2. The arithmetic that decides whether your hypothesis is worth trying

Run this first. It needs no GPU, no checkpoint, and no money:

```sh
uv run python docs/roofline.py
```

It prints where every byte goes in one decode step. The headline result:

| What moves | share of per-token bytes |
|---|---:|
| Weights, streamed once | **91.85%** |
| GQA `repeat_interleave` materialisation | **6.23%** |
| Recurrent state | 1.10% |
| KV cache read | 0.78% |
| SwiGLU intermediates | 0.026% |
| Norm + residual | 0.018% |

**A hypothesis cannot beat the share of bytes it touches.** That is the single most useful
sentence in this repo. A perfect, infinitely fast fused RMSNorm buys 0.018% — below the
harness's own noise band, so the hypothesis is not unlikely, it is *unmeasurable*. Five
hypotheses are in the graveyard for exactly this reason, closed by arithmetic rather than by
renting a GPU.

Batch-1 decode is a **weight-streaming** problem, not a kernel problem. Once a kernel
streams its data once it is at the roofline, and inductor is good at reaching the roofline
on simple memory-bound work — it emits Triton, so you would be hand-writing the kernel it
already generates. You beat it by **moving fewer bytes** (quantisation, layout, removing a
materialised intermediate) or by **running a different algorithm** (chunked scans,
online-softmax attention, split-K) — never by writing a tighter elementwise loop.

`docs/HYPOTHESES.md` ranks the open backlog by this measure and records why each graveyard
entry is dead.

## 3. Read what you are trying to beat

```sh
TORCH_LOGS=output_code python -m deltaforge.cli bench --weights … 2>&1 | tee inductor.txt
```

This prints the exact Triton inductor generated. **Do this before writing a kernel.** You
cannot claim to beat code you have not read, and you cannot explain why you won without it.
Also confirm the baseline actually compiled — a silent graph break drags `compiled` toward
eager and inflates every ratio in your favour, which is the first thing a sceptical reader
checks.

## 4. A session, start to finish

1. **Read `LEADERBOARD.md`** — current champion and every attempt so far.
2. **Read the graveyard in `docs/HYPOTHESES.md`.** Do not re-run a dead end.
3. **Pick one hypothesis.** State in one sentence: the mechanism, which of the three
   win-categories it uses (see `docs/HYPOTHESES.md`), and its share of per-token bytes.
   "Fuse it and see" is not a hypothesis.
4. **Branch `hyp/NNN-slug`.**
5. **Write and test the kernel locally.** No GPU needed — GPU time buys validation and
   measurement, not development.
   ```sh
   uv sync --extra dev
   uv run pytest
   uv run ruff check . && uv run ruff format --check .
   ```
   Put it in `src/deltaforge/kernels/`, register it, and add its installer to
   `deltaforge.model.INSTALLERS` **in the same commit**. A champion with no installer is a
   hard error, never a fallback: silently benchmarking the baseline while labelling it the
   candidate is the worst failure this harness could have.
6. **Dry-run the money machinery**, which spends nothing:
   ```sh
   remote/run_remote.sh --dry-run --session-id smoke
   ```
7. **Run it.** `remote/run_remote.sh` is the only entry point — it provisions, syncs, runs
   the GPU tests and both correctness gates, benchmarks, pulls results back and destroys the
   instance, with the whole body inside a trap on `EXIT`/`INT`/`TERM` so a crash still tears
   down.
   ```sh
   remote/run_remote.sh --session-id "$SESSION" --hypothesis "NNN-slug"
   ```
8. **Record the outcome** (section 6). Win or lose.

The benchmark runs four columns by default. `compiled` against `candidate_compiled` is the
score — identical `max-autotune` treatment, the only difference being who wrote the kernel.
`eager` and `candidate` are free diagnostics. `compiled_nocudagraphs` costs a whole extra
compilation and is **opt-in** via `--columns all`: run it on a calibration run, or when a
hypothesis is about launch overhead. Whichever set runs is recorded in the result.

### If this is the first session that ever gets a GPU

**No measurement exists yet.** Do not write a kernel. Spend the session on:

1. `pytest -m "gpu and weights"` — the weight-value oracle. Until it passes, the reference
   is proven structurally correct but not proven to interpret weight *values* correctly, and
   every number downstream of it is measuring an unvalidated model.
2. An **identity champion** — an installer that changes nothing. Run the full harness with
   it. Every column must come back at 1.00 ± noise. That is how you calibrate the harness,
   with zero kernel-writing risk. (The previous attempt used a real kernel for this and
   spent its whole budget on the least interesting one in the backlog.)
3. A `torch.profiler` per-kernel breakdown of batch-1 decode, saved to
   `results/baseline/`.

That artifact — "here is where decode actually spends its time" — is worth more than any
kernel, and everything after it is better aimed.

## 5. Money and safety

**At 60 cumulative billed minutes, this session may not start another hypothesis.**
`run_remote.sh` checks before each run and exits 3 when the session is spent. When that
happens: destroy any live instance, then finish recording and writing up the work you
already did — none of which needs a GPU — and stop. Do not start "just one more attempt".

The check happens before a run, never during one: a benchmark executing at minute 59
finishes normally, because killing it would waste the money already spent and leave nothing
recorded in exchange. The 90-minute watchdog is a backstop for hangs — **if it ever fires,
that is a fault and the writeup must say so.**

`VAST_API_KEY` lives in a gitignored `.env` at the repo root and reaches curl through a
config file on stdin, so it never enters argv, a file, or a log. **Never** print it, commit
it, or put it in a PR body. CI greps tracked files for it. The same applies to the optional
`DOCKER_LOGIN_USER` / `DOCKER_LOGIN_TOKEN`; request bodies go to curl through a `0600` temp
file rather than `argv`, and `remote/scripts_test.py` asserts that.

**If a run hangs before sshd answers, read `docs/GPU-ACCESS.md` before renting again.**
Nine rentals have been billed on this project and none produced a number. Eight died on an
anonymous Docker Hub pull; the ninth proved a `ghcr.io` image pulls fine and then exposed a
second blocker behind it. Both are fixed; that file records how, and what is still
unproven.

## 6. Recording the outcome — the part that matters

Sessions are only worth anything if they compound. **A loss is recorded as carefully as a
win**, with the mechanism that failed and why — not a restatement of the result, the
*cause*.

1. `results/hypotheses/NNN-slug.json` — written by the harness, win or lose.
2. `LEADERBOARD.md` — a row with the ratio and outcome.
3. `docs/HYPOTHESES.md` — a graveyard entry if it lost, or an updated open entry if the
   result was inconclusive.
4. Confirm `ledger/spend.jsonl` reflects the run.

**Promotion is not "it was faster."** Promote to champion only if the candidate's median
ratio exceeds the incumbent's by more than the interquartile spread of the scoring rounds
(`BenchResult.iqr_ratio`). A margin inside the noise band is **inconclusive**: it neither
promotes nor graveyards, and the hypothesis stays open for a cleaner measurement. Recording
a noise-band result as a win is how a leaderboard becomes fiction.

**Never carry a number across sessions.** Every session rents different physical hardware,
so the score is a ratio measured inside one process on one card, and the reigning champion
is re-benchmarked every time.

**Do not fabricate, estimate, or placeholder any number.** If a run produced no measurement,
the record says so.

## 7. Where things are

| Path | What it is |
|---|---|
| `docs/roofline.py` | **Run first.** Where the bytes go; the ceiling on any hypothesis. |
| `docs/HYPOTHESES.md` | The ranked backlog and the graveyard, with mechanisms. |
| `docs/ARCHITECTURE.md` | Resolved model facts. Read before writing any kernel. |
| `LEADERBOARD.md` | Champion and every attempt. |
| `src/deltaforge/reference.py` | **The baseline.** Never contains a custom kernel. |
| `src/deltaforge/config.py` | Model configs and the `tiny_config()` CPU test fixture. |
| `src/deltaforge/weights.py` | Checkpoint → model; explicit name map, loud on anything unmapped. |
| `src/deltaforge/model.py` | Assembly, champion installation, greedy decode. |
| `src/deltaforge/kernels/` | Kernels and the registry. |
| `src/deltaforge/harness/bench.py` | Interleaved A/B/A timing. Knows nothing about Qwen or Triton. |
| `src/deltaforge/harness/correctness.py` | Both gates. Reports magnitudes, never a bare bool. |
| `src/deltaforge/ledger.py` | Spend record and both budget gates, in Python. |
| `remote/run_remote.sh` | The one entry point for a GPU run. |
| `docs/GPU-ACCESS.md` | **Read if a run hangs before sshd.** The container-pull blocker and its fix. |
| `remote/lib.sh` | The same two gates in awk, plus the Vast API and lifecycle helpers. |
| `docs/superpowers/specs/2026-08-29-deltaforge-design.md` | The original design spec. Historical. |

Tests are colocated as `<module>_test.py`; `testpaths` is `src` and `remote`.

## 8. Things that will bite you

**Never modify `reference.py` to accommodate a kernel.** It is the definition of the
baseline; changing it invalidates every stored result. `reference_purity_test.py` enforces
by AST inspection that it contains no Triton, no FlashAttention and no
`F.scaled_dot_product_attention` — SDPA *is* one of the kernels under test. Kernels attach
through `deltaforge.model.INSTALLERS`, never by editing the reference. (Fixing the reference
because it *models the checkpoint wrongly* is a different thing and is correct to do.)

Model-level traps, all asserted by tests so CI catches them before the GPU does:

- **`head_dim` is 256, not `hidden_size / num_heads` (160).** The projections are not
  square. Most likely early bug.
- **RMSNorm scales by `1 + weight`** — the checkpoint stores zero-centred norm weights.
  Reading it as plain `weight` gives an all-zero activation and a model that still runs.
- **The gated RMSNorm inside Gated DeltaNet is the opposite**: plain `weight`, and it
  normalises *before* gating.
- **The attention output gate is config-driven.** Qwen3.5 omits `output_gate_type` and means
  sigmoid; Qwen3.8 declares `swish`. Hardcoding either gives a model that runs and emits
  plausible logits on the other checkpoint.
- **RoPE is partial** (64 of 256 dims) and **mRoPE-interleaved**.
- **The recurrent state is fp32** even though weights are bf16.
- **`q_proj` is doubled** for the output gate, and the split is per head, not a split of the
  flat projection.
- **The Gated DeltaNet projection layout differs from Qwen3-Next**: four separate
  projections, no head interleaving to undo.

## 9. Environments — the one thing that surprises people

**Local and CI (CPU).** `pyproject.toml` pins torch to PyTorch's **CPU index** via
`[tool.uv.sources]`, because the acceptance bar is that `uv run pytest` passes on a clean
CPU-only checkout and the default PyPI wheel drags ~3 GB of `nvidia-*` packages in.

**The rented box (GPU).** `run_remote.sh` installs torch from PyTorch's CUDA CDN and then
`pip install --no-deps -e .` on top, so the CPU pin can never downgrade the box.

`uv.lock` is gitignored: one lock file cannot serve both environments.

Anything needing a GPU or the checkpoint carries the `gpu` and `weights` pytest markers and
**skips on a real condition** — no CUDA, or no checkpoint — rather than being faked, so it
starts running by itself once both are present.

## 10. Maintaining this file

This is project memory, not a changelog. Add something only when it is durable knowledge
almost every future session needs and the code does not already show for itself. Prefer a
pointer to the authoritative file, command or test over copying a detail that will drift. If
something here turns out to be wrong, fix it in the same pass as the work that revealed it.
