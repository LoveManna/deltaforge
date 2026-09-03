# What a new working session reads first

You are picking up a project that runs as a series of independent sessions on different
rented GPUs. The whole value of the repo is that sessions **compound** instead of
rediscovering each other's dead ends. That only works if you record what happened.

**The standing rule: a loss is recorded as carefully as a win.** A hypothesis that failed,
written down with the mechanism that failed and why, is worth more to the next session
than a vague win. Visible failures are what make the log credible.

---

## Before you write any code

1. **Read `LEADERBOARD.md`.** Current champion, and every hypothesis already attempted.
2. **Read the graveyard in `docs/HYPOTHESES.md`.** Do not re-run a dead end.
3. **Pick one hypothesis**, from the backlog or new. **State it in one sentence before
   writing any code**, including the mechanism you expect to produce the win. "Fuse it and
   see" is not a hypothesis and produces a result nobody can learn from.
4. **Branch `hyp/NNN-slug`.**

## Writing the kernel — no GPU needed

Kernels are written and tested locally. GPU time buys validation and measurement, not
development. Set up with:

```sh
uv sync --extra dev     # resolves CPU torch; see "Environments" below
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Add the kernel under `src/deltaforge/kernels/`, register it in the registry, and add its
installer to `deltaforge.model.INSTALLERS` in the same commit. `apply_champions` refuses
to run if a champion has no installer — it will not quietly fall back to the reference,
because benchmarking the baseline while labelling it the candidate is the worst failure
this harness could have.

**Never modify `reference.py` to accommodate a kernel.** It is the definition of the
baseline; changing it invalidates every stored result. `reference_purity_test.py` enforces
that it contains no custom kernels, and it will fail your PR if you try.

## The session GPU-time gate — check it before you rent anything

**At 60 cumulative billed minutes, this session may not start another hypothesis.**

```sh
remote/run_remote.sh --session-id "$SESSION" --hypothesis "NNN-slug"
```

checks it for you and exits 3 if the session is spent. When that happens: destroy any live
instance, then finish recording, writing up and merging the work you already completed —
none of which needs a GPU — and end the session. Do not start "just one more attempt".

The check happens *before* a run starts and never during one. A benchmark executing at
minute 59 finishes normally; killing it halfway would waste the money already spent on it
and leave nothing recorded in exchange.

This is the gate expected to fire in ordinary operation. The 90-minute watchdog is a
backstop for hangs — **if the watchdog ever fires, that is a fault, and the session writeup
must say so** rather than treating it as a normal ending.

## Running on the GPU

`remote/run_remote.sh` is the only entry point. It provisions, syncs, runs the correctness
gates, benchmarks, pulls results back, and destroys the instance — with the whole body
inside a trap on `EXIT`/`INT`/`TERM`, so a crash still tears down.

Before you touch real money, prove the machinery still works:

```sh
remote/run_remote.sh --dry-run --session-id smoke
```

That exercises both budget gates, the offer filter, and the teardown trap without
contacting the create or destroy endpoints. `remote/scripts_test.py` runs the same paths in
CI.

`VAST_API_KEY` lives in a gitignored `.env` at the repo root. It is passed to curl through
a config file on stdin, so it never reaches argv, a file, or a log. **Never** print it,
commit it, or put it in a PR body.

## Recording the outcome — the part that matters

1. `results/hypotheses/NNN-slug.json` — written by the harness. **Win or lose.**
2. `LEADERBOARD.md` — a row for the hypothesis, with its ratio and outcome.
3. `docs/HYPOTHESES.md` — if it lost, a graveyard entry with the mechanism that failed and
   why. Not a restatement of the result: the *cause*.
4. Confirm `ledger/spend.jsonl` reflects the run.

### Promotion is not "it was faster"

Promote to champion only if the candidate's median ratio exceeds the incumbent's **by more
than the interquartile spread of the scoring rounds**. `BenchResult.iqr_ratio` gives you
that number.

A margin inside the noise band is recorded as **inconclusive**: it neither promotes nor
enters the graveyard, and the hypothesis stays open for a cleaner measurement. Recording a
noise-band result as a win is how a leaderboard becomes fiction.

### The reigning champion is re-benchmarked every session

Never carry a stored number forward as a comparison point. Every session runs on different
physical hardware, so a previous session's absolute milliseconds mean nothing here. That
is why the score is a ratio measured within a single process on a single card.

## Environments

**Local and CI (CPU).** `pyproject.toml` pins torch to PyTorch's CPU index via
`[tool.uv.sources]`, so `uv sync --extra dev` gives a working CPU environment without
dragging ~3 GB of `nvidia-*` wheels in. `uv run pytest` must pass on a machine with no
CUDA present.

**The rented box (GPU).** The image already ships torch 2.11 / CUDA 12.8 and a matching
Triton, so `run_remote.sh` installs with `pip install --no-deps -e .` on top of it. That
pin can never downgrade the box to CPU torch, and it avoids paying for a dependency
resolution on every provision.

`uv.lock` is deliberately **not** committed: the CPU and GPU environments differ
fundamentally, and a lock file pinned to one would fight the other.

## What is deferred, and how to tell

Anything needing a GPU or the 9 GB checkpoint is marked with the `gpu` and `weights`
pytest markers and **skips on a real condition** — no CUDA, or no checkpoint — rather than
being faked. It starts running by itself once both are present:

```sh
DELTAFORGE_WEIGHTS_DIR=/workspace/qwen3.5-4b uv run pytest -m "gpu and weights"
```

`src/deltaforge/oracle_test.py` is the main one: the full-weight correctness oracle against
HuggingFace `transformers`. Until it has run on real weights, the reference is proven
*structurally* correct (shapes, cache contract, causality, mRoPE reduction) but not proven
to interpret the weight **values** correctly. **The first funded session should run it
before anything else** — every benchmark number depends on it.

## Repository map

| Path | What it is |
|---|---|
| `src/deltaforge/reference.py` | The baseline. Never contains a custom kernel. |
| `src/deltaforge/config.py` | Model config, plus the `tiny_config()` CPU test fixture. |
| `src/deltaforge/weights.py` | Checkpoint → model, explicit name map, loud on anything unmapped. |
| `src/deltaforge/model.py` | Assembly, champion installation, greedy decode. |
| `src/deltaforge/kernels/` | The kernels and the registry. Currently empty of kernels. |
| `src/deltaforge/harness/bench.py` | Interleaved timing. Knows nothing about Qwen or Triton. |
| `src/deltaforge/harness/correctness.py` | Both gates. Reports magnitudes, never a bare bool. |
| `src/deltaforge/harness/report.py` | Results JSON and the markdown fragment. |
| `src/deltaforge/ledger.py` | Spend record and both budget gates, in Python. |
| `remote/lib.sh` | The same two gates in awk, plus the API and lifecycle helpers. |
| `docs/ARCHITECTURE.md` | The resolved model facts. Read before writing any kernel. |

Tests are colocated with the code they exercise, as `<module>_test.py`.

## Things that will bite you

- **`head_dim` is 256, not `hidden_size / num_heads` (160).** The projections are not
  square. Most likely early bug.
- **RMSNorm scales by `1 + weight`**, because the checkpoint stores zero-centred norm
  weights. Reading it as plain `weight` gives an all-zero activation and a model that
  still runs.
- **The gated RMSNorm inside Gated DeltaNet is the opposite**: plain `weight`, and it
  normalises *before* gating.
- **RoPE is partial** (64 of 256 dims) and **mRoPE-interleaved**.
- **The recurrent state is fp32** even though weights are bf16.
- **`q_proj` is doubled** for the output gate, and the split is per head, not a split of
  the flat projection.
- **The Gated DeltaNet projection layout differs from Qwen3-Next**: four separate
  projections, no head interleaving to undo.

All of these are asserted by tests. If you break one, CI tells you before the GPU does.

## Maintaining this file

This file is project memory, not a changelog. Add something only when it is durable
knowledge that almost every future session needs, and that the code does not already show
for itself. Prefer a pointer to the authoritative file, command or test over copying a
detail that will drift. If something here turns out to be wrong, fix it in the same pass
as the work that revealed it.
