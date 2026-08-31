# Project agent memory

**Read `AGENT.md` first.** It is the session workflow this project runs on — which
hypothesis to pick, the GPU-time gate, and the record-keeping duties — and it is named by
the design spec, so it stays where it is. This file holds the tooling notes that do not
belong in it.

## Commands

```sh
uv sync --extra dev                       # CPU-only env; see "Environments" below
uv run pytest                             # must pass with no CUDA present
uv run ruff check . && uv run ruff format --check .
remote/run_remote.sh --dry-run --session-id smoke   # cost machinery, spends nothing
```

Tests are colocated as `<module>_test.py` next to the code they exercise; `testpaths` is
`src` and `remote`.

## Environments — the one thing that surprises people

`pyproject.toml` pins torch to PyTorch's **CPU index** via `[tool.uv.sources]`. That is
deliberate: the acceptance bar is that `uv run pytest` works on a clean CPU-only checkout,
and the default PyPI wheel drags ~3 GB of `nvidia-*` packages in to get there.

The rented GPU box does **not** use that path. Its image already has torch 2.11 / CUDA
12.8, and `remote/run_remote.sh` installs on top with `pip install --no-deps -e .`, so the
CPU pin can never downgrade the box. `uv.lock` is gitignored for the same reason: one lock
file cannot serve both environments.

## Where the authority lives

| Question | File |
|---|---|
| Session workflow, GPU-time gate, recording duties | `AGENT.md` |
| Model architecture facts, and what is excluded | `docs/ARCHITECTURE.md` |
| What has been tried and what is dead | `docs/HYPOTHESES.md`, `LEADERBOARD.md` |
| Benchmark methodology and the baseline definition | `docs/superpowers/specs/2026-08-29-deltaforge-design.md` |
| Result and ledger record formats | `results/README.md`, `ledger/README.md` |

## Sharp edges

- **`reference.py` is the baseline and never gains a custom kernel.** Not Triton, not
  FlashAttention, not `F.scaled_dot_product_attention` — SDPA *is* one of the kernels
  under test. `reference_purity_test.py` enforces this by AST inspection and will fail the
  PR. Kernels attach through `deltaforge.model.INSTALLERS`, never by editing the reference.
- **A champion with no installer is a hard error, not a fallback.** `apply_champions`
  refuses to run rather than silently benchmark the baseline while labelling it the
  candidate.
- **Never carry a number across sessions.** Every session rents a different physical GPU,
  so the score is a ratio measured inside one process on one card. The champion is
  re-benchmarked every session.
- **A result inside the IQR is `inconclusive`, not a win.** See `AGENT.md` on promotion.
- **`VAST_API_KEY` lives in a gitignored `.env`** and reaches curl through a config file on
  stdin, so it never enters argv, a file, or a log. Never echo it, commit it, or put it in
  a PR body. CI greps tracked files for it.
- Model-level traps (head_dim 256 ≠ hidden/heads, RMSNorm scaling by `1 + weight`, partial
  RoPE, fp32 recurrent state, the doubled gated `q_proj`) are listed in `AGENT.md` and
  asserted by tests — CI catches them before the GPU does.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
