# Project agent memory

**Read [`AGENT.md`](AGENT.md).** It is the single entry point for a working session: what
this project is trying to prove, the arithmetic that decides whether a hypothesis is worth
trying, the session workflow, the repository map, and the traps.

Everything that used to live in this file is now there, so there is one place to look and
one place to update.

## Quick reference

```sh
uv sync --extra dev                                 # CPU-only env
uv run python docs/roofline.py                      # where the bytes go — run this first
uv run pytest                                       # must pass with no CUDA present
uv run ruff check . && uv run ruff format --check .
remote/run_remote.sh --dry-run --session-id smoke   # cost machinery, spends nothing
```

## Maintaining project memory

Keep `AGENT.md` for knowledge useful to almost every future session. Do not repeat what the
codebase already shows; point to the authoritative file or command instead. Prefer rewriting
or pruning existing entries over appending new ones, and keep entries concise.
