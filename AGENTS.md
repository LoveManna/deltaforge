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

uv run python -m deltaforge.cli fusion --batch NNN-slug        # which slots install a barrier
uv run python -m deltaforge.cli fusion --install some_kernel   # compile here, diff the generated code
```

## Maintaining project memory

Keep `AGENT.md` for knowledge useful to almost every future session. Do not repeat what the
codebase already shows; point to the authoritative file or command instead. Prefer rewriting
or pruning existing entries over appending new ones, and keep entries concise.

**Every session ends with a docs pass** — what it did, what it found, and what it proved
wrong. `AGENT.md` §6.1 says which docs to consider, and how to tell an entry worth writing
from a changelog nobody needs.
