# Spend ledger

`spend.jsonl` is an append-only record, one row per instance lifecycle event. It is
currently **empty: nothing has been provisioned and nothing has been spent.**

A `provision` row is written **before the instance is used**, then reconciled by a
`destroy` row carrying the actual minutes and cost. A session that dies mid-run therefore
still leaves a record of what it started — the failure this guards against is an agent
that provisions, crashes, and leaves an instance running with nothing to say it exists.

Two gates read this file:

- **Month-to-date** — provisioning refuses at >= $45 against a $50 budget.
- **Session GPU-time** — at 60 cumulative billed minutes a session may not start another
  hypothesis.

Both count an un-reconciled instance at its **estimated ceiling**, never at zero: a gate
that assumed the best about a possibly-still-running instance would fail open exactly when
money is being spent.

Two implementations read the same file — `src/deltaforge/ledger.py` in Python, and awk
inside `remote/lib.sh` for the shell scripts, which run before any Python environment
exists. `remote/scripts_test.py` asserts the two agree on shared fixtures.

That is why rows are **flat**: no nested objects, and no quotes, braces, commas or
backslashes inside string values. `ledger.py` rejects any row that would break the awk
reader.
