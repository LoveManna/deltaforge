# Hypothesis records

**Empty: no hypothesis has been attempted.**

One record per run, named `NNN-slug.json`, matching the branch `hyp/NNN-slug` and the row
in `LEADERBOARD.md`.

**Losses are recorded here too.** A hypothesis that was fast but incorrect, or simply
slower than the compiler, gets a full record with its timings and error magnitudes, plus a
graveyard entry in `docs/HYPOTHESES.md` explaining the mechanism that failed. Keeping only
the wins would make this a highlight reel instead of a log.

Format is described in `../README.md`.
