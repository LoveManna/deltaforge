"""DeltaForge: hand-written Triton kernels for the Qwen3.5-4B decode path.

The public surface is deliberately small:

* :mod:`deltaforge.reference` — the baseline, pure PyTorch, no custom kernels ever.
* :mod:`deltaforge.kernels` — the registry of hand-written kernels and their champions.
* :mod:`deltaforge.harness` — timing, correctness gates, results records.
* :mod:`deltaforge.ledger` — the append-only spend record and its two budget gates.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
