"""Measurement: interleaved timing, the two correctness gates, and results records.

`bench` knows nothing about Qwen, Triton or the registry — it times opaque named
callables, which is what lets its arithmetic be tested exhaustively without a GPU.
"""

from __future__ import annotations

__all__: list[str] = []
