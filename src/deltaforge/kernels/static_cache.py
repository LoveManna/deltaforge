"""021 — the decode step `max-autotune` refuses to CUDA-graph, and the one line that lets it.

Rental 38's `TORCH_LOGS=output_code` dump carries, 128 times::

    skipping cudagraphs due to mutated inputs (64 instances). Found from :
      File ".../reference.py", line 431, in _causal_conv
        cache.conv.copy_(x[..., -history:].to(cache.conv.dtype))

**So the compiled baseline has never been CUDA-graphed**, on any rental. That was recorded
after batch 004 as a curiosity. It is not a curiosity; it is the largest unattacked number
in the project, and this module is the cheapest possible test of it.

## The arithmetic, from the dump this repo already has

The decode graph is fully unrolled, so its launches can simply be counted::

    483 `triton_*.run(...)` call sites + 25 `extern_kernels.*` = **508 kernel launches per
    decoded token**

The compiled column runs that step in **7.30 ms** while moving **8587.80 MB**. A real
kernel reaches 75-90% of an RTX 5090's 1792 GB/s, so the memory-bound part of the step is
**4.79 ms at vendor peak and 5.3-6.4 ms at an achievable one**. The residue is 0.9-2.5 ms
spread over 508 launches: **1.8-4.9 µs each**, which is exactly what inductor's Python
launch path costs when nothing has been captured into a graph.

A CUDA graph replaces all 508 dispatches with one. **Ceiling: 7.30 / 5.8 ≈ 1.26x**, and
unlike every other hypothesis in this repository it asks nothing of our arithmetic — the
kernels that run are inductor's own, in inductor's own order.

## Why the compiler cannot do this for itself, which is the finding

`torch/_inductor/cudagraph_utils.py` is not being conservative by accident::

    mutation_indices = [
        idx for idx in func.mutated_input_idxs
        if not (idx in func.static_input_idxs or is_cuda_graph_recorded_tensor(inputs[idx]))
    ]

A CUDA graph bakes in the *addresses* its kernels write to. An input tensor that the graph
mutates is therefore only safe if the caller promises to pass the same storage every time,
and the compiler has no way to know whether the caller will. Parameters and buffers carry
that promise structurally; a `DecodeCache` built by the harness and handed in as an
argument does not, so all 64 mutated cache tensors — 24 conv histories, 24 recurrent
states, 16 KV slices — are counted and the whole region is skipped.

`torch._dynamo.mark_static_address` **is** that promise, and it is the mechanism every
production decode engine uses for exactly this reason: a persistent KV cache is the one
tensor an inference server can guarantee never moves.

So the candidate changes no arithmetic and writes no Triton. It allocates the same cache,
in the same shapes and dtypes, and says the thing about it that the compiler cannot infer.

## What must be observable, or the slot is worthless

Blocker 16 is the standing lesson here: a candidate that silently did not do the thing
under test still returns a plausible ratio. If cudagraphs fail to engage for some reason
this module did not anticipate, the slot measures 1.00 and *looks* like a refuted
hypothesis. `batch_run.cudagraphs_during` records the recorded-node count and the skip
counter alongside the ratio, so "it did not engage" and "it engaged and bought nothing"
are different rows rather than the same one.

**What the two counters should say.** The skip is logged once per *function id*, and the
dump's 128 occurrences are 128 decode steps: `cache_offset` reaches the graph as a symint
and `cudagraphify_impl` keys a recording on each distinct int, so every step of a
128-token decode is its own function id. Expect, therefore:

* slot 0 — `cudagraph_skips` around 128 and `cudagraph_nodes` 0. That is the *reference*
  being refused for the first time, and it is the baseline's behaviour on every rental
  this project has run.
* slot 021 engaged — `cudagraph_nodes` at least 128 and a skip delta near zero.
* slot 021 not engaged — a second ~128 skips, from the candidate's own function ids.

Those 128 recordings are also the most likely way this fails: they are 128 graphs in one
tree, and the rerecord limit that would refuse them is itself 128.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "install_static_decode_cache",
    "mark_cache_static",
    "static_cache_correctness_checks",
]


def mark_cache_static(cache: Any) -> int:
    """Promise dynamo that every tensor in ``cache`` keeps its address. Returns how many.

    Walks the cache rather than naming its fields, because a field this misses is a
    mutated input that still blocks the whole region: the check is all-or-nothing across
    the graph, so covering 63 of 64 buys exactly nothing.
    """
    from torch import Tensor  # noqa: PLC0415 - keeps the registry importable without torch
    from torch._dynamo import mark_static_address  # noqa: PLC0415

    marked = 0
    for layer in cache.layers:
        for value in vars(layer).values():
            if isinstance(value, Tensor):
                mark_static_address(value)
                marked += 1
    return marked


_PATCHED: dict[str, type] = {}


def _static_cache_model_class(base: type | None = None) -> type:
    """A subclass of ``base`` whose caches are allocated static.

    `batch_run.BatchRunner._make_column` builds each column's cache with
    ``model.new_cache(...)``, so the candidate's cache is already the candidate's to
    allocate. The reference is untouched and allocates its own, identically shaped, on the
    path it always has — which is what makes the two columns differ in this and nothing
    else.

    A class swap rather than a bound attribute, for the reason `tiled_gemv` gives: only one
    of the two is reliably traceable, and a graph break inside the candidate would hand
    back a ratio comparing two different amounts of compilation. Subclassed from whatever
    the model already is, because slots 026 and 027 compose this with the tiled LM head,
    which patches the root class too; anchoring either factory at `ReferenceModel` would
    make the second install discard the first without anything noticing.
    """
    if base is None:
        from ..reference import ReferenceModel  # noqa: PLC0415

        base = ReferenceModel
    key = f"static_cache:{base.__module__}.{base.__qualname__}"
    if key not in _PATCHED:

        class StaticCacheModel(base):  # type: ignore[misc, valid-type]
            def new_cache(self, batch_size: int, max_seq_len: int):
                cache = super().new_cache(batch_size, max_seq_len)
                self.deltaforge_static_cache_tensors = mark_cache_static(cache)
                return cache

        _PATCHED[key] = StaticCacheModel
    return _PATCHED[key]


def install_static_decode_cache(model, entry=None) -> None:
    if getattr(model, "_deltaforge_static_cache", False):
        return
    model._deltaforge_static_cache = True
    model.deltaforge_static_cache_tensors = 0
    model.__class__ = _static_cache_model_class(type(model))


def static_cache_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    """Layer 1 for a candidate with no arithmetic in it: that the promise was actually made.

    There is no kernel here to compare against a reference — the kernels are inductor's.
    What can fail is the promise silently covering only some of the cache, and that is
    fatal in a way a partial correctness error would not be: the mutation check is
    all-or-nothing over the graph, so 63 marked tensors out of 64 skip cudagraphs exactly
    as 0 would, and the slot would report a null for a hypothesis it never tested.

    So the check is a count, expressed as a numeric comparison the existing gate can score:
    tensors marked against tensors present.
    """
    import torch  # noqa: PLC0415

    from ..harness.correctness import check_kernel  # noqa: PLC0415

    installed = model.__class__
    if not getattr(model, "_deltaforge_static_cache", False):
        # Layer 1 runs against `self.reference`, which no install has touched, so the check
        # has to patch a copy of the behaviour onto it and put it back.
        model.__class__ = _static_cache_model_class(installed)
    try:
        cache = model.new_cache(1, 8)
        marked = int(getattr(model, "deltaforge_static_cache_tensors", 0))
    finally:
        model.__class__ = installed

    present = sum(
        1 for layer in cache.layers for value in vars(layer).values() if isinstance(value, torch.Tensor)
    )
    unmarked = [
        1
        for layer in cache.layers
        for value in vars(layer).values()
        if isinstance(value, torch.Tensor) and getattr(value, "_dynamo_static_input_type", None) is None
    ]
    return (
        check_kernel(
            "static_cache.every_cache_tensor_is_static",
            lambda: torch.tensor([float(present), 0.0]),
            lambda: torch.tensor([float(marked), float(len(unmarked))]),
            args=(),
            replaces="decode_cache",
            atol=0.0,
            rtol=0.0,
            note=(
                f"{marked} of {present} decode-cache tensors marked static; "
                "one unmarked tensor skips cudagraphs for the whole region"
            ),
        ),
    )
