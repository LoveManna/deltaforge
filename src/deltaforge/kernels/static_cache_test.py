"""021 has no arithmetic, so everything about it that can be wrong is checkable on a CPU.

What the rental measures is whether CUDA graphs are worth 1.26x. What this file measures is
whether the candidate actually asked for them — a distinction blocker 16 cost six slots to
learn, because a candidate that silently did not do the thing under test still returns a
plausible ratio.
"""

from __future__ import annotations

import torch

from ..config import tiny_config
from ..reference import ReferenceModel
from .static_cache import (
    _static_cache_model_class,
    install_static_decode_cache,
    mark_cache_static,
    static_cache_correctness_checks,
)


def _model():
    return ReferenceModel(tiny_config()).eval()


def _cache_tensors(cache):
    return [value for layer in cache.layers for value in vars(layer).values() if torch.is_tensor(value)]


def test_the_promise_torch_requires_is_the_promise_that_gets_made():
    """Pinned against torch's own spelling rather than against an assumption.

    `cudagraph_utils.check_for_mutation` exempts an input index that appears in
    `static_input_idxs`, and `_aot_autograd` puts an index there when the placeholder's
    `tensor_dict` carries `_dynamo_static_input_type` — which is the attribute
    `mark_static_address` sets. This test is the link in that chain this repository owns; if
    a torch upgrade renames the attribute, the hypothesis stops being true and this fails
    rather than the rental quietly measuring 1.00.
    """
    cache = _model().new_cache(1, 8)

    marked = mark_cache_static(cache)

    tensors = _cache_tensors(cache)
    assert marked == len(tensors) > 0
    assert all(t._dynamo_static_input_type == "unguarded" for t in tensors)


def test_every_cache_tensor_is_marked_not_merely_the_conv_history():
    """The dump names `cache.conv.copy_` because it prints the *first* mutating stack, and
    it counts 64 instances: 24 conv histories, 24 recurrent states, 16 KV slices for the
    eight full-attention layers. Marking only the one the message names would leave 40
    mutated inputs, and inductor's check is all-or-nothing over the region — so the slot
    would measure a null for a hypothesis it never tested."""
    from ..reference import _AttentionLayerCache, _LinearLayerCache

    model = _model()
    install_static_decode_cache(model)
    cache = model.new_cache(1, 8)

    kinds = {type(layer) for layer in cache.layers}
    assert kinds == {_AttentionLayerCache, _LinearLayerCache}, "the fixture has both layer types"
    assert all(getattr(t, "_dynamo_static_input_type", None) for t in _cache_tensors(cache))


def test_installing_changes_the_root_class_and_nothing_else():
    """No module is replaced, no parameter is copied, and no arithmetic moves. The only
    difference between this candidate and the reference is a property of the cache."""
    model = _model()
    before_modules = {name: type(m) for name, m in model.named_modules() if name}
    before_params = {name: p.data_ptr() for name, p in model.named_parameters()}

    install_static_decode_cache(model)

    assert {name: type(m) for name, m in model.named_modules() if name} == before_modules
    assert {name: p.data_ptr() for name, p in model.named_parameters()} == before_params
    assert type(model) is not ReferenceModel
    assert isinstance(model, ReferenceModel)


def test_installing_is_idempotent():
    model = _model()
    install_static_decode_cache(model)
    once = type(model)

    install_static_decode_cache(model)

    assert type(model) is once


def test_the_reference_is_left_allocating_ordinary_caches():
    """The two columns must differ in this and in nothing else, which means the reference
    has to keep getting an unmarked cache. `_make_column` builds each column's cache from
    that column's own model, so this is the whole of the isolation."""
    reference, candidate = _model(), _model()
    install_static_decode_cache(candidate)

    assert all(
        getattr(t, "_dynamo_static_input_type", None) is None
        for t in _cache_tensors(reference.new_cache(1, 8))
    )
    assert all(
        getattr(t, "_dynamo_static_input_type", None) for t in _cache_tensors(candidate.new_cache(1, 8))
    )


def test_the_layer_one_check_counts_and_restores_the_class():
    """Layer 1 runs against `self.reference`, which no install has touched, so the check
    has to borrow the behaviour and put the class back — leaving it patched would turn the
    reference into the candidate for every slot after this one."""
    model = _model()
    before = type(model)

    checks = static_cache_correctness_checks(model, device="cpu")

    assert type(model) is before
    assert len(checks) == 1
    assert checks[0].passed, checks[0]


def test_the_layer_one_check_fails_when_a_tensor_is_left_unmarked(monkeypatch):
    """A gate that cannot fail reports nothing. This is the failure it exists for."""
    from . import static_cache

    model = _model()

    def marks_all_but_one(cache):
        tensors = _cache_tensors(cache)
        for tensor in tensors[:-1]:
            torch._dynamo.mark_static_address(tensor)
        return len(tensors) - 1

    monkeypatch.setattr(static_cache, "mark_cache_static", marks_all_but_one)
    static_cache._PATCHED.clear()
    try:
        checks = static_cache_correctness_checks(model, device="cpu")
    finally:
        static_cache._PATCHED.clear()

    assert not checks[0].passed


def test_the_class_factory_stacks_rather_than_replacing():
    """Slots 026 and 027 compose this with the tiled LM head, which also swaps the root
    class. A factory anchored at `ReferenceModel` would discard whichever install ran
    first, and `_build_candidate`'s "did any module class change" check would still pass."""

    class AlreadyPatched(ReferenceModel):
        pass

    stacked = _static_cache_model_class(AlreadyPatched)

    assert issubclass(stacked, AlreadyPatched)
    assert _static_cache_model_class(AlreadyPatched) is stacked, "one class per composition"
    assert _static_cache_model_class(ReferenceModel) is not stacked
