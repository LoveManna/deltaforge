"""Assembly tests: what happens when champions exist, and when they do not."""

from __future__ import annotations

import pytest
import torch

from .config import tiny_config
from .kernels import REGISTRY, KernelRegistry, KernelStatus
from .model import InstallerMissing, apply_champions, build_model, greedy_decode
from .reference import ReferenceModel


def noop(*_args, **_kwargs):
    return None


@pytest.fixture
def model():
    return ReferenceModel(tiny_config())


def test_an_empty_registry_installs_nothing(model):
    """No kernels means the candidate column *is* the reference, unpatched."""
    assert apply_champions(model, KernelRegistry()) == ()


def test_the_shipped_registry_installs_its_champions(model):
    """Every champion in the process-wide registry installs without error on a tiny CPU
    model. Installation is structural — it swaps which forward runs — so it is checked
    here; whether the installed kernel is numerically right is the GPU gate's job."""
    installed = apply_champions(model, REGISTRY)

    assert set(installed) == {entry.name for entry in REGISTRY.champions().values()}


def test_installing_twice_is_a_no_op_rather_than_a_double_patch(model):
    apply_champions(model, REGISTRY)
    before = [type(layer) for layer in model.layers]

    apply_champions(model, REGISTRY)

    assert [type(layer) for layer in model.layers] == before


def test_a_champion_without_an_installer_fails_loudly(model):
    """Silently falling back to the reference would benchmark the baseline while
    labelling it the candidate — the worst failure this harness could have."""
    registry = KernelRegistry()
    registry.register("fused", noop, replaces="rms_norm", status=KernelStatus.CHAMPION)

    with pytest.raises(InstallerMissing, match="no installer is registered"):
        apply_champions(model, registry)


def test_a_registered_installer_is_called_with_its_entry(model):
    registry = KernelRegistry()
    entry = registry.register("fused", noop, replaces="rms_norm", status=KernelStatus.CHAMPION)
    seen = []

    applied = apply_champions(model, registry, installers={"rms_norm": lambda m, e: seen.append((m, e))})

    assert applied == ("fused",)
    assert seen == [(model, entry)]


def test_only_champions_are_installed(model):
    registry = KernelRegistry()
    registry.register("candidate_only", noop, replaces="rms_norm")
    registry.register("retired_one", noop, replaces="swiglu_mlp", status=KernelStatus.RETIRED)
    calls = []

    applied = apply_champions(
        model,
        registry,
        installers={"rms_norm": lambda m, e: calls.append(e), "swiglu_mlp": lambda m, e: calls.append(e)},
    )

    assert applied == ()
    assert calls == []


def test_apply_champions_checks_registry_invariants_first(model):
    from dataclasses import replace as dc_replace

    from .kernels import RegistryError

    registry = KernelRegistry()
    registry.register("a", noop, replaces="rms_norm", status=KernelStatus.CHAMPION)
    registry.register("b", noop, replaces="rms_norm")
    registry._entries["b"] = dc_replace(registry._entries["b"], status=KernelStatus.CHAMPION)

    with pytest.raises(RegistryError, match="2 champions"):
        apply_champions(model, registry)


def test_build_model_without_a_registry_is_the_pure_reference():
    """The eager and compiled columns must be the untouched baseline."""
    model = build_model(tiny_config(), dtype=torch.float32, registry=None)

    assert isinstance(model, ReferenceModel)
    assert not model.training


def test_build_model_with_the_empty_registry_still_works():
    model = build_model(tiny_config(), dtype=torch.float32, registry=REGISTRY)
    assert isinstance(model, ReferenceModel)


def test_greedy_decode_reuses_a_supplied_cache():
    torch.manual_seed(1)
    config = tiny_config()
    model = ReferenceModel(config).to(torch.float32).eval()
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.05)

    ids = torch.tensor([[5, 6, 7]])
    cache = model.new_cache(1, 3 + 4)
    tokens = greedy_decode(model, ids, 4, cache=cache)

    assert tokens.shape == (1, 4)
    # Three prompt tokens plus three generated ones were fed back through the cache. The
    # fourth generated token is the answer and is never consumed, so it is not committed.
    assert cache.seq_len == 3 + 4 - 1
