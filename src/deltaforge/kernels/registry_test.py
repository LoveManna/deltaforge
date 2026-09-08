"""Registry invariants.

The one that matters: at most one champion per replaceable operation. Two champions for
the same operation makes "what does model.py assemble?" ambiguous, and an ambiguous
champion makes the leaderboard a lie.
"""

from __future__ import annotations

import pytest

from . import REGISTRY, KernelRegistry, KernelStatus, RegistryError


def noop(*_args, **_kwargs):
    return None


@pytest.fixture
def registry():
    return KernelRegistry()


# -- the shipped registry -------------------------------------------------------------


def test_the_shipped_registry_holds_its_invariants():
    """Whatever kernels are registered, the process-wide registry must be internally
    consistent at import time: at most one champion per operation, and every entry
    replacing an operation the reference actually exposes."""
    REGISTRY.check_invariants()
    for op, entry in REGISTRY.champions().items():
        assert entry.replaces == op
        assert entry.status is KernelStatus.CHAMPION


def test_every_shipped_champion_has_an_installer():
    """A champion with no installer is a hard error at build time. Catching it here means
    catching it on a CPU laptop instead of after paying to provision a GPU."""
    from ..model import INSTALLERS

    for op, entry in REGISTRY.champions().items():
        assert entry.name in INSTALLERS, f"{entry.name!r} is champion of {op!r} with no installer"


def test_every_shipped_kernel_names_the_hypothesis_that_produced_it():
    """The registry is part of the record: an entry with no hypothesis cannot be traced
    back to the measurement that justified it."""
    for entry in REGISTRY:
        assert entry.hypothesis, f"{entry.name!r} carries no hypothesis id"


# -- registration ---------------------------------------------------------------------


def test_register_and_retrieve(registry):
    entry = registry.register("fused_rmsnorm", noop, replaces="rms_norm")

    assert registry.get("fused_rmsnorm") is entry
    assert entry.status is KernelStatus.CANDIDATE
    assert "fused_rmsnorm" in registry
    assert len(registry) == 1


def test_duplicate_name_is_rejected(registry):
    registry.register("k", noop, replaces="rms_norm")
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("k", noop, replaces="swiglu_mlp")


def test_unknown_replaced_operation_is_rejected(registry):
    with pytest.raises(RegistryError, match="unknown operation 'nonsense'"):
        registry.register("k", noop, replaces="nonsense")


def test_unknown_kernel_lookup_is_rejected(registry):
    with pytest.raises(RegistryError, match="no kernel registered as 'ghost'"):
        registry.get("ghost")


def test_status_accepts_a_plain_string(registry):
    entry = registry.register("k", noop, replaces="rms_norm", status="champion")
    assert entry.status is KernelStatus.CHAMPION


# -- the champion invariant -----------------------------------------------------------


def test_at_most_one_champion_per_operation(registry):
    registry.register("first", noop, replaces="gated_delta_rule", status=KernelStatus.CHAMPION)

    with pytest.raises(RegistryError, match="already holds it"):
        registry.register("second", noop, replaces="gated_delta_rule", status=KernelStatus.CHAMPION)

    assert registry.champion("gated_delta_rule").name == "first"


def test_different_operations_may_each_have_a_champion(registry):
    registry.register("a", noop, replaces="rms_norm", status=KernelStatus.CHAMPION)
    registry.register("b", noop, replaces="swiglu_mlp", status=KernelStatus.CHAMPION)

    champions = registry.champions()
    assert set(champions) == {"rms_norm", "swiglu_mlp"}
    assert champions["rms_norm"].name == "a"
    registry.check_invariants()


def test_promote_demotes_the_incumbent_in_the_same_step(registry):
    registry.register("old", noop, replaces="gqa_attention", status=KernelStatus.CHAMPION)
    registry.register("new", noop, replaces="gqa_attention")

    promoted = registry.promote("new")

    assert promoted.status is KernelStatus.CHAMPION
    assert registry.get("old").status is KernelStatus.RETIRED
    assert registry.champion("gqa_attention").name == "new"
    # Never two champions, never zero after there was one.
    assert len(registry.for_op("gqa_attention")) == 2
    registry.check_invariants()


def test_promoting_the_current_champion_is_a_no_op(registry):
    registry.register("only", noop, replaces="rms_norm", status=KernelStatus.CHAMPION)
    assert registry.promote("only").status is KernelStatus.CHAMPION
    assert registry.champion("rms_norm").name == "only"


def test_retired_kernels_stay_registered(registry):
    """A retired kernel plus its graveyard entry is what stops a later session re-running
    a dead end."""
    registry.register("beaten", noop, replaces="rms_norm", status=KernelStatus.CHAMPION)
    registry.retire("beaten")

    assert "beaten" in registry
    assert registry.get("beaten").status is KernelStatus.RETIRED
    assert registry.champion("rms_norm") is None


def test_operation_with_no_champion_returns_none(registry):
    registry.register("candidate_only", noop, replaces="kv_cache_update")
    assert registry.champion("kv_cache_update") is None
    assert registry.champions() == {}


def test_check_invariants_catches_a_registry_corrupted_behind_its_own_back(registry):
    """Enforcement lives in register/promote; this proves the audit is independent of
    them and would notice if something bypassed the API."""
    from dataclasses import replace as dc_replace

    registry.register("a", noop, replaces="rms_norm", status=KernelStatus.CHAMPION)
    registry.register("b", noop, replaces="rms_norm")
    registry._entries["b"] = dc_replace(registry._entries["b"], status=KernelStatus.CHAMPION)

    with pytest.raises(RegistryError, match="2 champions"):
        registry.check_invariants()


def test_check_invariants_catches_a_mismatched_key(registry):
    from dataclasses import replace as dc_replace

    registry.register("a", noop, replaces="rms_norm")
    registry._entries["a"] = dc_replace(registry._entries["a"], name="different")

    with pytest.raises(RegistryError, match="carries name"):
        registry.check_invariants()


# -- housekeeping ---------------------------------------------------------------------


def test_metadata_is_carried_through(registry):
    entry = registry.register("k", noop, replaces="qkv_projection_rope", hypothesis="003", notes="fuses RoPE")
    assert entry.hypothesis == "003"
    assert entry.notes == "fuses RoPE"
    assert entry.replaces == "qkv_projection_rope"


def test_unregister_and_clear(registry):
    registry.register("a", noop, replaces="rms_norm")
    registry.register("b", noop, replaces="swiglu_mlp")

    registry.unregister("a")
    assert "a" not in registry
    registry.clear()
    assert len(registry) == 0


def test_iteration_yields_entries(registry):
    registry.register("a", noop, replaces="rms_norm")
    registry.register("b", noop, replaces="swiglu_mlp")

    assert {e.name for e in registry} == {"a", "b"}


def test_registries_are_isolated_from_each_other():
    """Tests must not be able to poison the process-wide registry."""
    one, two = KernelRegistry(), KernelRegistry()
    one.register("k", noop, replaces="rms_norm")

    assert "k" not in two
    assert "k" not in REGISTRY


def test_every_registered_kernel_has_an_installer_not_only_the_champions():
    """Batch mode promotes retired entries into scoped registries, so a retired kernel
    with no installer is a slot that errors on the rented box rather than on a laptop."""
    from ..model import INSTALLERS

    for entry in REGISTRY:
        assert entry.name in INSTALLERS, f"{entry.name!r} is registered with no installer"


def test_every_deferred_relative_import_in_a_kernel_resolves():
    """Kernels import from the rest of the package *inside* functions, to keep Triton off
    the CPU import path. Nothing then checks those imports until the function runs -- and
    for a kernel body, that means on a rented GPU.

    `gated_delta_step` imported `STATE_DTYPE` from `..config`, where it has never lived
    (it is defined in `reference`). The CPU suite was green, and the error surfaced as a
    lost slot on rental 21 at 2026-09-08, in a batch that had already paid for a container,
    torch, a 9.32 GB checkpoint and a max-autotune compile.

    Resolve every one of them statically instead. This is a pure AST + getattr check: it
    needs no GPU, no Triton, and no kernel execution."""
    import ast
    import importlib
    import pathlib

    kernels_dir = pathlib.Path(__file__).resolve().parent
    failures = []

    for path in sorted(kernels_dir.glob("*.py")):
        if path.name.endswith("_test.py"):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            # `from ..reference import X` inside a function body.
            package = "deltaforge.kernels" if node.level == 1 else "deltaforge"
            target = f"{package}.{node.module}" if node.module else package
            try:
                module = importlib.import_module(target)
            except ImportError as exc:  # a missing module is just as fatal as a missing name
                failures.append(f"{path.name}:{node.lineno} cannot import {target}: {exc}")
                continue
            for alias in node.names:
                if alias.name != "*" and not hasattr(module, alias.name):
                    failures.append(
                        f"{path.name}:{node.lineno} imports {alias.name!r} from {target}, "
                        f"which does not define it"
                    )

    assert not failures, "unresolvable deferred imports:\n" + "\n".join(failures)
