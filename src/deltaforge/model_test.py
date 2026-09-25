"""Assembly tests: what happens when champions exist, and when they do not."""

from __future__ import annotations

import pytest
import torch

from .config import tiny_config
from .kernels import REGISTRY, KernelRegistry, KernelStatus
from .model import InstallerMissing, apply_champions, build_model, greedy_decode, prefill_setup
from .reference import ReferenceModel


def noop(*_args, **_kwargs):
    return None


@pytest.fixture
def config():
    return tiny_config()


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

    applied = apply_champions(model, registry, installers={"fused": lambda m, e: seen.append((m, e))})

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


def test_prefill_setup_runs_the_prefill_once_and_restores_it_after(config):
    """What the bench's per-round setup costs, and what it is allowed to cost.

    Rental 46 paid ~18 s a round against a timed region of ~2 s, and the difference is the
    2048-token prefill that `run_interleaved` runs as setup and then excludes from every
    measurement. At 15 scoring rounds and two columns that is most of a slot. The prefill
    is deterministic, so a round after the first only needs the state it left behind.
    """
    model = ReferenceModel(config)
    prompt = torch.randint(0, config.vocab_size, (1, 6))
    cache = model.new_cache(1, 16)
    calls = []
    original = model.forward

    def counted(*args, **kwargs):
        calls.append(args[0].shape[1])
        return original(*args, **kwargs)

    model.forward = counted

    setup = prefill_setup(model, prompt, cache)
    setup()
    after_first = cache.seq_len
    with torch.no_grad():
        model(prompt[:, -1:], cache, num_logits_to_keep=1)
    setup()
    setup()

    # 6 is the prefill, 1 is the decode step this test makes between the two setups.
    assert calls == [6, 1], "the prefill ran more than once"
    assert cache.seq_len == after_first == 6


def test_prefill_setup_leaves_the_cache_where_the_prefill_left_it(config):
    model = ReferenceModel(config)
    prompt = torch.randint(0, config.vocab_size, (1, 6))
    cache = model.new_cache(1, 16)

    setup = prefill_setup(model, prompt, cache)
    setup()
    with torch.no_grad():
        first, _ = model(prompt[:, -1:], cache, num_logits_to_keep=1)
    setup()
    with torch.no_grad():
        second, _ = model(prompt[:, -1:], cache, num_logits_to_keep=1)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_greedy_decode_uses_an_installed_decode_loop(model, config):
    """The bench times `greedy_decode`, so a candidate that replaces the loop must be
    reachable from there — a speculative decoder is not a module swap, and
    `apply_champions` has no other way to put one in front of the benchmark."""
    seen = {}

    def loop(runnable, input_ids, max_new_tokens, cache):
        seen["args"] = (runnable, input_ids.shape, max_new_tokens, cache)
        return torch.zeros((input_ids.shape[0], max_new_tokens), dtype=torch.long)

    model.decode_loop = loop
    ids = torch.randint(0, config.vocab_size, (1, 3))
    cache = model.new_cache(1, 16)

    out = greedy_decode(model, ids, 4, cache=cache)

    assert out.shape == (1, 4)
    assert seen["args"][0] is model
    assert seen["args"][2] == 4
    assert seen["args"][3] is cache


def test_greedy_decode_without_a_loop_is_unchanged(model, config):
    ids = torch.randint(0, config.vocab_size, (1, 3))

    out = greedy_decode(model, ids, 4)

    assert out.shape == (1, 4)


def test_a_loop_returning_the_wrong_number_of_tokens_is_refused(model, config):
    """The ratio divides by a fixed token count. A loop that emitted 130 tokens where the
    reference emitted 128 would look 1.6% faster for having done more work."""
    model.decode_loop = lambda runnable, input_ids, max_new_tokens, cache: torch.zeros(
        (1, max_new_tokens - 1), dtype=torch.long
    )
    ids = torch.randint(0, config.vocab_size, (1, 3))

    with pytest.raises(RuntimeError, match="returned 3 tokens, expected 4"):
        greedy_decode(model, ids, 4)
