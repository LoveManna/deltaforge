"""Manifest tests, and the structural install check for every hypothesis in a batch.

The install check is the highest-value test in this repo right now. A hypothesis whose
installer silently patches nothing produces a candidate identical to the reference,
measures 1.00, and is recorded as a well-behaved null result — indistinguishable from a
real finding unless something asserts otherwise. Under the old workflow that cost a
rental to discover. Here it costs a laptop test.

Triton is not present in the CPU environment, so these exercise *installation* — which
class each module ends up with — not numerics. The numerics are the GPU gate's job and
run on the rented box.
"""

from __future__ import annotations

import pytest

from .batch import scoped_registry
from .batches import BATCH_001, BATCH_002, BATCHES, get_batch
from .config import tiny_config
from .kernels import REGISTRY
from .model import apply_champions
from .reference import ReferenceModel


def module_classes(model) -> dict[str, str]:
    """Every submodule's class name, keyed by path. Two of these differ iff a patch landed."""
    return {name: type(module).__name__ for name, module in model.named_modules()}


@pytest.fixture
def model():
    return ReferenceModel(tiny_config())


def test_batch_001_has_between_seven_and_twelve_hypotheses():
    # The batch exists to amortise a ~15-minute fixed rental cost. Fewer than 7 does not
    # justify it; more than 12 will not fit the 120-minute session gate.
    assert 7 <= len(BATCH_001) <= 12


def test_batch_001_starts_with_the_calibration_slot():
    """A broken harness must be discovered in three minutes, not at the end of a rental."""
    assert BATCH_001.hypotheses[0].is_identity
    assert BATCH_001.calibration_slug == "000-identity"


def test_every_hypothesis_names_kernels_that_exist():
    for hyp in BATCH_001:
        for name in hyp.kernels:
            assert name in REGISTRY, f"{hyp.slug!r} names unregistered kernel {name!r}"


def test_every_hypothesis_has_an_installer():
    from .model import INSTALLERS

    for hyp in BATCH_001:
        for name in hyp.kernels:
            assert name in INSTALLERS, f"{hyp.slug!r} names {name!r}, which has no installer"


def test_every_hypothesis_builds_a_scoped_registry():
    for hyp in BATCH_001:
        scoped = scoped_registry(hyp, REGISTRY)
        assert len(scoped) == len(hyp.kernels)


@pytest.mark.parametrize("hypothesis", list(BATCH_001), ids=lambda h: h.slug)
def test_every_hypothesis_installs_and_actually_changes_the_model(hypothesis, model):
    """The check that stops a no-op candidate being recorded as a null result."""
    before = module_classes(model)

    applied = apply_champions(model, scoped_registry(hypothesis, REGISTRY))

    after = module_classes(model)
    if hypothesis.is_identity:
        assert applied == ()
        assert after == before, "the identity champion must leave the model untouched"
    else:
        assert set(applied) == set(hypothesis.kernels)
        assert after != before, (
            f"{hypothesis.slug!r} installed {applied} but changed no module class. A "
            "candidate identical to the reference measures 1.00 and would be recorded as "
            "a null result rather than as the bug it is."
        )


@pytest.mark.parametrize("hypothesis", list(BATCH_001), ids=lambda h: h.slug)
def test_installing_a_hypothesis_is_idempotent(hypothesis, model):
    registry = scoped_registry(hypothesis, REGISTRY)
    apply_champions(model, registry)
    once = module_classes(model)

    apply_champions(model, registry)

    assert module_classes(model) == once


def test_each_hypothesis_patches_a_distinct_set_of_modules():
    """Two hypotheses that patch identically are the same experiment run twice.

    002 and 003 both install the same Triton RMSNorm and differ only in *where*; if that
    distinction ever collapsed, the batch would silently spend two slots measuring one
    thing and report them as independent results.
    """
    patched: dict[str, frozenset[str]] = {}
    for hyp in BATCH_001:
        if hyp.is_identity:
            continue
        model = ReferenceModel(tiny_config())
        before = module_classes(model)
        apply_champions(model, scoped_registry(hyp, REGISTRY))
        after = module_classes(model)
        patched[hyp.slug] = frozenset(k for k in after if after[k] != before[k])

    assert patched["002-rmsnorm-only"] != patched["003-qk-norm-triton"]
    for slug, sites in patched.items():
        assert sites, f"{slug!r} patched nothing"

    seen: dict[frozenset[str], str] = {}
    for slug, sites in patched.items():
        # 006 and 008 both replace the whole attention module and so patch the same sites;
        # that is intended, and 008's docstring says why it is the control for 006.
        if slug in ("006-gqa-no-expand", "008-flash-decode-splitkv"):
            continue
        assert sites not in seen, f"{slug!r} and {seen[sites]!r} patch identical sites"
        seen[sites] = slug


def test_002_and_003_touch_different_norms():
    """002 is the hidden-size norms, 003 is the head-dim norms. Neither may leak."""
    hidden = ReferenceModel(tiny_config())
    before = module_classes(hidden)
    apply_champions(hidden, scoped_registry(BATCH_001.get("002-rmsnorm-only"), REGISTRY))
    hidden_sites = {k for k in module_classes(hidden) if module_classes(hidden)[k] != before[k]}

    qk = ReferenceModel(tiny_config())
    apply_champions(qk, scoped_registry(BATCH_001.get("003-qk-norm-triton"), REGISTRY))
    qk_sites = {k for k in module_classes(qk) if module_classes(qk)[k] != before[k]}

    assert not any("q_norm" in s or "k_norm" in s for s in hidden_sites)
    assert qk_sites and all("q_norm" in s or "k_norm" in s for s in qk_sites)


def test_007_only_touches_linear_attention_layers():
    model = ReferenceModel(tiny_config())
    before = module_classes(model)
    apply_champions(model, scoped_registry(BATCH_001.get("007-gated-delta-fused-step"), REGISTRY))
    after = module_classes(model)

    changed = {k for k in after if after[k] != before[k]}
    assert changed and all("linear_attn" in s for s in changed)


def test_predictions_are_registered_for_every_slot():
    """The prediction is the result. A slot without one cannot be scored."""
    for hyp in BATCH_001:
        assert hyp.prediction
        assert len(hyp.rationale) > 80, f"{hyp.slug!r} has a rationale too thin to be a claim"


def test_only_the_highest_byte_share_slot_predicts_a_win():
    """A batch that predicted wins everywhere would not be a prediction, it would be hope."""
    winners = [h for h in BATCH_001 if h.prediction == "win"]
    assert [h.slug for h in winners] == ["006-gqa-no-expand"]
    others = [h.byte_share for h in BATCH_001 if h.prediction != "win"]
    assert winners[0].byte_share > max(others)


def test_get_batch_rejects_an_unknown_id():
    with pytest.raises(SystemExit, match="unknown batch"):
        get_batch("nope")


def test_get_batch_returns_the_manifest():
    assert get_batch("001-calibration") is BATCH_001
    assert "001-calibration" in BATCHES


def test_a_batch_holds_seven_to_twelve_hypotheses_unless_it_is_calibrating():
    """The floor amortises a rental's fixed cost across measurements.

    A calibration batch's product IS that cost, measured, so the argument for the floor
    cannot apply to it — and rental 22 showed the cost is not what `docs/BATCHES.md`
    assumed.
    """
    assert 7 <= len(BATCH_001) <= 12
    assert BATCH_001.is_calibration is False

    assert len(BATCH_002) < 7
    assert BATCH_002.is_calibration is True


def test_batch_002_opens_with_the_identity_champion():
    assert BATCH_002.hypotheses[0].is_identity
    assert BATCH_002.calibration_slug == "000-identity"


def test_batch_002_can_score_a_kernel_hypothesis():
    """Calibration is necessary and is not a result. A rental that measures only the
    identity slot has proved the harness works and scored no hypothesis."""
    assert sum(1 for h in BATCH_002 if not h.is_identity) >= 1


def test_batch_002_is_registered_and_fetchable():
    assert get_batch("002-compile-cost") is BATCH_002
