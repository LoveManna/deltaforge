"""Correctness-gate tests, on the tiny config.

The important property is that the gates report **magnitudes**, not verdicts. A kernel
that squeaks past a 1e-2 tolerance at 9e-3 is one shape change away from failing, and the
record has to say so.
"""

from __future__ import annotations

import pytest
import torch

from ..config import tiny_config
from ..model import greedy_decode
from ..reference import ReferenceModel
from .correctness import (
    CorrectnessReport,
    EndToEndCheck,
    TokenMatch,
    check_distribution,
    check_end_to_end,
    check_kernel,
    error_magnitudes,
)


@pytest.fixture
def model():
    torch.manual_seed(4242)
    model = ReferenceModel(tiny_config()).to(torch.float32).eval()
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.05)
    return model


# -- error magnitudes -----------------------------------------------------------------


def test_identical_tensors_have_zero_error():
    x = torch.randn(4, 5)
    max_abs, max_rel, excluded, total = error_magnitudes(x, x.clone())

    assert max_abs == 0.0
    assert max_rel == 0.0
    assert total == 20


def test_absolute_and_relative_errors_are_computed_separately():
    reference = torch.tensor([[10.0, 100.0]])
    candidate = torch.tensor([[11.0, 101.0]])

    max_abs, max_rel, _excluded, _total = error_magnitudes(reference, candidate)

    assert max_abs == pytest.approx(1.0)
    # The worst *relative* error is on the smaller value, not the larger.
    assert max_rel == pytest.approx(0.1)


def test_near_zero_reference_values_are_excluded_from_the_relative_statistic():
    """A relative error against a near-zero reference is noise, and would otherwise
    dominate the statistic and hide a real problem elsewhere."""
    reference = torch.tensor([[1e-9, 5.0]])
    candidate = torch.tensor([[1e-3, 5.05]])

    max_abs, max_rel, excluded, total = error_magnitudes(reference, candidate, atol=1e-2)

    assert excluded == 1
    assert total == 2
    assert max_rel == pytest.approx(0.01, rel=1e-4)  # only the 5.0 element counted
    assert max_abs == pytest.approx(0.05, rel=1e-4)  # but absolute error still sees everything


def test_all_near_zero_gives_a_defined_relative_error():
    reference = torch.zeros(3)
    candidate = torch.full((3,), 1e-6)

    max_abs, max_rel, excluded, total = error_magnitudes(reference, candidate)

    assert excluded == total == 3
    assert max_rel == 0.0
    assert max_abs == pytest.approx(1e-6)


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="shape mismatch"):
        error_magnitudes(torch.zeros(2, 3), torch.zeros(3, 2))


# -- layer 1 --------------------------------------------------------------------------


def test_check_kernel_records_numbers_not_a_bare_verdict():
    def reference(x):
        return x * 2.0

    def candidate(x):
        return x * 2.0 + 0.001

    check = check_kernel("scaled", reference, candidate, (torch.ones(4, 4),), replaces="rms_norm")

    assert check.passed is True
    assert check.max_abs_err == pytest.approx(0.001, rel=1e-4)
    assert check.max_rel_err == pytest.approx(0.0005, rel=1e-4)
    assert check.replaces == "rms_norm"
    assert check.shape == (4, 4)
    assert check.rtol == 1e-2 and check.atol == 1e-2
    # The magnitude survives serialisation: it is the point of the record.
    assert check.to_dict()["max_abs_err"] == pytest.approx(0.001, rel=1e-4)


def test_check_kernel_fails_but_still_reports_how_wrong_it_was():
    """A candidate that was fast but incorrect is among the most valuable things a future
    session can read, so the failure carries its magnitudes."""

    def reference(x):
        return x

    def candidate(x):
        return x * 3.0

    check = check_kernel("broken", reference, candidate, (torch.ones(2, 2),))

    assert check.passed is False
    assert check.max_abs_err == pytest.approx(2.0)
    assert check.max_rel_err == pytest.approx(2.0)


def test_check_kernel_uses_bf16_tolerances_by_default():
    def reference(x):
        return x

    def candidate(x):
        return x + 0.005  # inside 1e-2, outside 1e-3

    assert check_kernel("k", reference, candidate, (torch.ones(3),)).passed
    assert not check_kernel("k", reference, candidate, (torch.ones(3),), rtol=1e-4, atol=1e-4).passed


def test_check_kernel_compares_every_output_of_a_multi_output_operation():
    """The delta rule returns both an output and a recurrent state; a kernel that gets
    the output right and the state wrong would pass a single-output check and then drift
    on the next token."""

    def reference(x):
        return x, x * 10.0

    def candidate(x):
        return x, x * 10.0 + 1.0  # second output is wrong

    check = check_kernel("two_outputs", reference, candidate, (torch.ones(2),))

    assert not check.passed
    assert check.max_abs_err == pytest.approx(1.0)


def test_check_kernel_rejects_mismatched_output_counts():
    with pytest.raises(ValueError, match="reference returned 2 tensors"):
        check_kernel("k", lambda x: (x, x), lambda x: x, (torch.ones(2),))


def test_check_kernel_rejects_a_non_tensor_result():
    with pytest.raises(TypeError, match="cannot compare"):
        check_kernel("k", lambda x: "nope", lambda x: "nope", (torch.ones(2),))


# -- layer 2 --------------------------------------------------------------------------


def test_identical_models_match_token_for_token(model):
    prompts = [[1, 2, 3], [7, 8]]

    result = check_end_to_end(model, model, prompts, max_new_tokens=6, prompt_digest="abc")

    assert result.passed
    assert result.prompt_digest == "abc"
    assert len(result.per_prompt) == 2
    for match in result.per_prompt:
        assert match.matched
        assert match.first_divergence is None
        assert match.num_tokens == 6


def test_a_drifting_model_is_caught_with_the_divergence_point(model):
    """The failure layer 1 structurally cannot see: every kernel passes in isolation, but
    the assembled pipeline accumulates drift."""
    drifted = ReferenceModel(tiny_config()).to(torch.float32).eval()
    drifted.load_state_dict(model.state_dict())
    with torch.no_grad():
        drifted.layers[1].mlp.down_proj.weight.add_(0.5)

    result = check_end_to_end(model, drifted, [[1, 2, 3]], max_new_tokens=8)

    assert not result.passed
    match = result.per_prompt[0]
    assert not match.matched
    assert match.first_divergence is not None
    assert match.reference_tokens != match.candidate_tokens


def test_greedy_decode_is_deterministic(model):
    ids = torch.tensor([[3, 1, 4, 1, 5]])
    first = greedy_decode(model, ids, 10)
    second = greedy_decode(model, ids, 10)

    assert torch.equal(first, second)
    assert first.shape == (1, 10)


def test_greedy_decode_returns_only_new_tokens(model):
    ids = torch.tensor([[3, 1, 4]])
    assert greedy_decode(model, ids, 5).shape == (1, 5)


def test_greedy_decode_rejects_a_zero_token_request(model):
    with pytest.raises(ValueError, match="at least 1"):
        greedy_decode(model, torch.tensor([[1]]), 0)


def test_end_to_end_gate_rejects_an_empty_prompt_set(model):
    with pytest.raises(ValueError, match="no prompts"):
        check_end_to_end(model, model, [])


def test_end_to_end_gate_rejects_an_empty_prompt(model):
    with pytest.raises(ValueError, match="prompt 0 is empty"):
        check_end_to_end(model, model, [[]])


def test_end_to_end_result_serialises_the_token_sequences():
    check = EndToEndCheck(
        max_new_tokens=3,
        prompt_digest="d",
        per_prompt=(
            TokenMatch(
                prompt_index=0,
                matched=False,
                num_tokens=3,
                first_divergence=1,
                reference_tokens=(1, 2, 3),
                candidate_tokens=(1, 9, 3),
            ),
        ),
    )
    data = check.to_dict()

    assert data["passed"] is False
    assert data["per_prompt"][0]["first_divergence"] == 1
    assert data["per_prompt"][0]["reference_tokens"] == [1, 2, 3]


# -- the combined report --------------------------------------------------------------


def test_both_gates_are_required_to_pass():
    passing_kernel = check_kernel("k", lambda x: x, lambda x: x, (torch.ones(2),))
    failing_kernel = check_kernel("k", lambda x: x, lambda x: x * 5, (torch.ones(2),))
    passing_e2e = EndToEndCheck(
        max_new_tokens=1,
        prompt_digest="d",
        per_prompt=(TokenMatch(0, True, 1, None, (1,), (1,)),),
    )
    failing_e2e = EndToEndCheck(
        max_new_tokens=1,
        prompt_digest="d",
        per_prompt=(TokenMatch(0, False, 1, 0, (1,), (2,)),),
    )

    assert CorrectnessReport((passing_kernel,), passing_e2e).passed
    assert not CorrectnessReport((failing_kernel,), passing_e2e).passed
    assert not CorrectnessReport((passing_kernel,), failing_e2e).passed
    assert not CorrectnessReport((failing_kernel,), failing_e2e).passed


def test_an_empty_report_passes_vacuously():
    """With no kernels registered there is nothing for layer 1 to compare. That is the
    bootstrap state, and it must not read as a failure."""
    assert CorrectnessReport().passed
    assert CorrectnessReport().worst_max_abs_err == 0.0


def test_report_surfaces_the_worst_magnitudes_across_kernels():
    small = check_kernel("small", lambda x: x, lambda x: x + 0.001, (torch.ones(2),))
    large = check_kernel("large", lambda x: x, lambda x: x + 0.009, (torch.ones(2),))

    report = CorrectnessReport((small, large))

    assert report.passed, "both are inside the bf16 tolerance"
    assert report.worst_max_abs_err == pytest.approx(0.009, rel=1e-4)
    data = report.to_dict()
    assert len(data["layer1_kernel_checks"]) == 2
    assert data["worst_max_abs_err"] == pytest.approx(0.009, rel=1e-4)


# -- layer 2, approximate -------------------------------------------------------------


def test_an_identical_model_agrees_perfectly_with_itself(model):
    result = check_distribution(
        model, model, [[1, 2, 3], [7, 8]], top1_threshold=1.0, kl_threshold=0.0, max_new_tokens=4
    )

    assert result.passed
    assert result.top1_agreement == 1.0
    assert result.mean_kl == pytest.approx(0.0, abs=1e-6)
    assert result.num_positions > 0


def test_the_scored_positions_are_the_prompt_plus_the_references_own_continuation(model):
    """Teacher-forced over where the reference actually goes, not over the prompt alone.

    The prompt is a handful of positions of a distribution the model has not committed to;
    the decode regime is what the benchmark measures, so it is what the gate must score.
    """
    result = check_distribution(
        model, model, [[1, 2, 3]], top1_threshold=1.0, kl_threshold=0.0, max_new_tokens=5
    )

    assert result.num_positions == 3 + 5
    assert result.context_tokens == 3 + 5


def test_a_perturbed_model_fails_on_agreement_and_reports_how_far(model):
    perturbed = ReferenceModel(tiny_config()).to(torch.float32).eval()
    perturbed.load_state_dict(model.state_dict())
    with torch.no_grad():
        perturbed.layers[1].mlp.down_proj.weight.add_(0.5)

    result = check_distribution(
        model, perturbed, [[1, 2, 3]], top1_threshold=0.98, kl_threshold=0.01, max_new_tokens=8
    )

    assert not result.passed
    assert result.top1_agreement < 1.0
    assert result.mean_kl > 0.0
    assert result.max_kl >= result.mean_kl


def test_agreement_alone_cannot_pass_a_shredded_distribution(model):
    """Both statistics are required, and this is why.

    A candidate can keep the argmax and still destroy everything under it. Gating on
    agreement alone would call that correct; the KL bar is what sees it.
    """

    class HalfLogits(ReferenceModel):
        """Scales the logits by 0.5. A monotone map, so every argmax is unchanged and the
        agreement statistic is exactly 1.0 — while the distribution underneath is a
        different one."""

        def project_logits(self, hidden_states):
            return super().project_logits(hidden_states) * 0.5

    perturbed = HalfLogits(tiny_config()).to(torch.float32).eval()
    perturbed.load_state_dict(model.state_dict())

    result = check_distribution(
        model, perturbed, [[1, 2, 3]], top1_threshold=1.0, kl_threshold=1e-4, max_new_tokens=6
    )

    assert result.top1_agreement == 1.0
    assert result.mean_kl > 1e-4
    assert not result.passed


def test_the_distribution_gate_serialises_its_bars_alongside_its_numbers(model):
    result = check_distribution(
        model, model, [[1, 2]], top1_threshold=0.97, kl_threshold=0.02, max_new_tokens=3
    )

    payload = result.to_dict()
    assert payload["policy"] == "approximate"
    assert payload["top1_threshold"] == 0.97
    assert payload["kl_threshold"] == 0.02
    assert payload["passed"] is True


def test_a_report_with_a_distribution_check_says_which_policy_it_used(model):
    result = check_distribution(
        model, model, [[1, 2]], top1_threshold=1.0, kl_threshold=0.0, max_new_tokens=2
    )

    report = CorrectnessReport(distribution=result)

    assert report.passed
    assert report.to_dict()["layer2_policy"] == "approximate"
    assert report.to_dict()["layer2_end_to_end"] is None


def test_a_failing_distribution_check_fails_the_report(model):
    failing = check_distribution(
        model, model, [[1, 2]], top1_threshold=1.1, kl_threshold=0.0, max_new_tokens=2
    )

    assert not CorrectnessReport(distribution=failing).passed


def test_the_distribution_gate_rejects_an_empty_prompt_set(model):
    with pytest.raises(ValueError, match="no prompts"):
        check_distribution(model, model, [], top1_threshold=1.0, kl_threshold=0.0)


def test_the_distribution_gate_rejects_an_empty_prompt(model):
    with pytest.raises(ValueError, match="prompt 0 is empty"):
        check_distribution(model, model, [[]], top1_threshold=1.0, kl_threshold=0.0)
