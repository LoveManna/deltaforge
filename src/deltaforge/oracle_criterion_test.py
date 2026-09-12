"""CPU tests for the oracle gate's decision rule.

The rule in `oracle_test.classify_step` decides whether the reference is validated, and
therefore whether any benchmark number this project produces is admissible. It runs only
on a rented GPU with a 9.32 GB checkpoint in front of it, which is the worst possible place
to discover that its arithmetic is wrong. The rule itself is pure, so it is exercised here
on hand-built logit rows with no CUDA and no weights.

The criterion, registered 2026-09-12 before the rental that tests it:

* same argmax -> agree;
* different argmax, and *neither* model prefers its own pick by more than the two models'
  logits disagree by on that row -> a rounding tie, not a disagreement;
* otherwise -> the models genuinely disagree.
"""

from __future__ import annotations

import torch

from .oracle_test import classify_step


def test_identical_rows_agree():
    row = torch.tensor([1.0, 5.0, 2.0])
    verdict, metrics = classify_step(row, row.clone())
    assert verdict == "agree"
    assert metrics["ours"] == metrics["theirs"] == 1
    assert metrics["noise"] == 0.0


def test_the_same_pick_agrees_even_when_the_rows_differ():
    """Agreement is about the argmax, not the logits. Two implementations always differ."""
    ours = torch.tensor([1.0, 5.0, 2.0])
    theirs = torch.tensor([1.3, 5.4, 1.8])
    verdict, _ = classify_step(ours, theirs)
    assert verdict == "agree"


def test_a_near_tie_the_two_models_split_is_a_tie_not_a_disagreement():
    """Tokens 1 and 2 are 0.1 apart for both models, while the models' own logits differ
    by 0.5. Which token wins is decided by rounding well below the noise the two
    implementations already carry."""
    ours = torch.tensor([0.0, 5.05, 4.95])
    theirs = torch.tensor([0.5, 4.95, 5.05])
    verdict, metrics = classify_step(ours, theirs)
    assert verdict == "tie", metrics
    assert metrics["ours"] == 1 and metrics["theirs"] == 2
    assert metrics["our_margin"] < metrics["noise"]
    assert metrics["their_margin"] < metrics["noise"]


def test_a_confident_disagreement_fails():
    """The failure mode the gate exists for: a model with the wrong head_dim or norm
    convention does not produce near-ties, it produces confident wrong answers."""
    ours = torch.tensor([0.0, 30.0, 1.0])
    theirs = torch.tensor([0.0, 1.0, 30.0])
    verdict, metrics = classify_step(ours, theirs)
    assert verdict == "disagree", metrics


def test_one_model_being_confident_is_enough_to_fail():
    """Both margins must be inside the noise. If we are torn and HuggingFace is certain,
    that is HuggingFace telling us we are wrong -- not a coin landing differently."""
    # Ours is nearly tied between 1 and 2; theirs prefers 2 by a mile.
    ours = torch.tensor([0.0, 5.01, 5.00])
    theirs = torch.tensor([0.0, 1.00, 20.00])
    verdict, metrics = classify_step(ours, theirs)
    assert verdict == "disagree", metrics
    assert metrics["our_margin"] < metrics["noise"], "our side alone would have looked fine"
    assert metrics["their_margin"] > metrics["noise"]


def test_the_rule_is_symmetric_in_which_model_is_confident():
    ours = torch.tensor([0.0, 20.00, 1.00])
    theirs = torch.tensor([0.0, 5.00, 5.01])
    verdict, _ = classify_step(ours, theirs)
    assert verdict == "disagree"


def test_a_margin_exactly_at_the_noise_floor_fails_closed():
    """The comparison is strict `<`. A margin that merely equals the noise is not evidence
    of a tie, and a gate this load-bearing should refuse rather than round in its own
    favour."""
    ours = torch.tensor([0.0, 1.0, 0.0])
    theirs = torch.tensor([0.0, 0.0, 1.0])
    verdict, metrics = classify_step(ours, theirs)
    assert metrics["our_margin"] == metrics["noise"] == 1.0
    assert verdict == "disagree"


def test_bf16_rows_are_accepted_and_compared_in_fp32():
    """The real rows arrive as bf16. Comparing them in bf16 would quantise the very
    margins the rule is measuring."""
    ours = torch.tensor([0.0, 5.05, 4.95], dtype=torch.bfloat16)
    theirs = torch.tensor([0.5, 4.95, 5.05], dtype=torch.bfloat16)
    verdict, metrics = classify_step(ours, theirs)
    assert verdict in {"tie", "agree", "disagree"}
    assert isinstance(metrics["our_margin"], float)


def test_the_metrics_report_one_bf16_ulp_against_the_oracle_scale():
    """8 mantissa bits, so one ULP is the scale over 128. The number the writeup quotes."""
    ours = torch.tensor([0.0, 32.0, 30.0])
    theirs = torch.tensor([0.0, 30.0, 32.0])
    _, metrics = classify_step(ours, theirs)
    assert metrics["one_bf16_ulp"] == 32.0 / 128
