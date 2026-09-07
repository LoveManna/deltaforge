"""Config tests, including the architecture facts that a kernel must honour.

These pin the details that are easy to get wrong and expensive to debug on rented time:
head_dim is not hidden/heads, RoPE is partial, the layer schedule is 3:1, and the linear
attention layers have twice as many value heads as key heads.
"""

from __future__ import annotations

import json

import pytest

from .config import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    SIGMOID_GATE,
    SWISH_GATE,
    ModelConfig,
    from_hf_config,
    qwen3_5_4b_config,
    qwen3_8_27b_config,
    tiny_config,
)
from .weights import load_manifest


def _minimal_text_config() -> dict:
    """The smallest `text_config` `from_hf_config` accepts, for gate-reading tests."""
    return {
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 4,
        "vocab_size": 248320,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rope_parameters": {
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    }


@pytest.fixture(scope="module")
def config():
    return qwen3_5_4b_config()


# -- the real model -------------------------------------------------------------------


def test_layer_schedule_is_three_linear_then_one_full(config):
    assert config.num_hidden_layers == 32
    assert len(config.layer_types) == 32
    # 8 repetitions of [linear, linear, linear, full].
    assert config.layer_types == ((LINEAR_ATTENTION, LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION) * 8)
    assert len(config.layer_indices(FULL_ATTENTION)) == 8
    assert len(config.layer_indices(LINEAR_ATTENTION)) == 24
    assert config.layer_indices(FULL_ATTENTION) == (3, 7, 11, 15, 19, 23, 27, 31)


def test_head_dim_is_not_hidden_size_over_num_heads(config):
    """The single most likely early bug: head_dim is 256, hidden/heads is 160."""
    assert config.head_dim == 256
    assert config.hidden_size // config.num_attention_heads == 160
    assert config.head_dim != config.hidden_size // config.num_attention_heads


def test_gqa_ratio_is_four_to_one(config):
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 4
    assert config.num_key_value_groups == 4


def test_rope_is_partial_over_a_quarter_of_the_head(config):
    assert config.partial_rotary_factor == 0.25
    assert config.rotary_dim == 64
    assert config.head_dim - config.rotary_dim == 192, "192 dims pass through unrotated"


def test_mrope_sections_sum_to_half_the_rotary_dim(config):
    assert config.mrope_section == (11, 11, 10)
    assert sum(config.mrope_section) == config.rotary_dim // 2 == 32
    assert config.mrope_interleaved is True
    assert config.rope_theta == 1e7


def test_linear_attention_dims(config):
    assert config.linear_num_key_heads == 16
    assert config.linear_num_value_heads == 32
    assert config.linear_value_groups == 2, "two value heads share each key head"
    assert config.linear_key_dim == 2048
    assert config.linear_value_dim == 4096
    # q, k and v all flow through the short causal depthwise conv.
    assert config.linear_conv_dim == 2048 + 2048 + 4096 == 8192
    assert config.linear_conv_kernel_dim == 4


def test_ffn_is_dense(config):
    """Spec section 16 left this open; the published config closes it. `mlp_only_layers`
    is empty and there is no expert, router or MoE key anywhere in text_config, so
    hypothesis 7 (fused MoE routing) is retired."""
    assert config.intermediate_size == 9216
    # The decisive evidence is the checkpoint itself: a sparse FFN would carry expert or
    # router tensors, and there are none.
    names = load_manifest()["tensors"]
    assert not any("expert" in name or "router" in name or "gate.weight" in name for name in names)
    # Every layer has exactly the three dense SwiGLU projections.
    mlp = [n for n in names if ".mlp." in n and n.startswith("model.language_model.")]
    assert len(mlp) == 3 * 32


def test_embeddings_are_tied(config):
    assert config.tie_word_embeddings
    assert "lm_head.weight" not in load_manifest()["tensors"]


def test_recurrent_state_is_float32(config):
    assert config.mamba_ssm_dtype == "float32"


# -- validation -----------------------------------------------------------------------


def test_layer_types_length_must_match_layer_count():
    with pytest.raises(ValueError, match="layer_types has 2 entries"):
        ModelConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=3,
            vocab_size=10,
            layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        )


def test_unknown_layer_type_is_rejected():
    with pytest.raises(ValueError, match="unknown layer types"):
        ModelConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            vocab_size=10,
            layer_types=("sliding_window",),
        )


def test_mrope_section_must_match_the_rotary_dim():
    with pytest.raises(ValueError, match="expected rotary_dim // 2"):
        ModelConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            vocab_size=32,
            layer_types=(FULL_ATTENTION,),
            head_dim=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            partial_rotary_factor=0.25,
            mrope_section=(5, 5, 5),
        )


def test_gqa_ratio_must_divide_evenly():
    with pytest.raises(ValueError, match="multiple of num_key_value_heads"):
        ModelConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            vocab_size=32,
            layer_types=(FULL_ATTENTION,),
            head_dim=32,
            num_attention_heads=5,
            num_key_value_heads=2,
            mrope_section=(2, 1, 1),
        )


def test_non_float32_recurrent_state_is_rejected():
    with pytest.raises(ValueError, match="only float32 recurrent state"):
        ModelConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            vocab_size=32,
            layer_types=(FULL_ATTENTION,),
            head_dim=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            mrope_section=(2, 1, 1),
            mamba_ssm_dtype="bfloat16",
        )


# -- the tiny fixture -----------------------------------------------------------------


def test_tiny_config_mirrors_the_real_structural_ratios():
    tiny = tiny_config()
    real = qwen3_5_4b_config()

    assert tiny.num_key_value_groups == 2, "GQA is exercised, at a smaller ratio"
    assert tiny.linear_value_groups == real.linear_value_groups == 2
    assert tiny.partial_rotary_factor == real.partial_rotary_factor
    assert tiny.rotary_dim < tiny.head_dim, "partial RoPE is exercised"
    assert tiny.head_dim * tiny.num_attention_heads != tiny.hidden_size
    assert tiny.attn_output_gate == real.attn_output_gate
    assert set(tiny.layer_types) == set(real.layer_types)
    assert tiny.linear_conv_kernel_dim == real.linear_conv_kernel_dim


# -- parsing a real config.json -------------------------------------------------------


def test_from_hf_config_reads_text_config_only(tmp_path):
    raw = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "vision_config": {"depth": 24, "hidden_size": 1024, "out_hidden_size": 2560},
        "text_config": {
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "num_hidden_layers": 4,
            "vocab_size": 248320,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
            "rope_parameters": {
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))

    config = from_hf_config(path)

    assert config.hidden_size == 2560
    assert config.head_dim == 256
    assert config.rotary_dim == 64
    assert config.num_hidden_layers == 4
    # The vision tower never reaches the text config.
    assert not hasattr(config, "vision_config")
    # Unmodelled text keys are preserved rather than silently dropped, so a new one is
    # visible instead of invisible.
    assert config.extra["mtp_num_hidden_layers"] == 1


# -- the attention output gate ----------------------------------------------------------
#
# Qwen3.5 omits `output_gate_type` and means sigmoid; Qwen3.8 declares "swish". Reading
# the wrong one produces a model that runs and emits plausible logits, which is the class
# of bug this project is least able to detect, so it is pinned here rather than assumed.


def test_a_checkpoint_without_output_gate_type_means_sigmoid(tmp_path):
    raw = {"text_config": _minimal_text_config()}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))

    assert from_hf_config(path).output_gate_type == SIGMOID_GATE


def test_a_checkpoint_declaring_swish_is_read_as_swish(tmp_path):
    raw = {"text_config": {**_minimal_text_config(), "output_gate_type": "swish"}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))

    assert from_hf_config(path).output_gate_type == SWISH_GATE


def test_an_unknown_gate_type_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="output_gate_type"):
        ModelConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            vocab_size=512,
            layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            output_gate_type="gelu",
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            partial_rotary_factor=0.25,
            mrope_section=(2, 1, 1),
        )


def test_the_two_shipped_models_disagree_about_the_gate():
    """If these ever match, one of them has been transcribed wrongly."""
    assert qwen3_5_4b_config().output_gate_type == SIGMOID_GATE
    assert qwen3_8_27b_config().output_gate_type == SWISH_GATE
