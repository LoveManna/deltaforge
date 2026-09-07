"""Text-decode configuration for the Qwen3.5/3.8 hybrid-attention family.

Only the ``text_config`` half of the checkpoint's ``config.json`` is modelled here.
The vision tower and the multi-token-prediction head are deliberately outside the
decode path (see README, "What is and is not benchmarked").

Two checkpoints are supported. They share one architecture (``model_type: qwen3_5``),
so the same :mod:`deltaforge.reference` serves both:

* ``Qwen/Qwen3.5-4B`` — **the benchmark target.** 32 layers, 8.4 GB of text decode
  weights, tied embeddings, a **sigmoid** attention output gate. Small enough that the
  checkpoint downloads in minutes and a bf16 baseline plus a quantised candidate both
  fit on one 32 GB card, which is what the interleaved A/B/A protocol requires.
* ``Qwen/Qwen3.8-27B`` — verified, **not the target.** 64 layers, 53.8 GB, untied
  embeddings, a **swish** output gate. Present because a session asked "what about the
  latest Qwen?" and the answer is worth not re-deriving: Qwen3.8 ships no small
  checkpoint (27B, a 2.4T MoE, and a 360 GB Flash-Next), and Qwen3.6 is 27B/35B-A3B
  only, so the newest *small* Qwen is 3.5. All 851 of this config's decode parameters
  were checked against the published safetensors headers on 2026-09-04 and match, so a
  future session with an 80 GB card can select it with ``--model`` and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

SIGMOID_GATE = "sigmoid"
SWISH_GATE = "swish"
OUTPUT_GATE_TYPES = (SIGMOID_GATE, SWISH_GATE)


@dataclass(frozen=True)
class ModelConfig:
    """The subset of a checkpoint's ``text_config`` that the decode path needs.

    Field names mirror the checkpoint's keys so that a diff against ``config.json``
    is readable. Values for the published models live in :data:`MODELS`.
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    layer_types: tuple[str, ...]
    rms_norm_eps: float = 1e-6

    # Full-attention (gated attention) layers.
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    attn_output_gate: bool = True
    attention_bias: bool = False
    #: How the attention output gate is applied. Qwen3.5 omits the key and means
    #: ``sigmoid``; Qwen3.8 declares ``swish``. Reading the wrong one produces a model
    #: that runs and emits plausible logits, so it is a config field rather than a
    #: constant in the reference.
    output_gate_type: str = SIGMOID_GATE

    # Linear-attention (Gated DeltaNet) layers.
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # RoPE. ``mrope_section`` entries sum to rotary_dim // 2.
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, ...] = (11, 11, 10)
    mrope_interleaved: bool = True
    max_position_embeddings: int = 262144

    tie_word_embeddings: bool = True
    hidden_act: str = "silu"

    # The recurrent state stays fp32 even though the weights are bf16
    # (``mamba_ssm_dtype`` in the checkpoint). Any future scan kernel must honour this.
    mamba_ssm_dtype: str = "float32"

    # Provenance, so a results record can say which config produced it.
    name: str = "qwen3.5-4b"
    extra: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries "
                f"but num_hidden_layers is {self.num_hidden_layers}"
            )
        unknown = set(self.layer_types) - {LINEAR_ATTENTION, FULL_ATTENTION}
        if unknown:
            raise ValueError(f"unknown layer types: {sorted(unknown)}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        if self.rotary_dim % 2:
            raise ValueError(f"rotary_dim must be even, got {self.rotary_dim}")
        if sum(self.mrope_section) != self.rotary_dim // 2:
            raise ValueError(
                f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)}, "
                f"expected rotary_dim // 2 = {self.rotary_dim // 2}"
            )
        if self.mamba_ssm_dtype != "float32":
            raise ValueError("only float32 recurrent state is supported")
        if self.output_gate_type not in OUTPUT_GATE_TYPES:
            raise ValueError(
                f"unknown output_gate_type {self.output_gate_type!r}; known: {list(OUTPUT_GATE_TYPES)}"
            )

    # -- derived shapes -------------------------------------------------------

    @property
    def rotary_dim(self) -> int:
        """Number of head dimensions RoPE actually rotates.

        ``partial_rotary_factor`` is 0.25 on this model: only the first 64 of each
        256-wide head is rotated, the remaining 192 pass through untouched.
        """
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        """Channels flowing through the short causal depthwise conv: q, k and v."""
        return 2 * self.linear_key_dim + self.linear_value_dim

    @property
    def linear_value_groups(self) -> int:
        """How many value heads share one key head."""
        return self.linear_num_value_heads // self.linear_num_key_heads

    def layer_indices(self, layer_type: str) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == layer_type)


def _hybrid_layer_types(num_layers: int, full_attention_interval: int = 4) -> tuple[str, ...]:
    """The 3:1 linear/full schedule both checkpoints use: every 4th layer is full."""
    return tuple(
        LINEAR_ATTENTION if (i + 1) % full_attention_interval else FULL_ATTENTION for i in range(num_layers)
    )


def qwen3_8_27b_config() -> ModelConfig:
    """Not the target — see the module docstring. From ``Qwen/Qwen3.8-27B``.

    Verified against the published checkpoint on 2026-09-04: all 851 decode parameters
    match the safetensors headers in name and shape, with nothing missing and nothing
    unmapped. See ``docs/ARCHITECTURE.md``.

    Differs from Qwen3.5-4B in ways that matter: **untied embeddings** (a separate
    2.54 GB ``lm_head``), a **swish** output gate rather than sigmoid, 64 layers, and
    48 linear value heads sharing 16 key heads.
    """
    return ModelConfig(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        vocab_size=248320,
        layer_types=_hybrid_layer_types(64),
        rms_norm_eps=1e-6,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        attn_output_gate=True,
        output_gate_type=SWISH_GATE,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rope_theta=1e7,
        partial_rotary_factor=0.25,
        mrope_section=(11, 11, 10),
        mrope_interleaved=True,
        max_position_embeddings=262144,
        tie_word_embeddings=False,
        name="qwen3.8-27b",
    )


def qwen3_5_4b_config() -> ModelConfig:
    """**The benchmark target**, transcribed from ``Qwen/Qwen3.5-4B`` ``config.json``.

    Verified against the published checkpoint on 2026-08-30; see ``docs/ARCHITECTURE.md``
    for the tensor shapes this was cross-checked against.
    """
    return ModelConfig(
        hidden_size=2560,
        intermediate_size=9216,
        num_hidden_layers=32,
        vocab_size=248320,
        layer_types=_hybrid_layer_types(32),
        rms_norm_eps=1e-6,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        attn_output_gate=True,
        output_gate_type=SIGMOID_GATE,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rope_theta=1e7,
        partial_rotary_factor=0.25,
        mrope_section=(11, 11, 10),
        mrope_interleaved=True,
        max_position_embeddings=262144,
        tie_word_embeddings=True,
        name="qwen3.5-4b",
    )


def tiny_config() -> ModelConfig:
    """A 2-layer stand-in used by every CPU test.

    One linear-attention layer and one full-attention layer, which is the smallest
    configuration that still exercises the real layer schedule, the GQA ratio, the
    linear-attention key/value head ratio, partial RoPE and both cache kinds.
    This is a first-class fixture, not a throwaway: the CPU test suite proves shapes,
    causality, the cache contract and step-vs-chunk equivalence against it.
    """
    return ModelConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        vocab_size=512,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        rms_norm_eps=1e-6,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        attn_output_gate=True,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        rope_theta=1e4,
        partial_rotary_factor=0.25,
        # rotary_dim = 8, so the sections must sum to 4.
        mrope_section=(2, 1, 1),
        mrope_interleaved=True,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        name="tiny",
    )


#: Published checkpoint -> the config transcribed from it.
#:
#: A session normally passes ``--weights`` and the config is read from the checkpoint's
#: own ``config.json`` by :func:`from_hf_config`. This table exists so the CPU test suite
#: can assert the transcription against the tensor manifest without a download, and so
#: ``--model`` has something to name.
MODELS: dict[str, Callable[[], ModelConfig]] = {
    "Qwen/Qwen3.5-4B": qwen3_5_4b_config,
    "Qwen/Qwen3.8-27B": qwen3_8_27b_config,
}

#: What a session benchmarks unless it says otherwise. Kept small on purpose: the
#: benchmark holds a bf16 baseline and a candidate in one process on one card.
DEFAULT_REPO_ID = "Qwen/Qwen3.5-4B"


def model_config(repo_id: str = DEFAULT_REPO_ID) -> ModelConfig:
    try:
        return MODELS[repo_id]()
    except KeyError:
        raise ValueError(f"unknown model {repo_id!r}; known: {sorted(MODELS)}") from None


def from_hf_config(path: str | Path) -> ModelConfig:
    """Build a :class:`ModelConfig` from a HuggingFace ``config.json``.

    Reads ``text_config`` only. Any vision or MTP keys present are ignored on
    purpose and reported by :func:`deltaforge.weights.load_weights`.
    """
    raw = json.loads(Path(path).read_text())
    text = raw.get("text_config", raw)
    rope = text.get("rope_parameters", {})

    layer_types = tuple(text["layer_types"])
    known = {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "vocab_size",
        "layer_types",
        "rms_norm_eps",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "attn_output_gate",
        "attention_bias",
        "output_gate_type",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "max_position_embeddings",
        "tie_word_embeddings",
        "hidden_act",
        "mamba_ssm_dtype",
        "rope_parameters",
    }
    return ModelConfig(
        hidden_size=text["hidden_size"],
        intermediate_size=text["intermediate_size"],
        num_hidden_layers=text["num_hidden_layers"],
        vocab_size=text["vocab_size"],
        layer_types=layer_types,
        rms_norm_eps=text.get("rms_norm_eps", 1e-6),
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        attn_output_gate=text.get("attn_output_gate", True),
        attention_bias=text.get("attention_bias", False),
        # Absent means sigmoid: Qwen3.5 predates the key, Qwen3.8 declares "swish".
        output_gate_type=text.get("output_gate_type", SIGMOID_GATE),
        linear_num_key_heads=text["linear_num_key_heads"],
        linear_num_value_heads=text["linear_num_value_heads"],
        linear_key_head_dim=text["linear_key_head_dim"],
        linear_value_head_dim=text["linear_value_head_dim"],
        linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
        rope_theta=rope.get("rope_theta", 1e7),
        partial_rotary_factor=rope.get("partial_rotary_factor", 1.0),
        mrope_section=tuple(rope.get("mrope_section", ())),
        mrope_interleaved=rope.get("mrope_interleaved", False),
        max_position_embeddings=text.get("max_position_embeddings", 262144),
        tie_word_embeddings=text.get("tie_word_embeddings", True),
        hidden_act=text.get("hidden_act", "silu"),
        mamba_ssm_dtype=text.get("mamba_ssm_dtype", "float32"),
        name=raw.get("model_type", "qwen3_5"),
        extra={k: v for k, v in text.items() if k not in known},
    )
