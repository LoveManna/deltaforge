"""Bytes moved per decode token, as a function of how the weights are stored.

``docs/roofline.py`` has always computed this and printed it. It is importable now because
the benchmark needs it: a ratio says a kernel lost, and only bytes-over-time says whether it
lost on bandwidth or on instruction issue. Batch 003 turned on exactly that distinction and
had to derive it by hand from two files after the rental was over.

The weight figures come from the published checkpoint's tensor manifest rather than from
shapes re-derived here, so the ceiling this prints is the one the card will actually stream.

Units are SI throughout — MB is 1e6 bytes and GB/s is 1e9 bytes per second — because the
vendor bandwidth every result is scored against (1792 GB/s on an RTX 5090) is quoted that
way. Batch 003's writeup mixed SI megabytes with binary gigabytes per second and reported
achieved bandwidths 2.4% low as a result; nothing downstream of it was a ratio, so no
verdict moved, but the numbers in that table are not the ones this module produces.

No torch. This is arithmetic over a ``ModelConfig`` and a JSON manifest, and it is tested
on a CPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import MODELS, ModelConfig
from ..weights import load_manifest, map_checkpoint_name

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "WEIGHT_REGIONS",
    "decode_bytes_per_token",
    "traffic",
    "validate_weight_bits",
    "weight_bytes",
]

#: Bytes per element, by the dtype string a safetensors header uses.
BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8}

#: The regions a hypothesis may re-encode. ``head`` is the tied embedding matrix when the
#: checkpoint ties it and the separate ``lm_head`` when it does not: either way it is the
#: tensor the final projection streams. Norm weights are 0.333 MB of 8411 and nothing
#: quantises a norm scale, so they are deliberately absent: there is no bit width a
#: manifest could usefully set for them.
#:
#: ``linear_attn_gates`` is ``in_proj_a`` and ``in_proj_b`` — 48 projections **32 channels
#: wide**, 7.86 MB/token between them, and the only sites in this model that no tile can
#: give parallelism to. They are their own region because batch 006 is the first batch to
#: quantise the layer projections *without* them, and a manifest that could not say so
#: would have to credit a candidate with a saving it never collected.
WEIGHT_REGIONS = ("mlp", "linear_attn", "linear_attn_gates", "full_attn", "head")

#: ``layers`` is an alias, not a region. Every batch-004 slot that quantises "the layer
#: projections" means all of them, and spelling them out in each manifest invites one to
#: drift from the others. The gates are included here so the alias keeps exactly the
#: meaning it had for batches 003 and 004, whose records were measured with it.
_LAYER_REGIONS = ("mlp", "linear_attn", "linear_attn_gates", "full_attn")


def _IS_GATE_PROJECTION(name: str) -> bool:  # noqa: N802 - a predicate, named like the constant it guards
    """``in_proj_a`` / ``in_proj_b``: the two 32-channel projections per linear-attn layer."""
    return ".in_proj_a." in name or ".in_proj_b." in name


def _tensor_bytes(meta: dict) -> int:
    n = 1
    for dim in meta["shape"]:
        n *= dim
    return n * BYTES.get(meta["dtype"], 2)


def weight_bytes(repo_id: str) -> dict[str, int]:
    """Decode-path weight bytes, grouped. Vision tower and MTP head are excluded."""
    groups: dict[str, int] = {}
    for name, meta in load_manifest(repo_id)["tensors"].items():
        if map_checkpoint_name(name) is None:
            continue  # vision tower or MTP head: not in the decode path
        size = _tensor_bytes(meta)
        if "embed_tokens" in name:
            key = "embeddings"
        elif name.startswith("lm_head"):
            key = "lm_head"
        elif ".mlp." in name:
            key = "mlp"
        elif "linear_attn" in name:
            key = "linear_attn_gates" if _IS_GATE_PROJECTION(name) else "linear_attn"
        elif "self_attn" in name:
            key = "full_attn"
        else:
            key = "norms"
        groups[key] = groups.get(key, 0) + size
    return groups


def traffic(config: ModelConfig, groups: dict[str, int], batch: int, context: int) -> dict[str, int]:
    """Bytes moved per decoded token, by what moves them."""
    hidden, inter = config.hidden_size, config.intermediate_size
    layers = config.num_hidden_layers
    full = len(config.layer_indices("full_attention"))
    linear = len(config.layer_indices("linear_attention"))
    bf = 2

    weights = sum(groups.values())
    if config.tie_word_embeddings:
        # One tensor serves both; it is read in full for the lm_head matmul, and the
        # embedding lookup reads a single row. Counting it once is correct.
        pass
    else:
        # The embedding matrix is read one row per token, not in full.
        weights -= groups.get("embeddings", 0)

    kv = full * 2 * config.num_key_value_heads * context * config.head_dim * bf * batch
    # `repeat_interleave` to expand KV heads to query heads materialises `groups` copies,
    # written then read back by the attention matmul.
    expand = kv * config.num_key_value_groups * 2
    state = (
        linear
        * config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * 4  # fp32, fixed by mamba_ssm_dtype
        * 2  # read + write
        * batch
    )
    # Two residual+norm pairs per layer: add reads x and residual and writes residual,
    # then the norm reads and writes.
    norms = 2 * layers * (hidden * bf * 3 + hidden * bf * 2) * batch
    swiglu = layers * 4 * inter * bf * batch
    qkv = (
        full
        * 2
        * (2 * config.num_attention_heads + 2 * config.num_key_value_heads)
        * config.head_dim
        * bf
        * batch
    )
    return {
        "weights, streamed once": weights,
        "GQA repeat_interleave materialisation": expand,
        "recurrent state, read + write": state,
        "KV cache read": kv,
        "SwiGLU intermediates": swiglu,
        "norm + residual": norms,
        "QKV / RoPE intermediates": qkv,
    }


#: The fields that decide which tensor manifest a config describes. Everything here
#: changes a tensor shape; nothing here is a naming or provenance detail.
_SHAPE_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "tie_word_embeddings",
)


def _shape_of(config: ModelConfig) -> tuple:
    return tuple(getattr(config, field) for field in _SHAPE_FIELDS)


def _repo_for(config: ModelConfig) -> str:
    """The published checkpoint a config describes, matched on shape.

    ``ModelConfig`` does not carry a repo id — it is the shape of a model, not a pointer to
    one — but the manifest is keyed by repo, so the lookup lives here rather than in every
    caller.

    **Matched on shape, not on ``name``.** A rental's config comes from the checkpoint's own
    ``config.json`` via `from_hf_config`, which sets ``name`` from ``model_type`` —
    ``'qwen3_5'``, never the ``'qwen3.5-4b'`` the transcription in `MODELS` carries. So a
    name match resolved the only configs that never need resolving and failed on every one
    that does: rental 38 reported "no byte model" on all seven slots and shipped no
    bandwidth at all. The shape fields are what the manifest is actually about.
    """
    wanted = _shape_of(config)
    for repo_id, factory in MODELS.items():
        if _shape_of(factory()) == wanted:
            return repo_id
    raise ValueError(
        f"no tensor manifest for a config shaped like {config.name!r} "
        f"(hidden {config.hidden_size}, {config.num_hidden_layers} layers, "
        f"vocab {config.vocab_size}); known: "
        f"{sorted(factory().name for factory in MODELS.values())}"
    )


def _region_group(config: ModelConfig, region: str) -> str:
    """The manifest group a region names, for this checkpoint's tying convention."""
    if region != "head":
        return region
    return "embeddings" if config.tie_word_embeddings else "lm_head"


def _expand(weight_bits: Mapping[str, int]) -> dict[str, int]:
    """Region -> stored bit width, with ``layers`` resolved and everything else defaulting to 16.

    Raises rather than ignoring an unknown key. A manifest typo that silently scored a
    candidate against bf16 byte counts would look exactly like a kernel that saved nothing.
    """
    bits = dict.fromkeys(WEIGHT_REGIONS, 16)
    for name, width in weight_bits.items():
        if name == "layers":
            targets = _LAYER_REGIONS
        elif name in WEIGHT_REGIONS:
            targets = (name,)
        else:
            raise ValueError(f"unknown weight region {name!r}; known: {[*WEIGHT_REGIONS, 'layers']}")
        for target in targets:
            bits[target] = width
    return bits


def validate_weight_bits(weight_bits: Mapping[str, int]) -> None:
    """Raise on an unknown region name. `Hypothesis` calls this so a typo in a manifest
    fails on a laptop rather than scoring a candidate against the wrong byte count."""
    _expand(weight_bits)


#: The traffic row `max-autotune` does not move. Rental 38's `TORCH_LOGS=output_code` dump
#: shows inductor folding the GQA head expansion into index arithmetic — the attention
#: `bmm` reads `in_ptr1 + (r0_2 + 256*x0 + 557056*(x1 // 4))` straight out of the
#: unexpanded KV cache — so nothing is materialised and nothing is read back.
#:
#: `traffic` keeps the row because the **eager** reference really does perform that copy,
#: and `docs/roofline.py` documents eager. Both *scored* columns are compiled, so both fold
#: it, and dividing their time by a byte count that includes it understated the achieved
#: bandwidth of every column this project has reported: the compiled baseline came out at
#: 73% of a 5090's peak when the honest figure is 65.7%.
_FOLDED_BY_INDUCTOR = "GQA repeat_interleave materialisation"


def decode_bytes_per_token(
    config: ModelConfig,
    *,
    weight_bits: Mapping[str, int] | None = None,
    context_length: int = 2048,
    decode_tokens: int = 128,
    batch: int = 1,
    repo_id: str | None = None,
    compiled: bool = True,
) -> float:
    """MB moved per decoded token, with each weight region stored at ``weight_bits`` bits.

    ``context_length`` is the prompt, and the KV cache is read over
    ``context_length + decode_tokens`` positions — the same convention ``docs/roofline.py``
    prints under, so the two agree by construction rather than by coincidence.

    ``compiled`` (the default, because both scored columns are compiled) drops the GQA
    expansion row that inductor folds away. Pass ``False`` for what eager moves.
    """
    groups = dict(weight_bytes(repo_id or _repo_for(config)))
    bits = _expand(weight_bits or {})
    for region, width in bits.items():
        group = _region_group(config, region)
        if group in groups:
            groups[group] = int(round(groups[group] * width / 16.0))
    rows = traffic(config, groups, batch, context_length + decode_tokens)
    if compiled:
        rows.pop(_FOLDED_BY_INDUCTOR, None)
    return sum(rows.values()) / 1e6
