"""Load the HuggingFace safetensors checkpoint into :class:`~deltaforge.reference.ReferenceModel`.

The name map is explicit rather than heuristic, and anything that does not map is a
loud failure: a silently skipped tensor produces a model that runs, produces plausible
logits, and is wrong — the worst possible outcome for a project whose entire claim rests
on the reference being correct.

Two families of tensors are skipped **on purpose**, and both are reported by name count
so the skip is visible rather than assumed:

* ``model.visual.*`` — the vision tower. Text-only decode never instantiates it.
* ``mtp.*`` — the multi-token-prediction head. Excluded from the decode path and from
  the benchmark; see the README for why including it would make the headline number
  incomparable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DEFAULT_REPO_ID

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from .config import ModelConfig
    from .reference import ReferenceModel

__all__ = [
    "LoadReport",
    "WeightMappingError",
    "load_manifest",
    "manifest_path",
    "load_weights",
    "map_checkpoint_name",
]

TEXT_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."
MTP_PREFIX = "mtp."

DATA_DIR = Path(__file__).parent / "data"


def manifest_path(repo_id: str) -> Path:
    """Where the tensor manifest for a published checkpoint lives."""
    slug = repo_id.split("/")[-1].replace(".", "_").replace("-", "_").lower()
    return DATA_DIR / f"{slug}_manifest.json"


# Checkpoint suffix (below `model.language_model.layers.<i>.`) -> reference suffix
# (below `layers.<i>.`). Identity for most of them; kept explicit so that a rename on
# either side fails loudly instead of silently falling through to a heuristic.
_LAYER_SUFFIX_MAP = {
    "input_layernorm.weight": "input_layernorm.weight",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
    # Full-attention layers.
    "self_attn.q_proj.weight": "self_attn.q_proj.weight",
    "self_attn.k_proj.weight": "self_attn.k_proj.weight",
    "self_attn.v_proj.weight": "self_attn.v_proj.weight",
    "self_attn.o_proj.weight": "self_attn.o_proj.weight",
    "self_attn.q_norm.weight": "self_attn.q_norm.weight",
    "self_attn.k_norm.weight": "self_attn.k_norm.weight",
    # Linear-attention (Gated DeltaNet) layers.
    "linear_attn.in_proj_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight": "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight": "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight": "linear_attn.in_proj_a.weight",
    "linear_attn.conv1d.weight": "linear_attn.conv1d.weight",
    "linear_attn.A_log": "linear_attn.A_log",
    "linear_attn.dt_bias": "linear_attn.dt_bias",
    "linear_attn.norm.weight": "linear_attn.norm.weight",
    "linear_attn.out_proj.weight": "linear_attn.out_proj.weight",
}

_TOP_LEVEL_MAP = {
    f"{TEXT_PREFIX}embed_tokens.weight": "embed_tokens.weight",
    f"{TEXT_PREFIX}norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",  # absent when embeddings are tied
}

_LAYER_RE = re.compile(re.escape(TEXT_PREFIX) + r"layers\.(\d+)\.(.+)$")


class WeightMappingError(RuntimeError):
    """Raised when the checkpoint and the reference model do not line up."""


@dataclass(frozen=True)
class LoadReport:
    """What happened during a load. Every field is a count or a list of names."""

    loaded: tuple[str, ...] = ()
    skipped_vision: tuple[str, ...] = ()
    skipped_mtp: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unmapped: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unmapped or self.mismatched)

    def summary(self) -> str:
        lines = [
            f"loaded {len(self.loaded)} tensors",
            f"skipped {len(self.skipped_vision)} vision-tower tensors (excluded by design)",
            f"skipped {len(self.skipped_mtp)} MTP-head tensors (excluded by design)",
        ]
        for label, names in (
            ("missing from checkpoint", self.missing),
            ("unmapped in checkpoint", self.unmapped),
            ("shape/dtype mismatch", self.mismatched),
        ):
            if names:
                shown = ", ".join(names[:8])
                more = f" (+{len(names) - 8} more)" if len(names) > 8 else ""
                lines.append(f"{len(names)} {label}: {shown}{more}")
        return "\n".join(lines)


def map_checkpoint_name(name: str) -> str | None:
    """Checkpoint tensor name -> reference parameter name.

    Returns ``None`` for tensors that are deliberately not part of the decode path
    (vision tower, MTP head). Raises :class:`WeightMappingError` for a
    ``model.language_model.*`` tensor that is not recognised — a new tensor appearing
    inside the text decoder is a real change we must not paper over.
    """
    if name.startswith(VISION_PREFIX) or name.startswith(MTP_PREFIX):
        return None
    if name in _TOP_LEVEL_MAP:
        return _TOP_LEVEL_MAP[name]

    match = _LAYER_RE.match(name)
    if match is None:
        raise WeightMappingError(f"unrecognised checkpoint tensor: {name!r}")
    layer_idx, suffix = match.group(1), match.group(2)
    mapped = _LAYER_SUFFIX_MAP.get(suffix)
    if mapped is None:
        raise WeightMappingError(f"unrecognised tensor {name!r}: no mapping for layer suffix {suffix!r}")
    return f"layers.{layer_idx}.{mapped}"


@lru_cache(maxsize=4)
def load_manifest(repo_id: str = DEFAULT_REPO_ID) -> dict:
    """A published checkpoint's tensor names, dtypes and shapes — headers, no weights.

    It exists so the CPU test suite can prove the name map and every reference
    parameter shape against the real checkpoint without a 54 GB download. Regenerate
    with ``python -m deltaforge.data.make_manifest Qwen/<model>``.
    """
    path = manifest_path(repo_id)
    if not path.exists():
        raise WeightMappingError(f"no tensor manifest for {repo_id!r} at {path}")
    return json.loads(path.read_text())


def expected_parameter_shapes(config: ModelConfig, repo_id: str = DEFAULT_REPO_ID) -> dict[str, list[int]]:
    """Reference parameter name -> shape, derived from the checkpoint manifest."""
    shapes: dict[str, list[int]] = {}
    for name, meta in load_manifest(repo_id)["tensors"].items():
        mapped = map_checkpoint_name(name)
        if mapped is not None:
            shapes[mapped] = meta["shape"]
    if config.tie_word_embeddings:
        shapes.pop("lm_head.weight", None)
    return shapes


def _iter_checkpoint_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return [path / shard for shard in sorted(set(weight_map.values()))]
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files under {path}")
    return shards


def load_weights(
    model: ReferenceModel,
    path: str | Path,
    *,
    strict: bool = True,
    dtype: torch.dtype | None = None,
    verbose: bool = True,
) -> LoadReport:
    """Load a checkpoint directory (or a single shard) into ``model`` in place.

    Raises :class:`WeightMappingError` when ``strict`` and anything is missing,
    unmapped, or the wrong shape.
    """
    import torch  # noqa: PLC0415 - deferred so the module imports without torch present
    from safetensors import safe_open  # noqa: PLC0415

    path = Path(path)
    # Parameters only. Buffers here (the RoPE inverse frequencies) are computed from the
    # config, not stored in the checkpoint; treating them as load destinations would
    # report them as permanently "missing".
    destinations = dict(model.named_parameters())

    loaded: list[str] = []
    skipped_vision: list[str] = []
    skipped_mtp: list[str] = []
    unmapped: list[str] = []
    mismatched: list[str] = []
    seen: set[str] = set()

    for shard in _iter_checkpoint_files(path):
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118 - safe_open has no __contains__
                if name.startswith(VISION_PREFIX):
                    skipped_vision.append(name)
                    continue
                if name.startswith(MTP_PREFIX):
                    skipped_mtp.append(name)
                    continue

                try:
                    target = map_checkpoint_name(name)
                except WeightMappingError:
                    unmapped.append(name)
                    continue
                if target is None:  # pragma: no cover - covered by the prefix checks
                    continue
                if target not in destinations:
                    # e.g. lm_head.weight when embeddings are tied.
                    unmapped.append(name)
                    continue

                tensor = handle.get_tensor(name)
                param = destinations[target]
                if tuple(tensor.shape) != tuple(param.shape):
                    mismatched.append(
                        f"{name}: checkpoint {tuple(tensor.shape)} != model {tuple(param.shape)}"
                    )
                    continue
                with torch.no_grad():
                    param.copy_(tensor.to(dtype or param.dtype))
                loaded.append(name)
                seen.add(target)

    missing = tuple(sorted(set(destinations) - seen))
    report = LoadReport(
        loaded=tuple(sorted(loaded)),
        skipped_vision=tuple(sorted(skipped_vision)),
        skipped_mtp=tuple(sorted(skipped_mtp)),
        missing=missing,
        unmapped=tuple(sorted(unmapped)),
        mismatched=tuple(sorted(mismatched)),
    )
    if verbose:
        print(report.summary())
    if strict and not report.ok:
        raise WeightMappingError(
            "checkpoint did not map cleanly onto the reference model:\n" + report.summary()
        )
    return report
