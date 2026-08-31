"""Weight-mapping tests, against the real checkpoint's tensor manifest.

The manifest holds every tensor name, dtype and shape read from the published
safetensors headers — no weight data, about 80 KB. That is enough to prove exhaustively,
on CPU with no download, that:

* every one of the 738 real tensors is either mapped or deliberately skipped,
* every reference parameter has a checkpoint tensor of exactly the right shape.

Those two facts are most of what "the reference matches the checkpoint" means. What is
left — that the *values* are interpreted correctly — needs the oracle, and is deferred.
"""

from __future__ import annotations

import pytest
import torch

from .config import qwen3_5_4b_config, tiny_config
from .reference import ReferenceModel
from .weights import (
    MTP_PREFIX,
    VISION_PREFIX,
    LoadReport,
    WeightMappingError,
    expected_parameter_shapes,
    load_manifest,
    load_weights,
    map_checkpoint_name,
)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def tensor_names(manifest):
    return list(manifest["tensors"])


# -- the manifest itself --------------------------------------------------------------


def test_manifest_describes_the_published_checkpoint(manifest, tensor_names):
    assert manifest["repo_id"] == "Qwen/Qwen3.5-4B"
    assert len(manifest["shards"]) == 2
    assert len(tensor_names) == 738
    # Roughly 9 GB of bf16 weights.
    assert 8e9 < manifest["total_size"] < 1.1e10


def test_manifest_carries_no_weight_data(manifest):
    """It must stay a metadata file: names, dtypes and shapes only."""
    for meta in manifest["tensors"].values():
        assert set(meta) == {"dtype", "shape"}
        assert isinstance(meta["shape"], list)


# -- the name map -------------------------------------------------------------------


def test_every_checkpoint_tensor_is_mapped_or_deliberately_skipped(tensor_names):
    """No silent fall-through. A tensor we neither map nor consciously skip would
    produce a model that runs, gives plausible logits, and is wrong."""
    mapped, skipped = [], []
    for name in tensor_names:
        target = map_checkpoint_name(name)  # raises on anything unrecognised
        (skipped if target is None else mapped).append(name)

    assert len(mapped) + len(skipped) == len(tensor_names)
    assert all(n.startswith((VISION_PREFIX, MTP_PREFIX)) for n in skipped)

    # Derived rather than pasted, so the expectation states the architecture:
    #   24 linear-attention layers x 14 tensors  (2 norms, 3 MLP, 9 Gated DeltaNet)
    # +  8 full-attention layers   x 11 tensors  (2 norms, 3 MLP, 6 attention)
    # +  embed_tokens and the final norm
    assert len(mapped) == 24 * 14 + 8 * 11 + 2 == 426
    assert len(skipped) == 738 - 426 == 312


def test_the_vision_tower_and_mtp_head_are_the_only_things_skipped(tensor_names):
    vision = [n for n in tensor_names if n.startswith(VISION_PREFIX)]
    mtp = [n for n in tensor_names if n.startswith(MTP_PREFIX)]

    assert vision and mtp
    assert all(map_checkpoint_name(n) is None for n in vision + mtp)
    # And nothing in the text decoder is skipped.
    text = [n for n in tensor_names if n.startswith("model.language_model.")]
    assert all(map_checkpoint_name(n) is not None for n in text)


def test_an_unrecognised_text_tensor_fails_loudly():
    """A new tensor appearing inside the text decoder is a real architecture change and
    must not be quietly ignored."""
    with pytest.raises(WeightMappingError, match="no mapping for layer suffix"):
        map_checkpoint_name("model.language_model.layers.0.self_attn.sink.weight")
    with pytest.raises(WeightMappingError, match="unrecognised checkpoint tensor"):
        map_checkpoint_name("model.language_model.something_new.weight")


def test_mapping_examples():
    assert (
        map_checkpoint_name("model.language_model.layers.7.self_attn.q_proj.weight")
        == "layers.7.self_attn.q_proj.weight"
    )
    assert (
        map_checkpoint_name("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        == "layers.0.linear_attn.in_proj_qkv.weight"
    )
    assert map_checkpoint_name("model.language_model.embed_tokens.weight") == "embed_tokens.weight"
    assert map_checkpoint_name("model.language_model.norm.weight") == "norm.weight"


# -- shapes agree with the reference model --------------------------------------------


def test_reference_parameters_match_the_checkpoint_shapes_exactly():
    """The strongest CPU-only statement available about the reference being right: it is
    built with exactly the parameters the real checkpoint contains, at exactly the right
    shapes. This is what catches a head_dim or projection-width mistake without renting
    anything.

    Built on the meta device so nothing is allocated: the real model is 4.2 B parameters.
    """
    config = qwen3_5_4b_config()
    with torch.device("meta"):
        model = ReferenceModel(config)

    expected = expected_parameter_shapes(config)
    actual = {name: list(p.shape) for name, p in model.named_parameters()}

    assert set(actual) == set(expected), (
        f"only in model: {sorted(set(actual) - set(expected))}; "
        f"only in checkpoint: {sorted(set(expected) - set(actual))}"
    )
    mismatched = {k: (actual[k], expected[k]) for k in actual if actual[k] != expected[k]}
    assert not mismatched, f"shape mismatches (model, checkpoint): {mismatched}"


def test_the_shape_check_pins_the_awkward_dimensions():
    """Spot-check the values most likely to be wrong, so a future refactor that breaks
    them fails with a readable message rather than a set-difference dump."""
    config = qwen3_5_4b_config()
    expected = expected_parameter_shapes(config)

    # q_proj is doubled by the output gate: 16 heads * 256 * 2.
    assert expected["layers.3.self_attn.q_proj.weight"] == [8192, 2560]
    # k/v are GQA-narrow: 4 heads * 256.
    assert expected["layers.3.self_attn.k_proj.weight"] == [1024, 2560]
    assert expected["layers.3.self_attn.o_proj.weight"] == [2560, 4096]
    # Norms are over the head dimension only.
    assert expected["layers.3.self_attn.q_norm.weight"] == [256]
    # Linear attention: fused qkv, separate z/b/a.
    assert expected["layers.0.linear_attn.in_proj_qkv.weight"] == [8192, 2560]
    assert expected["layers.0.linear_attn.in_proj_z.weight"] == [4096, 2560]
    assert expected["layers.0.linear_attn.in_proj_b.weight"] == [32, 2560]
    assert expected["layers.0.linear_attn.in_proj_a.weight"] == [32, 2560]
    # Depthwise conv: one filter per channel.
    assert expected["layers.0.linear_attn.conv1d.weight"] == [8192, 1, 4]
    assert expected["embed_tokens.weight"] == [248320, 2560]


def test_tied_embeddings_mean_there_is_no_lm_head_to_load():
    config = qwen3_5_4b_config()
    assert "lm_head.weight" not in expected_parameter_shapes(config)


def test_checkpoint_keeps_the_recurrent_parameters_in_float32(manifest):
    """mamba_ssm_dtype is float32, and the checkpoint reflects it: A_log and the gated
    norm are stored fp32 while everything around them is bf16."""
    tensors = manifest["tensors"]
    assert tensors["model.language_model.layers.0.linear_attn.A_log"]["dtype"] == "F32"
    assert tensors["model.language_model.layers.0.linear_attn.norm.weight"]["dtype"] == "F32"
    assert tensors["model.language_model.layers.0.mlp.gate_proj.weight"]["dtype"] == "BF16"


# -- loading ------------------------------------------------------------------------


def test_load_report_summarises_what_was_skipped():
    report = LoadReport(
        loaded=("a", "b"),
        skipped_vision=("model.visual.x",),
        skipped_mtp=("mtp.y", "mtp.z"),
    )
    summary = report.summary()

    assert report.ok
    assert "loaded 2 tensors" in summary
    assert "skipped 1 vision-tower tensors (excluded by design)" in summary
    assert "skipped 2 MTP-head tensors (excluded by design)" in summary


def test_load_report_is_not_ok_when_something_is_missing():
    report = LoadReport(loaded=("a",), missing=("layers.0.mlp.up_proj.weight",))
    assert not report.ok
    assert "1 missing from checkpoint" in report.summary()


def test_load_weights_round_trips_a_tiny_checkpoint(tmp_path):
    """Exercise the real loader end to end on a checkpoint small enough for CI: build a
    tiny model, save it under checkpoint names, load it into a fresh model, compare."""
    safetensors_torch = pytest.importorskip("safetensors.torch")

    config = tiny_config()
    torch.manual_seed(11)
    source = ReferenceModel(config)
    for param in source.parameters():
        torch.nn.init.normal_(param, std=0.1)

    inverse = {
        "embed_tokens.weight": "model.language_model.embed_tokens.weight",
        "norm.weight": "model.language_model.norm.weight",
    }
    payload = {}
    for name, param in source.named_parameters():
        checkpoint_name = inverse.get(name) or f"model.language_model.layers.{name[len('layers.') :]}"
        payload[checkpoint_name] = param.detach().clone()
    # Include tensors that must be skipped, to prove the skip path runs.
    payload["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(4, 4)
    payload["mtp.fc.weight"] = torch.zeros(4, 4)

    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file(payload, str(path))

    destination = ReferenceModel(config)
    report = load_weights(destination, path, verbose=False)

    assert report.ok
    assert len(report.skipped_vision) == 1
    assert len(report.skipped_mtp) == 1
    assert not report.missing and not report.unmapped and not report.mismatched

    ids = torch.randint(0, config.vocab_size, (1, 4))
    torch.testing.assert_close(source(ids)[0], destination(ids)[0], rtol=0, atol=0)


def test_load_weights_raises_when_a_tensor_is_missing(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")

    config = tiny_config()
    model = ReferenceModel(config)
    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file(
        {"model.language_model.norm.weight": torch.zeros(config.hidden_size)}, str(path)
    )

    with pytest.raises(WeightMappingError, match="did not map cleanly"):
        load_weights(model, path, verbose=False)


def test_load_weights_raises_on_a_shape_mismatch(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")

    config = tiny_config()
    model = ReferenceModel(config)
    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file({"model.language_model.norm.weight": torch.zeros(7)}, str(path))

    with pytest.raises(WeightMappingError, match="did not map cleanly"):
        load_weights(model, path, verbose=False)

    report = load_weights(model, path, strict=False, verbose=False)
    assert report.mismatched
    assert "checkpoint (7,) != model (128,)" in report.mismatched[0]
