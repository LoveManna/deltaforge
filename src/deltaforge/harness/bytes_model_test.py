"""The byte arithmetic, pinned against the numbers ``docs/roofline.py`` has always printed.

Every expected value here is the checkpoint manifest's own arithmetic, not a round number
chosen to make a test pass. If one of these changes, the model changed.
"""

from __future__ import annotations

import pytest

from ..config import qwen3_5_4b_config
from .bytes_model import decode_bytes_per_token


def test_all_bf16_reproduces_the_roofline_scripts_total():
    """The number docs/roofline.py prints, from the same arithmetic in an importable place."""
    mb = decode_bytes_per_token(qwen3_5_4b_config(), weight_bits={}, context_length=2048, compiled=False)
    assert abs(mb - 9158.23) < 1.0


def test_the_compiled_default_drops_the_expansion_inductor_folds_away():
    """Rental 38 read the generated code: `repeat_interleave` is not in it, and the
    attention bmm indexes the unexpanded KV cache with `x1 // 4`.

    Both scored columns are compiled, so the default has to be the compiled byte count.
    Dividing a compiled column's time by the eager total is what put the baseline at
    "1308 GB/s, 73% of peak" when it is really 1177 and 65.7%.
    """
    config = qwen3_5_4b_config()
    eager = decode_bytes_per_token(config, weight_bits={}, context_length=2048, compiled=False)
    compiled = decode_bytes_per_token(config, weight_bits={}, context_length=2048)
    assert abs(compiled - 8587.80) < 1.0
    assert abs((eager - compiled) - 570.43) < 1.0


def test_halving_the_layer_projections_removes_half_their_bytes():
    config = qwen3_5_4b_config()
    full = decode_bytes_per_token(config, weight_bits={}, context_length=2048)
    int8 = decode_bytes_per_token(config, weight_bits={"layers": 8}, context_length=2048)
    # Layer projections are 7140 MB/token of the 8411 MB of weights.
    assert abs((full - int8) - 3570.0) < 5.0


def test_quantising_the_head_is_worth_its_own_share():
    config = qwen3_5_4b_config()
    layers = decode_bytes_per_token(config, weight_bits={"layers": 8}, context_length=2048)
    both = decode_bytes_per_token(config, weight_bits={"layers": 8, "head": 8}, context_length=2048)
    assert abs((layers - both) - 635.5) < 5.0


def test_an_unknown_region_is_refused_rather_than_ignored():
    """A typo in a manifest must not silently score against bf16 bytes."""
    with pytest.raises(ValueError, match="unknown weight region"):
        decode_bytes_per_token(qwen3_5_4b_config(), weight_bits={"mpl": 8}, context_length=2048)


def test_the_config_a_rented_box_builds_resolves_to_the_same_manifest():
    """On a rental the config comes from the checkpoint's own `config.json`, and
    `from_hf_config` sets `name` from `model_type` -- `'qwen3_5'`, not `'qwen3.5-4b'`.

    Matching on the name therefore never resolved a config that had actually been loaded
    from a checkpoint, which is the only kind a rental ever has. Rental 38 reported
    "no byte model" on every slot for exactly this and shipped no bandwidth at all. The
    lookup matches on the shape fields that determine the tensor manifest instead, because
    those are what the manifest is about.
    """
    from dataclasses import replace

    as_a_rental_builds_it = replace(qwen3_5_4b_config(), name="qwen3_5")

    mb = decode_bytes_per_token(as_a_rental_builds_it, weight_bits={}, context_length=2048)

    assert abs(mb - 8587.80) < 1.0


def test_a_config_whose_shape_matches_no_manifest_is_still_refused():
    """The guard has to stay a guard: an unknown model must report nothing, not the wrong
    checkpoint's byte count."""
    from ..config import tiny_config

    with pytest.raises(ValueError, match="no tensor manifest"):
        decode_bytes_per_token(tiny_config(), weight_bits={}, context_length=2048)
