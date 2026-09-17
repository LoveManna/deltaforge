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
    mb = decode_bytes_per_token(qwen3_5_4b_config(), weight_bits={}, context_length=2048)
    assert abs(mb - 9158.23) < 1.0


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
