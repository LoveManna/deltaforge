"""CPU tests for the baseline, against the tiny config fixture.

The tiny config is the smallest model that still exercises everything structural: both
layer kinds in the real order, the GQA ratio, the linear-attention key/value head ratio,
partial RoPE, the output gate, and both cache kinds. Everything here runs in float32 on
CPU in well under a second, which is what makes it a gate the CI can hold.

What these prove: shapes, the cache contract, causality, the mRoPE-to-RoPE reduction, and
step-versus-chunk equivalence of the recurrence. What they cannot prove is that the
*weights* are interpreted correctly — that needs the real checkpoint and the HuggingFace
oracle, and is deferred to the first funded session (see ``oracle_test.py``).
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from .config import FULL_ATTENTION, LINEAR_ATTENTION, tiny_config
from .reference import (
    DecodeCache,
    ReferenceModel,
    RMSNorm,
    RotaryEmbedding,
    apply_output_gate,
    apply_partial_rope,
    recurrent_gated_delta_rule,
    rotate_half,
)


@pytest.fixture
def config():
    return tiny_config()


@pytest.fixture
def model(config):
    """A deterministically initialised tiny model.

    Norm weights get a small non-zero value on purpose: with the checkpoint's
    zero-centred convention, leaving them at zero would make the `(1 + w)` scaling
    indistinguishable from a plain `w` bug.
    """
    torch.manual_seed(20260830)
    model = ReferenceModel(config).to(torch.float32).eval()
    for name, param in model.named_parameters():
        if name.endswith("A_log"):
            torch.nn.init.uniform_(param, -1.0, 1.0)
        elif "norm" in name or "layernorm" in name:
            torch.nn.init.normal_(param, mean=0.0, std=0.1)
        else:
            torch.nn.init.normal_(param, std=0.05)
    return model


# -- shapes and structure -------------------------------------------------------------


def test_tiny_config_has_one_layer_of_each_kind(config):
    assert config.layer_types == (LINEAR_ATTENTION, FULL_ATTENTION)
    assert config.layer_indices(LINEAR_ATTENTION) == (0,)
    assert config.layer_indices(FULL_ATTENTION) == (1,)


def test_forward_returns_logits_over_the_vocabulary(model, config):
    ids = torch.randint(0, config.vocab_size, (2, 5))
    logits, cache = model(ids)

    assert logits.shape == (2, 5, config.vocab_size)
    assert cache is None
    assert torch.isfinite(logits).all()


def test_num_logits_to_keep_trims_the_output(model, config):
    ids = torch.randint(0, config.vocab_size, (2, 5))
    full, _ = model(ids)
    last, _ = model(ids, num_logits_to_keep=1)

    assert last.shape == (2, 1, config.vocab_size)
    torch.testing.assert_close(last[:, 0], full[:, -1])


def test_lm_head_is_tied_to_the_embedding(model):
    assert model.config.tie_word_embeddings
    assert model.lm_head_weight is model.embed_tokens.weight
    assert not hasattr(model, "lm_head")


def test_attention_projections_are_not_square(config):
    """head_dim is 256 while hidden/heads is 160 on the real model; the tiny config
    reproduces that asymmetry so a 'square projection' assumption fails here too."""
    assert config.head_dim * config.num_attention_heads != config.hidden_size
    model = ReferenceModel(config)
    attn = model.layers[1].self_attn
    # q_proj is doubled because the attention output is gated.
    assert attn.q_proj.weight.shape == (
        config.num_attention_heads * config.head_dim * 2,
        config.hidden_size,
    )
    assert attn.k_proj.weight.shape == (
        config.num_key_value_heads * config.head_dim,
        config.hidden_size,
    )


# -- RMSNorm convention ---------------------------------------------------------------


def test_rmsnorm_scale_is_one_plus_weight():
    """The checkpoint stores zero-centred norm weights. Reading them as a plain scale
    yields an all-zero activation, which is the classic silent way to get this model
    wrong."""
    norm = RMSNorm(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    with torch.no_grad():
        norm.weight.zero_()
        identity_scaled = norm(x)
        rms = x.pow(2).mean(-1, keepdim=True).add(norm.eps).rsqrt()
        torch.testing.assert_close(identity_scaled, x * rms)
        assert not torch.allclose(identity_scaled, torch.zeros_like(x))

        norm.weight.fill_(1.0)
        torch.testing.assert_close(norm(x), x * rms * 2.0)


# -- RoPE -----------------------------------------------------------------------------


def test_partial_rope_leaves_the_tail_untouched(config):
    rotary = RotaryEmbedding(config)
    assert config.rotary_dim == 8
    assert config.rotary_dim < config.head_dim

    positions = torch.arange(6).unsqueeze(0)
    cos, sin = rotary(positions, torch.float32)
    x = torch.randn(1, config.num_attention_heads, 6, config.head_dim)
    out = apply_partial_rope(x, cos, sin)

    torch.testing.assert_close(out[..., config.rotary_dim :], x[..., config.rotary_dim :])
    assert not torch.allclose(out[..., : config.rotary_dim], x[..., : config.rotary_dim])


def test_rope_at_position_zero_is_the_identity(config):
    rotary = RotaryEmbedding(config)
    cos, sin = rotary(torch.zeros(1, 1, dtype=torch.long), torch.float32)
    x = torch.randn(1, 2, 1, config.head_dim)

    torch.testing.assert_close(apply_partial_rope(x, cos, sin), x)


def test_mrope_reduces_exactly_to_standard_rope_for_text_only(config):
    """The reduction the whole text-decode path depends on, checked rather than assumed.

    Qwen3.5 is multimodal, so RoPE positions are 3-dimensional (t, h, w) and the
    frequency bands are interleaved between the three sections. For text-only generation
    HuggingFace expands one row of text positions across all three sections, so every
    band — whichever section it is drawn from — sees the same position index, and the
    construction collapses to standard RoPE.

    Assumption + a plausible-looking output is how a subtly wrong RoPE ships, so this
    asserts exact equality against a standard RoPE computed independently.
    """
    rotary = RotaryEmbedding(config)
    assert rotary.mrope_interleaved
    assert sum(config.mrope_section) == config.rotary_dim // 2

    positions = torch.arange(11).unsqueeze(0)  # (batch=1, seq=11)
    three_rows = positions.unsqueeze(0).expand(3, -1, -1)  # what text-only decode builds
    assert torch.equal(three_rows[0], three_rows[1])
    assert torch.equal(three_rows[1], three_rows[2])

    cos_mrope, sin_mrope = rotary(three_rows, torch.float32)

    # Standard RoPE, computed here from first principles, not from the module.
    inv_freq = 1.0 / (
        config.rope_theta ** (torch.arange(0, config.rotary_dim, 2, dtype=torch.float32) / config.rotary_dim)
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)

    torch.testing.assert_close(cos_mrope, emb.cos(), rtol=0, atol=0)
    torch.testing.assert_close(sin_mrope, emb.sin(), rtol=0, atol=0)

    # And the 2-D convenience form must take the same path.
    cos_2d, sin_2d = rotary(positions, torch.float32)
    torch.testing.assert_close(cos_2d, cos_mrope, rtol=0, atol=0)
    torch.testing.assert_close(sin_2d, sin_mrope, rtol=0, atol=0)


def test_mrope_does_diverge_when_the_sections_differ(config):
    """The reduction above is a property of text-only inputs, not of the code being a
    no-op. With genuinely different section positions the result must change, otherwise
    the interleaving is not wired up at all."""
    rotary = RotaryEmbedding(config)
    positions = torch.arange(7).unsqueeze(0)
    same = positions.unsqueeze(0).expand(3, -1, -1)
    different = torch.stack([positions, positions + 3, positions + 5])

    cos_same, _ = rotary(same, torch.float32)
    cos_diff, _ = rotary(different, torch.float32)

    assert not torch.allclose(cos_same, cos_diff)


def test_rotate_half_matches_its_definition():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


# -- the delta-rule recurrence --------------------------------------------------------


def _delta_inputs(seq_len: int, heads: int = 4, k_dim: int = 16, v_dim: int = 16, batch: int = 2):
    torch.manual_seed(7)
    return {
        "query": torch.randn(batch, seq_len, heads, k_dim),
        "key": torch.randn(batch, seq_len, heads, k_dim),
        "value": torch.randn(batch, seq_len, heads, v_dim),
        # g is a log-space decay, so it must be <= 0.
        "g": -torch.rand(batch, seq_len, heads),
        "beta": torch.rand(batch, seq_len, heads),
    }


def test_delta_rule_step_by_step_matches_whole_sequence():
    """Step-versus-chunk equivalence.

    Feeding the sequence one token at a time, carrying the recurrent state forward, must
    give exactly what feeding the whole sequence at once gives. This is the property the
    decode cache relies on, and it is also the property any future chunked/parallel scan
    (hypothesis 4) will have to preserve — this test is the contract that restructuring
    must satisfy.
    """
    seq_len = 9
    inputs = _delta_inputs(seq_len)

    whole, final_state = recurrent_gated_delta_rule(**inputs)

    stepped = []
    state = None
    for t in range(seq_len):
        out, state = recurrent_gated_delta_rule(
            **{k: v[:, t : t + 1] for k, v in inputs.items()}, initial_state=state
        )
        stepped.append(out)
    stepped = torch.cat(stepped, dim=1)

    torch.testing.assert_close(stepped, whole, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state, final_state, rtol=1e-5, atol=1e-6)


def test_delta_rule_split_in_two_chunks_matches_whole_sequence():
    """The same property at chunk granularity rather than single steps."""
    seq_len = 10
    inputs = _delta_inputs(seq_len)
    whole, whole_state = recurrent_gated_delta_rule(**inputs)

    first, state = recurrent_gated_delta_rule(**{k: v[:, :4] for k, v in inputs.items()})
    second, state = recurrent_gated_delta_rule(
        **{k: v[:, 4:] for k, v in inputs.items()}, initial_state=state
    )

    torch.testing.assert_close(torch.cat([first, second], dim=1), whole, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state, whole_state, rtol=1e-5, atol=1e-6)


def test_delta_rule_state_is_float32_regardless_of_input_dtype():
    """The checkpoint sets mamba_ssm_dtype=float32: the recurrent state stays fp32 even
    though the weights are bf16. Any future scan kernel must keep this."""
    inputs = {k: v.to(torch.bfloat16) for k, v in _delta_inputs(4).items()}
    out, state = recurrent_gated_delta_rule(**inputs)

    assert state.dtype == torch.float32
    assert out.dtype == torch.bfloat16, "output returns to the input dtype"


def test_delta_rule_is_causal():
    """Changing token t must not change any output before t."""
    seq_len = 6
    inputs = _delta_inputs(seq_len)
    baseline, _ = recurrent_gated_delta_rule(**inputs)

    perturbed = {k: v.clone() for k, v in inputs.items()}
    perturbed["value"][:, 3] += 10.0
    after, _ = recurrent_gated_delta_rule(**perturbed)

    torch.testing.assert_close(after[:, :3], baseline[:, :3], rtol=1e-6, atol=1e-7)
    assert not torch.allclose(after[:, 3], baseline[:, 3])


def test_delta_rule_zero_beta_leaves_the_state_only_decayed():
    """beta is the write rate: at zero, nothing is written and the state only decays."""
    inputs = _delta_inputs(3)
    inputs["beta"] = torch.zeros_like(inputs["beta"])
    out, state = recurrent_gated_delta_rule(**inputs)

    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=1e-7)
    torch.testing.assert_close(state, torch.zeros_like(state), rtol=0, atol=1e-7)


# -- the cache contract ---------------------------------------------------------------


def test_incremental_decode_matches_a_single_full_forward(model, config):
    """The contract the whole benchmark rests on: decoding token by token through the
    cache must reproduce what one full-sequence forward produces."""
    ids = torch.randint(0, config.vocab_size, (2, 7))
    full, _ = model(ids)

    cache = model.new_cache(batch_size=2, max_seq_len=16)
    pieces = []
    prefill, cache = model(ids[:, :4], cache)
    pieces.append(prefill)
    for t in range(4, 7):
        step, cache = model(ids[:, t : t + 1], cache)
        pieces.append(step)
    incremental = torch.cat(pieces, dim=1)

    torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-4)
    assert cache.seq_len == 7


def test_token_by_token_from_empty_matches_full_forward(model, config):
    """The degenerate case: no prefill at all, one token at a time from position zero."""
    ids = torch.randint(0, config.vocab_size, (1, 5))
    full, _ = model(ids)

    cache = model.new_cache(batch_size=1, max_seq_len=8)
    pieces = [model(ids[:, t : t + 1], cache)[0] for t in range(5)]

    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-4, atol=1e-4)


def test_cache_reset_returns_to_a_clean_state(model, config):
    ids = torch.randint(0, config.vocab_size, (1, 4))
    cache = model.new_cache(batch_size=1, max_seq_len=8)

    first, _ = model(ids, cache)
    cache.reset()
    assert cache.seq_len == 0
    second, _ = model(ids, cache)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_cache_allocates_the_right_state_for_each_layer_kind(config):
    cache = DecodeCache(config, batch_size=3, max_seq_len=11)

    linear = cache.linear(0)
    assert linear.conv.shape == (3, config.linear_conv_dim, config.linear_conv_kernel_dim - 1)
    assert linear.recurrent.shape == (
        3,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    )
    assert linear.recurrent.dtype == torch.float32

    attention = cache.attention(1)
    assert attention.keys.shape == (3, config.num_key_value_heads, 11, config.head_dim)
    assert attention.values.shape == attention.keys.shape


def test_cache_rejects_the_wrong_layer_kind(config):
    cache = DecodeCache(config, batch_size=1, max_seq_len=4)
    with pytest.raises(TypeError, match="not a full-attention layer"):
        cache.attention(0)
    with pytest.raises(TypeError, match="not a linear-attention layer"):
        cache.linear(1)


def test_cache_overflow_is_an_error_not_silent_corruption(model, config):
    cache = model.new_cache(batch_size=1, max_seq_len=4)
    ids = torch.randint(0, config.vocab_size, (1, 3))
    model(ids, cache)

    with pytest.raises(ValueError, match="cache overflow"):
        model(torch.randint(0, config.vocab_size, (1, 3)), cache)


def test_kv_cache_is_preallocated_not_grown(model, config):
    """Growing the cache with torch.cat is quadratic and would make the baseline
    artificially slow, inflating every ratio we report."""
    cache = model.new_cache(batch_size=1, max_seq_len=32)
    keys = cache.attention(1).keys
    storage_before = keys.data_ptr()

    model(torch.randint(0, config.vocab_size, (1, 5)), cache)

    assert cache.attention(1).keys.data_ptr() == storage_before
    assert cache.attention(1).keys.shape[2] == 32


# -- causality through the whole model ------------------------------------------------


def test_model_is_causal(model, config):
    """Changing a later token must not change any earlier position's logits."""
    ids = torch.randint(0, config.vocab_size, (1, 8))
    baseline, _ = model(ids)

    changed = ids.clone()
    changed[0, 5] = (changed[0, 5] + 1) % config.vocab_size
    after, _ = model(changed)

    torch.testing.assert_close(after[:, :5], baseline[:, :5], rtol=1e-4, atol=1e-4)
    assert not torch.allclose(after[:, 5], baseline[:, 5])


def test_attention_masking_is_position_aware_with_a_cache(model, config):
    """A cached prefill followed by a multi-token chunk must mask against absolute
    positions, not against positions within the chunk."""
    ids = torch.randint(0, config.vocab_size, (1, 6))
    full, _ = model(ids)

    cache = model.new_cache(batch_size=1, max_seq_len=12)
    first, _ = model(ids[:, :2], cache)
    rest, _ = model(ids[:, 2:], cache)  # a 4-token chunk on top of a 2-token cache

    torch.testing.assert_close(torch.cat([first, rest], dim=1), full, rtol=1e-4, atol=1e-4)


# -- dtype and numerics ---------------------------------------------------------------


def test_forward_runs_in_bfloat16(config):
    torch.manual_seed(3)
    model = ReferenceModel(config).to(torch.bfloat16).eval()
    ids = torch.randint(0, config.vocab_size, (1, 4))
    logits, _ = model(ids)

    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits.float()).all()


def test_recurrent_state_stays_fp32_in_a_bfloat16_model(config):
    model = ReferenceModel(config).to(torch.bfloat16).eval()
    cache = model.new_cache(batch_size=1, max_seq_len=4)

    assert cache.linear(0).recurrent.dtype == torch.float32
    model(torch.randint(0, config.vocab_size, (1, 2)), cache)
    assert cache.linear(0).recurrent.dtype == torch.float32


def test_gated_attention_output_gate_is_actually_applied(model, config):
    """Zeroing the gate half of q_proj drives sigmoid(gate) to 0.5 uniformly; the output
    must change. If the gate were being ignored, this would be a no-op."""
    ids = torch.randint(0, config.vocab_size, (1, 4))
    before, _ = model(ids)

    attn = model.layers[1].self_attn
    with torch.no_grad():
        reshaped = attn.q_proj.weight.view(config.num_attention_heads, 2, config.head_dim, -1)
        reshaped[:, 1].zero_()
    after, _ = model(ids)

    assert not torch.allclose(before, after)


def test_conv_history_is_carried_across_calls(model, config):
    """The short causal conv needs kernel_size-1 tokens of history; without it, a chunked
    forward would silently differ from a single one. Proven by the equivalence tests
    above, and pinned here at the state level."""
    cache = model.new_cache(batch_size=1, max_seq_len=8)
    assert torch.all(cache.linear(0).conv == 0)

    model(torch.randint(0, config.vocab_size, (1, 3)), cache)

    assert torch.any(cache.linear(0).conv != 0)


def test_scaling_uses_head_dim_not_hidden_over_heads(config):
    model = ReferenceModel(config)
    assert model.layers[1].self_attn.scaling == pytest.approx(1.0 / math.sqrt(config.head_dim))


def test_the_output_gate_nonlinearity_follows_the_config():
    """A kernel or a reference that hardcodes one gate is wrong on the other checkpoint.

    Both gates are monotonic and near 1 for large positive inputs, so a wrong choice looks
    almost right on real activations. Pinning both here means the difference is a test
    failure rather than a plausible-looking logit.
    """
    gate = torch.tensor([-2.0, 0.0, 3.0])

    sigmoid = apply_output_gate(gate, "sigmoid")
    swish = apply_output_gate(gate, "swish")

    torch.testing.assert_close(sigmoid, torch.sigmoid(gate))
    torch.testing.assert_close(swish, torch.nn.functional.silu(gate))
    # They disagree everywhere that matters, including in sign below zero.
    assert not torch.allclose(sigmoid, swish)
    assert sigmoid[0] > 0 and swish[0] < 0

    with pytest.raises(ValueError, match="output_gate_type"):
        apply_output_gate(gate, "gelu")


def test_attention_uses_the_gate_the_config_names():
    """Swapping only `output_gate_type` must change the model's output."""
    base = tiny_config()
    swish_config = replace(base, output_gate_type="swish", name="tiny-swish")

    torch.manual_seed(0)
    sigmoid_model = ReferenceModel(base).eval()
    swish_model = ReferenceModel(swish_config).eval()
    swish_model.load_state_dict(sigmoid_model.state_dict())

    ids = torch.randint(0, base.vocab_size, (1, 6))
    with torch.no_grad():
        a, _ = sigmoid_model(ids)
        b, _ = swish_model(ids)

    assert not torch.allclose(a, b), "the gate type made no difference; it is being ignored"


def test_a_cache_snapshot_restores_the_state_a_prefill_left_behind(model, config):
    """The bench's rounds are what this is for: prefill once, restore per round.

    Each scoring round used to re-run the whole 2048-token prefill as setup. It is
    excluded from the timed region, so it never touched a ratio — it just made a round
    cost ~18 s of billed rental where the timed region costs ~2. Restoring a snapshot
    reproduces the same starting state at the price of a device-to-device copy.
    """
    ids = torch.randint(0, config.vocab_size, (1, 4))
    cache = model.new_cache(batch_size=1, max_seq_len=16)
    model(ids, cache)
    snapshot = cache.snapshot()
    expected, _ = model(ids[:, -1:], cache)

    cache.restore(snapshot)
    assert cache.seq_len == snapshot.seq_len
    again, _ = model(ids[:, -1:], cache)

    torch.testing.assert_close(again, expected, rtol=0, atol=0)


def test_restoring_a_snapshot_writes_into_the_cache_tensors_it_already_had(model, config):
    """In place, because an address that moves between rounds is a different experiment.

    `034-static-cache-cudagraphs` won 2.0% by promising inductor these tensors never move
    (`mark_static_address` puts them in `static_input_idxs`, which is what skips the
    per-call alignment check). A restore that rebound them to fresh storage would silently
    void that slot and any CUDA-graph work behind it.
    """
    cache = model.new_cache(batch_size=1, max_seq_len=16)
    model(torch.randint(0, config.vocab_size, (1, 4)), cache)
    snapshot = cache.snapshot()
    attn_layer = next(i for i, t in enumerate(config.layer_types) if t == FULL_ATTENTION)
    linear_layer = next(i for i, t in enumerate(config.layer_types) if t != FULL_ATTENTION)
    keys = cache.attention(attn_layer).keys
    recurrent = cache.linear(linear_layer).recurrent

    cache.restore(snapshot)

    assert cache.attention(attn_layer).keys is keys
    assert cache.linear(linear_layer).recurrent is recurrent


def test_a_snapshot_is_a_copy_and_not_a_view_of_the_live_cache(model, config):
    """A snapshot aliasing the cache would restore whatever the last round happened to leave."""
    cache = model.new_cache(batch_size=1, max_seq_len=16)
    model(torch.randint(0, config.vocab_size, (1, 4)), cache)
    snapshot = cache.snapshot()
    before = snapshot.tensors[0].clone()

    model(torch.randint(0, config.vocab_size, (1, 1)), cache)

    torch.testing.assert_close(snapshot.tensors[0], before, rtol=0, atol=0)


def test_restoring_a_snapshot_from_a_differently_shaped_cache_is_refused(config):
    small = DecodeCache(config, batch_size=1, max_seq_len=8)
    large = DecodeCache(config, batch_size=1, max_seq_len=16)

    with pytest.raises(ValueError, match="different cache"):
        large.restore(small.snapshot())
