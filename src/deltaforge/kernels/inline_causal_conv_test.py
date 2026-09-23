"""045, checked on a laptop — including its numerics, which is new.

Every other candidate in this directory is a Triton body: a string a GPU compiles, whose
arithmetic the CPU suite cannot see into. Batch 005 paid for that when
`_tiled_gemv_scaled_kernel` shipped, passed CI and review, and returned unscaled dot
products on a rented box (`AGENT.md` §8).

This candidate is torch operations, so the thing under test *is* the thing that runs, and
these assertions are about what it computes rather than only about where it installs.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ..config import tiny_config
from ..reference import GatedDeltaNet, ReferenceModel, _LinearLayerCache
from .fused_causal_conv import fused_causal_conv_step
from .inline_causal_conv import inline_causal_conv_step, install_inline_causal_conv


@pytest.fixture
def model():
    return ReferenceModel(tiny_config()).eval()


def _delta_net(model) -> GatedDeltaNet:
    nets = [m for m in model.modules() if isinstance(m, GatedDeltaNet)]
    assert nets, "the tiny config has no linear-attention layer"
    return nets[0]


def test_the_decode_step_is_bit_identical_to_the_reference_in_bf16(model):
    """The claim, run rather than re-derived.

    `F.conv1d` accumulates the four taps in fp32 and returns bf16; `F.silu` then runs on
    that bf16 tensor. This decomposition does the same in a different summation order, and
    the orders differ by ~1e-7 in fp32 — below one bf16 ULP, so the rounding erases it.
    That is why the rounding is where it is, and asserting `rtol=0, atol=0` is what would
    catch it moving.
    """
    model = model.to(torch.bfloat16)
    net = _delta_net(model)
    channels = net.conv1d.weight.shape[0]
    torch.manual_seed(0)
    x = torch.randn(1, channels, 1).bfloat16()
    history = torch.randn(1, channels, 3).bfloat16()

    cache = _LinearLayerCache(conv=history.clone(), recurrent=torch.zeros(1))
    expected = GatedDeltaNet._causal_conv(net, x, cache)

    advanced = history.clone()
    got = inline_causal_conv_step(x, advanced, net.conv1d.weight)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(advanced, cache.conv, rtol=0, atol=0)


def test_it_computes_what_the_triton_kernel_computes(model):
    """The two conv slots are read against each other, so they must agree by construction.

    `044` and `045` attack the same three launches by two different routes — an opaque
    custom op and a decomposition inductor can fuse — and the whole value of running both
    is that their *ratios* differ while their arithmetic does not. A difference in the
    numbers they compute would make that comparison meaningless, and it would show up on
    a rental as two different correctness results rather than as the one bug it is.
    """
    model = model.to(torch.bfloat16)
    net = _delta_net(model)
    channels = net.conv1d.weight.shape[0]
    torch.manual_seed(2)
    x = torch.randn(1, channels, 1).bfloat16()
    history = torch.randn(1, channels, 3).bfloat16()

    # The Triton path needs a card; its *spec* is the same expression this one computes,
    # and `fused_causal_conv_test` pins that spec against the reference independently.
    # What is checked here is the one thing a laptop can check about the pair: that the
    # op refuses the same shapes, and that the reference agrees with both.
    cache = _LinearLayerCache(conv=history.clone(), recurrent=torch.zeros(1))
    expected = GatedDeltaNet._causal_conv(net, x, cache)

    advanced = history.clone()
    got = inline_causal_conv_step(x, advanced, net.conv1d.weight)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    with pytest.raises(ValueError, match="decode step"):
        fused_causal_conv_step(torch.zeros(1, 8, 2), torch.zeros(1, 8, 3), torch.zeros(8, 1, 4))
    with pytest.raises(ValueError, match="decode step"):
        inline_causal_conv_step(torch.zeros(1, 8, 2), torch.zeros(1, 8, 3), torch.zeros(8, 1, 4))


def test_the_history_advances_in_the_right_direction(model):
    """The failure this test exists for decodes fluent nonsense rather than crashing.

    A history shifted the wrong way still has the right shape and the right dtype, and the
    model still runs. Distinct per-tap values are what make the direction visible.
    """
    net = _delta_net(model)
    channels = net.conv1d.weight.shape[0]
    history = torch.zeros(1, channels, 3)
    history[..., 0], history[..., 1], history[..., 2] = 1.0, 2.0, 3.0
    x = torch.full((1, channels, 1), 4.0)

    inline_causal_conv_step(x, history, net.conv1d.weight)

    assert history[0, 0].tolist() == [2.0, 3.0, 4.0]


def test_a_decode_step_runs_on_a_cpu_rather_than_raising(model):
    """The deliberate difference from `fused_causal_conv`, and it is a property not a gap.

    The Triton kernel raises on a CPU because a silent fallback would run the reference
    inside a measured region while the harness recorded it as the candidate. This
    candidate has no fallback to make: it *is* torch operations, so the same code runs
    everywhere and there is nothing for a device check to protect.
    """
    install_inline_causal_conv(model)
    net = _delta_net(model)
    channels = net.conv1d.weight.shape[0]
    cache = _LinearLayerCache(conv=torch.zeros(1, channels, 3), recurrent=torch.zeros(1))

    out = net._causal_conv(torch.zeros(1, channels, 1), cache)

    assert out.shape == (1, channels, 1)


def test_the_prefill_routes_back_to_the_reference_unchanged(model):
    net = _delta_net(model)
    channels = net.conv1d.weight.shape[0]
    torch.manual_seed(1)
    x = torch.randn(1, channels, 16)

    before = _LinearLayerCache(conv=torch.zeros(1, channels, 3), recurrent=torch.zeros(1))
    expected = GatedDeltaNet._causal_conv(net, x, before)

    install_inline_causal_conv(model)
    after = _LinearLayerCache(conv=torch.zeros(1, channels, 3), recurrent=torch.zeros(1))
    got = _delta_net(model)._causal_conv(x, after)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(after.conv, before.conv, rtol=0, atol=0)


def test_it_refuses_a_shape_it_does_not_implement():
    with pytest.raises(ValueError, match=r"\(B, C, 3\)"):
        inline_causal_conv_step(torch.zeros(1, 8, 1), torch.zeros(1, 8, 2), torch.zeros(8, 1, 4))
    with pytest.raises(ValueError, match="4-tap"):
        inline_causal_conv_step(torch.zeros(1, 8, 1), torch.zeros(1, 8, 3), torch.zeros(8, 1, 5))


def test_installing_patches_every_linear_attention_layer_and_is_idempotent(model):
    install_inline_causal_conv(model)
    patched = {id(m): type(m) for m in model.modules() if isinstance(m, GatedDeltaNet)}
    assert patched and all(cls is not GatedDeltaNet for cls in patched.values())

    install_inline_causal_conv(model)

    assert {id(m): type(m) for m in model.modules() if isinstance(m, GatedDeltaNet)} == patched


def test_installing_shares_the_reference_conv_weight(model):
    before = {name: p.data_ptr() for name, p in model.named_parameters()}

    install_inline_causal_conv(model)

    assert {name: p.data_ptr() for name, p in model.named_parameters()} == before
    assert all(isinstance(m.conv1d, nn.Conv1d) for m in model.modules() if isinstance(m, GatedDeltaNet))
