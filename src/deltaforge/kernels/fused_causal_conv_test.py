"""The parts of 025 that are decidable without a GPU.

The Triton kernel itself needs a card, and `fused_causal_conv_correctness_checks` runs it
there against `GatedDeltaNet._causal_conv` itself. What is checkable here is the part that
would be wrong *silently*: the history semantics, the routing, and the refusal to fall back.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ..config import tiny_config
from ..reference import GatedDeltaNet, ReferenceModel, _LinearLayerCache
from .fused_causal_conv import fused_causal_conv_step, install_fused_causal_conv


@pytest.fixture
def model():
    return ReferenceModel(tiny_config()).eval()


def _delta_net(model) -> GatedDeltaNet:
    nets = [m for m in model.modules() if isinstance(m, GatedDeltaNet)]
    assert nets, "the tiny config has no linear-attention layer"
    return nets[0]


def test_the_kernels_arithmetic_matches_the_reference_step_for_step(model):
    """The kernel's spec, written as torch and checked against the method it replaces.

    This is not the kernel — it is what the kernel must compute, and it is here because the
    two things that go wrong in a cache-shifting kernel go wrong quietly. The history could
    advance in the wrong direction (the model still runs, and decodes fluent nonsense), and
    the rounding could happen in the wrong place: `F.conv1d` accumulates in fp32 and returns
    bf16, and `F.silu` then runs on that bf16 tensor. Batch 003 lost a slot to a reference
    that dequantised in fp32 while the implementation rounded to bf16 first.
    """
    model = model.to(torch.bfloat16)  # the dtype a rental runs, and the dtype the claim is about
    net = _delta_net(model)
    channels = net.conv1d.weight.shape[0]
    torch.manual_seed(0)
    x = torch.randn(1, channels, 1).bfloat16()
    history = torch.randn(1, channels, 3).bfloat16()

    cache = _LinearLayerCache(conv=history.clone(), recurrent=torch.zeros(1))
    expected = GatedDeltaNet._causal_conv(net, x, cache)

    w = net.conv1d.weight.reshape(channels, 4).float()
    taps = torch.cat((history, x), dim=-1).float()  # [h0, h1, h2, x]
    acc = (taps * w.unsqueeze(0)).sum(dim=-1)
    rounded = acc.to(x.dtype).to(torch.float32)
    kernel_out = (rounded * torch.sigmoid(rounded)).to(x.dtype).unsqueeze(-1)
    kernel_history = taps[..., 1:].to(x.dtype)

    # Bit-for-bit, in bf16. The fp32 versions of these two expressions differ by about
    # 1e-7 because cuDNN sums the four taps in a different order; rounding to bf16 erases
    # that, which is the whole reason the kernel rounds where it does.
    torch.testing.assert_close(kernel_out, expected, rtol=0, atol=0)
    torch.testing.assert_close(kernel_history, cache.conv, rtol=0, atol=0)


def test_the_prefill_routes_back_to_the_reference_unchanged(model):
    """`run_interleaved` excludes every setup from the timed region and the prefill is
    setup, so a multi-token call takes the reference's path — the same boundary
    `tiled_gemv.GEMV_MAX_ROWS` draws, and the only one this kernel has."""
    net = _delta_net(model)
    channels = net.conv1d.weight.shape[0]
    torch.manual_seed(1)
    x = torch.randn(1, channels, 16)

    before = _LinearLayerCache(conv=torch.zeros(1, channels, 3), recurrent=torch.zeros(1))
    expected = GatedDeltaNet._causal_conv(net, x, before)

    install_fused_causal_conv(model)
    after = _LinearLayerCache(conv=torch.zeros(1, channels, 3), recurrent=torch.zeros(1))
    got = _delta_net(model)._causal_conv(x, after)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(after.conv, before.conv, rtol=0, atol=0)


def test_a_decode_step_on_cpu_raises_rather_than_running_the_reference(model):
    """There is no fallback here, deliberately.

    A candidate that quietly ran the reference on an unsupported input would produce a
    plausible number for the wrong thing and the harness would record it as the kernel.
    Failing turns that into an errored slot, which is a result.
    """
    install_fused_causal_conv(model)
    net = _delta_net(model)
    channels = net.conv1d.weight.shape[0]
    cache = _LinearLayerCache(conv=torch.zeros(1, channels, 3), recurrent=torch.zeros(1))

    with pytest.raises(RuntimeError, match="requires a CUDA tensor"):
        net._causal_conv(torch.zeros(1, channels, 1), cache)


def test_the_op_refuses_a_shape_it_does_not_implement():
    with pytest.raises(ValueError, match="decode step"):
        fused_causal_conv_step(torch.zeros(1, 8, 2), torch.zeros(1, 8, 3), torch.zeros(8, 1, 4))
    with pytest.raises(ValueError, match=r"\(B, C, 3\)"):
        fused_causal_conv_step(torch.zeros(1, 8, 1), torch.zeros(1, 8, 2), torch.zeros(8, 1, 4))


def test_installing_patches_every_linear_attention_layer_and_is_idempotent(model):
    install_fused_causal_conv(model)
    patched = {id(m): type(m) for m in model.modules() if isinstance(m, GatedDeltaNet)}
    assert patched and all(cls is not GatedDeltaNet for cls in patched.values())

    install_fused_causal_conv(model)

    assert {id(m): type(m) for m in model.modules() if isinstance(m, GatedDeltaNet)} == patched


def test_installing_shares_the_reference_conv_weight(model):
    """The kernel reads `self.conv1d.weight` directly, so nothing is copied and
    `cli._assert_parameters_are_shared` stays true of the candidate."""
    before = {name: p.data_ptr() for name, p in model.named_parameters()}

    install_fused_causal_conv(model)

    assert {name: p.data_ptr() for name, p in model.named_parameters()} == before
    assert all(isinstance(m.conv1d, nn.Conv1d) for m in model.modules() if isinstance(m, GatedDeltaNet))
