"""What batch 009's three head registrations can be checked for without a GPU.

The claim under test is about *how a program reaches inductor*, and most of that is
decidable on a laptop: whether the op is registered at all, whether the two alternative
heads install exactly where the champion installs, and — for the one candidate with no
Triton in it — whether it computes what it is supposed to compute.

`AGENT.md` §8: a Triton body is a string a GPU compiles, and a slot whose registration
silently fell back would report a plausible ratio for a candidate the manifest does not
name. That is the failure these assert against.
"""

from __future__ import annotations

import pytest
import torch

from ..config import tiny_config
from ..reference import ReferenceModel
from .tiled_gemv import (
    TiledLMHead,
    dequantise_int4_k_major,
    head_shape_key,
    quantise_int4_k_major,
)
from .visible_int4_head import (
    DequantInt4LMHead,
    VisibleInt4LMHead,
    install_int4_head_torch_dequant,
    install_int4_head_triton_op,
    install_int4_mlp_torch_dequant,
    torch_dequant_gemv_int4,
)


@pytest.fixture
def model():
    return ReferenceModel(tiny_config()).eval().to(torch.bfloat16)


def test_the_triton_op_is_actually_registered_as_one():
    """The slot's entire content is the registration, so its absence must not be silent.

    `torch.library.triton_op` builds the op whether or not Triton is installed — only
    `wrap_triton` needs the compiler, and that is not reached until a call — so this is
    checkable on the CPU box. If the torch on the rented card lacked `triton_op`, the
    module raises rather than running the champion's body unregistered.
    """
    assert hasattr(torch.ops.deltaforge, "tiled_gemv_int4_visible")

    schema = str(torch.ops.deltaforge.tiled_gemv_int4_visible.default._schema)
    assert schema.startswith("deltaforge::tiled_gemv_int4_visible(Tensor x, Tensor packed, Tensor scale,")
    assert schema.endswith("-> Tensor")


def test_the_torch_dequant_head_computes_the_champions_reference_expression(model):
    """`056` must be the same function as `054`, or the pair measures two things.

    The expression is the one `tiled_int4_correctness_checks` uses as the Triton kernel's
    reference — dequantise, **round the scaled weight to bf16 because the kernel does**,
    then accumulate in fp32. Batch 003 lost a slot to a reference that dequantised in fp32
    while the implementation rounded first, so this is asserted rather than assumed.
    """
    weight = model.lm_head_weight
    quantised = quantise_int4_k_major(weight.detach())
    packed, scale, group = quantised.qweight, quantised.scale, quantised.group_size
    torch.manual_seed(0)
    x = torch.randn(1, 1, packed.shape[0] * 2, dtype=weight.dtype)

    expected = (
        (
            x.reshape(-1, x.shape[-1]).float()
            @ dequantise_int4_k_major(packed, scale, group_size=group).to(x.dtype).float()
        )
        .to(x.dtype)
        .reshape(1, 1, packed.shape[1])
    )
    got = torch_dequant_gemv_int4(x, packed, scale, group)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("install", "head_cls"),
    [
        (install_int4_head_triton_op, VisibleInt4LMHead),
        (install_int4_head_torch_dequant, DequantInt4LMHead),
    ],
    ids=["triton_op", "torch_dequant"],
)
def test_both_alternatives_install_exactly_where_the_champion_installs(install, head_cls, model):
    """Same class swap, same attribute, same quantised weights — only `forward` differs.

    If these installed differently from `_install_head`, the slot pair would compare two
    installations rather than two registrations, and the one variable the batch is built
    around would not be the only one.
    """
    before = type(model)

    install(model)

    assert isinstance(model.tiled_lm_head, head_cls)
    assert type(model).project_logits is not ReferenceModel.project_logits
    assert type(model) is not before
    assert model.tiled_lm_head.kind == "int4"
    assert head_shape_key(model.tiled_lm_head)[0] == "int4"


@pytest.mark.parametrize(
    "install",
    [install_int4_head_triton_op, install_int4_head_torch_dequant, install_int4_mlp_torch_dequant],
    ids=["triton_op", "torch_dequant", "mlp"],
)
def test_installing_is_idempotent(install, model):
    install(model)
    once = {name: type(module) for name, module in model.named_modules()}

    install(model)

    assert {name: type(module) for name, module in model.named_modules()} == once


def test_the_alternative_heads_are_not_the_champions_head(model):
    """`054` and `055` must be distinguishable, or the batch measured the same slot twice."""
    install_int4_head_triton_op(model)

    assert not isinstance(model.tiled_lm_head, TiledLMHead)


def test_a_pinned_tile_does_not_follow_the_alternative_heads(model):
    """`_TUNED` is process-global, and these installers clear it exactly as `_install_head` does."""
    from .tiled_gemv import _heuristic_shape, _launch_shape, pin_launch_shape

    install_int4_head_triton_op(model)
    kind, n, k = head_shape_key(model.tiled_lm_head)
    pin_launch_shape(kind, n, k, (128, 64, 1, 8, 3))

    fresh = ReferenceModel(tiny_config()).eval().to(torch.bfloat16)
    install_int4_head_torch_dequant(fresh)

    assert _launch_shape(n, k, kind) == _heuristic_shape(n, k)


def test_the_mlp_installer_leaves_the_head_and_the_attention_projections_alone(model):
    """52.75% of the bytes is the MLP; folding other sites in is what made an aggregate
    byte rate unreadable across batches 003 and 004."""
    install_int4_mlp_torch_dequant(model)

    changed = {name for name, module in model.named_modules() if type(module).__name__ == "DequantInt4Linear"}
    assert changed
    assert all(name.split(".")[-1] in {"gate_proj", "up_proj", "down_proj"} for name in changed)
    assert not hasattr(model, "tiled_lm_head")


def test_the_mlp_installer_shares_the_reference_parameters(model):
    """`w_k_major` and `qscale` are buffers; `cli._assert_parameters_are_shared` walks
    `named_parameters` and rejects anything without a counterpart in the reference."""
    before = {name: p.data_ptr() for name, p in model.named_parameters()}

    install_int4_mlp_torch_dequant(model)

    assert {name: p.data_ptr() for name, p in model.named_parameters()} == before
