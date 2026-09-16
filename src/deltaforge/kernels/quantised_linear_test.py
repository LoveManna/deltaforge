"""CPU tests for batch 003's quantisers, installers and their invariants.

Triton is absent here, so nothing below runs a kernel. What it *can* test is everything
the kernel depends on being told: the quantisation arithmetic, the int4 packing, which
modules each install touches, that the reference's parameters stay shared, and the two
numeric facts the GPU path assumes — that a round trip through int8 lands within half a
scale step, and that the packing is its own inverse.

The dequantisers are not merely helpers: `dequantise_int8` and `dequantise_int4` are the
*references* the layer-1 GPU gate compares the Triton kernels against, so a bug in them
would make the gate agree with the kernel about the wrong answer. They are tested here
against an independent, deliberately naive implementation for that reason.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ..batch import scoped_registry
from ..batches import BATCH_003
from ..config import tiny_config
from ..model import apply_champions
from ..reference import ReferenceModel
from . import REGISTRY
from .quantised_linear import (
    GEMV_MAX_ROWS,
    INT4_MAX,
    INT8_MAX,
    Bf16GemvLinear,
    Int4GemvLinear,
    Int8DequantLinear,
    Int8GemvLinear,
    dequantise_int4,
    dequantise_int8,
    group_size_for,
    quantise_int4_grouped,
    quantise_int8_per_channel,
)


@pytest.fixture
def model():
    return ReferenceModel(tiny_config())


# -- the quantisers -------------------------------------------------------------------


def test_int8_uses_one_scale_per_output_channel():
    weight = torch.randn(7, 64)
    weight[3] *= 100.0  # one row far out of scale with the rest

    quantised = quantise_int8_per_channel(weight)

    assert quantised.qweight.dtype is torch.int8
    assert quantised.scale.shape == (7,)
    # Per-channel is the whole point: a tensor-wide scale would spend the 8-bit range on
    # row 3 and quantise every other row to a handful of levels.
    assert quantised.scale[3] > 10 * quantised.scale[0]


def test_int8_round_trip_lands_within_half_a_scale_step():
    """The bound the correctness thresholds are derived from.

    Symmetric rounding cannot be off by more than half a step by construction; if it ever
    is, the scale is being computed against the wrong axis and every downstream KL number
    would be measuring that bug instead of quantisation.
    """
    weight = torch.randn(16, 128)

    quantised = quantise_int8_per_channel(weight)
    restored = dequantise_int8(quantised.qweight, quantised.scale)

    assert torch.all((restored - weight).abs() <= quantised.scale.unsqueeze(1) / 2 + 1e-6)


def test_int8_saturates_rather_than_wrapping():
    quantised = quantise_int8_per_channel(torch.randn(4, 256) * 50)
    assert int(quantised.qweight.abs().max()) <= INT8_MAX


def test_a_zero_row_does_not_produce_nan():
    """A divide by zero here would make plausible-looking NaN logits, not a crash."""
    weight = torch.randn(3, 32)
    weight[1] = 0.0

    quantised = quantise_int8_per_channel(weight)

    assert torch.isfinite(quantised.scale).all()
    assert torch.isfinite(dequantise_int8(quantised.qweight, quantised.scale)).all()


def test_chunking_does_not_change_the_result(monkeypatch):
    """The row chunking exists to bound peak memory, so it must be invisible."""
    from . import quantised_linear

    weight = torch.randn(64, 256)
    whole = quantise_int8_per_channel(weight)
    monkeypatch.setattr(quantised_linear, "ROW_CHUNK_ELEMENTS", 256 * 3)
    chunked = quantise_int8_per_channel(weight)

    assert torch.equal(whole.qweight, chunked.qweight)
    assert torch.equal(whole.scale, chunked.scale)


# -- int4 packing ---------------------------------------------------------------------


@pytest.mark.parametrize("k", [256, 512, 2560, 9216, 4096])
def test_the_real_models_widths_all_take_the_128_group(k):
    assert group_size_for(k) == 128


def test_a_narrow_weight_falls_back_to_one_group_per_half():
    # tiny_config's out_proj is K=64. The packing pairs j with j + K/2, so a group must
    # divide K/2 as well as K, which 128 does not here.
    assert group_size_for(64) == 32


def test_group_size_refuses_an_odd_k():
    with pytest.raises(ValueError, match="even K"):
        group_size_for(255)


def test_int4_packs_two_values_per_byte_and_unpacks_to_the_same_ones():
    weight = torch.randn(5, 512)

    quantised = quantise_int4_grouped(weight)

    assert quantised.qweight.dtype is torch.uint8
    assert quantised.qweight.shape == (5, 256)
    assert quantised.scale.shape == (5, 512 // 128)

    # Independent unpack: byte j holds element j low and element j + K/2 high, offset by 8.
    packed = quantised.qweight.to(torch.int16)
    low = (packed & 0x0F) - 8
    high = (packed >> 4) - 8
    naive = torch.cat((low, high), dim=1).float()
    naive = (naive.reshape(5, 4, 128) * quantised.scale.unsqueeze(2)).reshape(5, 512)

    assert torch.equal(naive, dequantise_int4(quantised.qweight, quantised.scale, group_size=128))


def test_int4_round_trip_lands_within_half_a_group_step():
    weight = torch.randn(8, 256)

    quantised = quantise_int4_grouped(weight)
    restored = dequantise_int4(quantised.qweight, quantised.scale, group_size=quantised.group_size)

    step = quantised.scale.repeat_interleave(quantised.group_size, dim=1)
    assert torch.all((restored - weight).abs() <= step / 2 + 1e-5)
    assert int(((quantised.qweight & 0x0F).to(torch.int16) - 8).abs().max()) <= INT4_MAX


def test_int4_is_coarser_than_int8_on_the_same_weight():
    """Sanity on the direction of the trade the batch is making."""
    weight = torch.randn(8, 256)

    err8 = (dequantise_int8(*_q8(weight)) - weight).abs().mean()
    q4 = quantise_int4_grouped(weight)
    err4 = (dequantise_int4(q4.qweight, q4.scale, group_size=q4.group_size) - weight).abs().mean()

    assert err4 > err8


def _q8(weight):
    quantised = quantise_int8_per_channel(weight)
    return quantised.qweight, quantised.scale


# -- installation ---------------------------------------------------------------------


def _install(model, slug):
    apply_champions(model, scoped_registry(BATCH_003.get(slug), REGISTRY))


def _linear_classes(model) -> dict[str, str]:
    return {
        name: type(module).__name__ for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }


def test_int8_mlp_touches_the_mlp_and_nothing_else(model):
    _install(model, "011-int8-mlp")

    classes = _linear_classes(model)
    patched = {name for name, cls in classes.items() if cls == "Int8GemvLinear"}
    assert patched
    assert all(".mlp." in name for name in patched)
    assert not any(name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) for name in patched)


def test_int8_all_linear_touches_every_projection_in_every_layer(model):
    _install(model, "012-int8-all-linear")

    classes = _linear_classes(model)
    in_layers = {name: cls for name, cls in classes.items() if name.startswith("layers.")}
    assert in_layers
    assert set(in_layers.values()) == {"Int8GemvLinear"}


def test_only_the_full_variants_replace_the_lm_head(model):
    _install(model, "012-int8-all-linear")
    assert type(model).__name__ == "ReferenceModel"

    full = ReferenceModel(tiny_config())
    _install(full, "013-int8-full")
    assert type(full).__name__ == "QuantisedHeadModel"
    assert hasattr(full, "quantised_lm_head")


def test_the_quantised_head_holds_the_embedding_matrix_it_replaced():
    """The head has no `nn.Linear`, so nothing else would catch it quantising the wrong
    tensor — `lm_head_weight` is a property, and picking up `norm.weight` or a layer's
    projection instead would still produce a model that runs and emits plausible logits.

    The op itself cannot run here: `require_cuda` refuses a CPU tensor by design, which is
    why this checks the weights rather than the logits. The GPU gate checks the logits.
    """
    torch.manual_seed(0)
    reference = ReferenceModel(tiny_config()).eval()
    candidate = ReferenceModel(tiny_config()).eval()
    candidate.load_state_dict(reference.state_dict())

    _install(candidate, "013-int8-full")

    head = candidate.quantised_lm_head
    assert head.qweight.shape == reference.lm_head_weight.shape
    restored = dequantise_int8(head.qweight, head.qscale)
    error = (restored - reference.lm_head_weight.float()).abs().max()
    assert error > 0, "the head was quantised but lost no precision at all"
    assert error <= head.qscale.max() / 2 + 1e-6


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("009-gemv-bf16-control", Bf16GemvLinear),
        ("010-int8-dequant-torch", Int8DequantLinear),
        ("011-int8-mlp", Int8GemvLinear),
        ("012-int8-all-linear", Int8GemvLinear),
        ("013-int8-full", Int8GemvLinear),
        ("014-int4-full", Int4GemvLinear),
    ],
)
def test_each_install_produces_its_own_module_class(slug, expected, model):
    _install(model, slug)
    assert expected.__name__ in set(_linear_classes(model).values())


def test_the_quantised_copies_are_buffers_not_parameters(model):
    """`cli._assert_parameters_are_shared` walks named_parameters and would reject a
    quantised weight registered as one — it has no counterpart in the reference."""
    before = {name for name, _ in model.named_parameters()}

    _install(model, "012-int8-all-linear")

    assert {name for name, _ in model.named_parameters()} == before
    buffers = {name for name, _ in model.named_buffers()}
    assert any(name.endswith("qweight") for name in buffers)
    assert any(name.endswith("qscale") for name in buffers)


def test_the_bf16_control_quantises_nothing(model):
    _install(model, "009-gemv-bf16-control")

    assert not any(name.endswith("qweight") for name, _ in model.named_buffers())


def test_the_reference_parameters_are_still_shared_after_installing(model):
    from ..cli import _assert_parameters_are_shared

    candidate = ReferenceModel(tiny_config())
    candidate.load_state_dict(model.state_dict(), assign=True)
    _install(candidate, "013-int8-full")

    _assert_parameters_are_shared(model, candidate)  # raises SystemExit on a violation


# -- the one place the op does not run its own kernel ---------------------------------


def test_the_gemv_threshold_covers_every_workload_the_harness_scores():
    """Above `GEMV_MAX_ROWS` the op dequantises instead of running the kernel.

    That is only ever meant to happen in the untimed prefill. If a workload's batch size
    ever climbed past the threshold, the scored region would quietly stop measuring the
    kernel — the exact class of failure blocker 16 was.
    """
    from ..cli import DEFAULT_WORKLOADS

    largest = max(workload["batch_size"] for workload in DEFAULT_WORKLOADS.values())
    assert largest <= GEMV_MAX_ROWS


# -- launch geometry and op registration ----------------------------------------------


@pytest.mark.parametrize(
    ("n", "label"),
    [
        (9216, "mlp gate/up"),
        (2560, "mlp down, o_proj, out_proj"),
        (8192, "in_proj_qkv"),
        (4096, "in_proj_z"),
        (248320, "tied lm head"),
        (1024, "k_proj and v_proj"),
        (8192, "q_proj, doubled for the output gate"),
    ],
)
def test_every_projection_in_the_model_gets_enough_programs_to_fill_a_card(n, label):
    """A GEMV's only parallelism is over output channels, so the grid IS ``N / BLOCK_N``.

    An RTX 5090 has 170 SMs and a 4090 has 128. A projection that launches fewer programs
    than the card has SMs loses on idle hardware rather than on bytes, and the result would
    read as 'quantisation did not help'.
    """
    from .quantised_linear import _launch_shape

    block_n, block_k, _ = _launch_shape(n)
    programs = -(-n // block_n)

    assert programs >= 128, f"{label}: {programs} programs is below a 4090's SM count"
    assert block_n * block_k == 4096


def test_the_two_gating_projections_cannot_be_spread_and_are_negligible():
    """`in_proj_a` and `in_proj_b` are 32 output channels wide, so no block size reaches a
    full card. Recorded rather than hidden: they are 0.03% of per-token bytes, and the fix
    would be split-K, which is a different kernel and a different hypothesis."""
    from .quantised_linear import _launch_shape

    block_n, _, _ = _launch_shape(32)
    assert -(-32 // block_n) < 128


def test_the_tile_stays_constant_as_the_rows_per_program_change():
    from .quantised_linear import _launch_shape

    narrow = _launch_shape(2560)
    wide = _launch_shape(9216)

    assert narrow[0] < wide[0]
    assert narrow[1] > wide[1]
    assert narrow[0] * narrow[1] == wide[0] * wide[1]


@pytest.mark.parametrize("name", ["bf16_gemv", "int8_gemv", "int4_gemv"])
def test_each_custom_op_is_registered_under_the_deltaforge_namespace(name):
    """Opaque to dynamo is the point: inductor may CUDA-graph around these but may not
    decompose them back into the materialise-then-matmul they exist to avoid."""
    assert hasattr(torch.ops.deltaforge, name)


def test_the_fake_kernels_give_dynamo_the_right_shape_and_dtype():
    """`torch.compile` traces these without ever running them, so the fake kernel is what
    the graph is built from. A wrong shape here is a compile-time failure on a rented box."""
    from .quantised_linear import _bf16_gemv_fake, _int4_gemv_fake, _int8_gemv_fake

    x = torch.randn(1, 1, 256, dtype=torch.bfloat16)
    assert _bf16_gemv_fake(x, torch.empty(512, 256, dtype=torch.bfloat16)).shape == (1, 1, 512)
    assert _int8_gemv_fake(x, torch.empty(512, 256, dtype=torch.int8), torch.empty(512)).shape == (1, 1, 512)
    assert _int4_gemv_fake(x, torch.empty(512, 128, dtype=torch.uint8), torch.empty(512, 2), 128).shape == (
        1,
        1,
        512,
    )
    int8_out = _int8_gemv_fake(x, torch.empty(512, 256, dtype=torch.int8), torch.empty(512))
    assert int8_out.dtype is torch.bfloat16
