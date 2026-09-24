"""Parser tests, written against the shapes the last three rentals' dumps actually had."""

from __future__ import annotations

from .fusion import diff_reports, parse_output_code, render_diff

# The reference's tail, as rental 46 recorded it: one reduction carrying the matmul, the
# final RMSNorm and the residual add, and the logits as its largest buffer.
REFERENCE = """\
V0923 11:02:03.100000 4242 torch/_inductor/codecache.py:2145] [0/0] [__output_code] \
triton_poi_fused_add_mul_0 = async_compile.triton('triton_poi_fused_add_mul_0', '''
V0923 11:02:03.100000 4242 torch/_inductor/codecache.py:2145] [0/0] [__output_code] \
triton_red_fused__to_copy__unsafe_view_add_mean_mm_mul_pow_rsqrt_slice_t_view_43 \
= async_compile.triton('x', '''
V0923 11:02:03.100000 4242 torch/_inductor/codecache.py:2145] [0/0] [__output_code]     \
buf7 = empty_strided_cuda((1, 2560), (2560, 1), torch.bfloat16)
V0923 11:02:03.100000 4242 torch/_inductor/codecache.py:2145] [0/0] [__output_code]     \
buf9 = empty_strided_cuda((1, 248320), (248320, 1), torch.bfloat16)
V0923 11:02:03.100000 4242 torch/_inductor/codecache.py:2145] [0/0] [__output_code]     \
extern_kernels.convolution(buf3, arg2_1, stride=(1,), padding=(0,), groups=4096)
V0923 11:02:03.100000 4242 torch/_inductor/codecache.py:2145] [0/0] [__output_code]     \
triton_poi_fused_add_mul_0.run(buf7, arg1_1, 2560, stream=stream0)
"""

# The same region with the four causal-conv taps behind `torch.library.custom_op`: the
# convolution is gone, an opaque dispatch is in its place, and the graph carries buffers
# the reference did not need.
CUSTOM_OP = """\
triton_poi_fused_add_mul_0 = async_compile.triton('triton_poi_fused_add_mul_0', '''
triton_red_fused__to_copy__unsafe_view_add_mean_mm_mul_pow_rsqrt_slice_t_view_43 \
= async_compile.triton('x', '''
triton_red_fused_sum_44 = async_compile.triton('x', '''
    buf7 = empty_strided_cuda((1, 2560), (2560, 1), torch.bfloat16)
    buf8 = empty_strided_cuda((1, 4096, 3), (12288, 3, 1), torch.bfloat16)
    buf9 = empty_strided_cuda((1, 248320), (248320, 1), torch.bfloat16)
    buf10 = torch.ops.deltaforge.fused_causal_conv_step.default(buf3, arg2_1, buf8)
    triton_poi_fused_add_mul_0.run(buf7, arg1_1, 2560, stream=stream0)
"""


def test_it_reads_kernels_allocations_and_externs_out_of_a_prefixed_dump():
    report = parse_output_code(REFERENCE)

    assert report.kernel_counts == {"triton_poi": 1, "triton_red": 1}
    assert report.extern_calls == ("extern_kernels.convolution",)
    assert report.custom_op_dispatches == ()
    assert [a.name for a in report.allocations] == ["buf7", "buf9"]
    assert report.largest_allocation.elements == 248320


def test_a_launch_is_counted_separately_from_a_definition():
    """The decode graph is fully unrolled, so one kernel is called many times."""
    report = parse_output_code(REFERENCE)

    assert len(report.kernels) == 2
    assert report.launches == 1


def test_it_sees_the_whole_fused_tail_in_one_kernel():
    """`054` against `056` turned on exactly this: unpack, mm, norm and residual together."""
    report = parse_output_code(REFERENCE)

    assert report.fused_together("mm", "mean", "rsqrt", "add")
    assert not report.fused_together("mm", "bitwise_and")
    assert [k.name for k in report.kernels_mentioning("mm")] == [
        "triton_red_fused__to_copy__unsafe_view_add_mean_mm_mul_pow_rsqrt_slice_t_view_43"
    ]


def test_an_opaque_custom_op_is_read_as_a_dispatch_and_not_as_an_aten_call():
    report = parse_output_code(CUSTOM_OP)

    assert report.custom_op_dispatches == ("torch.ops.deltaforge.fused_causal_conv_step",)


def test_the_diff_names_the_barrier_the_candidate_introduced():
    diff = diff_reports(parse_output_code(REFERENCE), parse_output_code(CUSTOM_OP))

    assert diff.introduced_barriers == ("torch.ops.deltaforge.fused_causal_conv_step",)
    assert diff.extern_delta == -1
    assert diff.allocation_delta == 1
    assert diff.kernel_delta == {"triton_poi": 0, "triton_red": 1}
    assert diff.verdicts()[0].startswith("BARRIER")


def test_a_candidate_that_changes_nothing_structural_gets_no_verdict():
    diff = diff_reports(parse_output_code(REFERENCE), parse_output_code(REFERENCE))

    assert diff.verdicts() == ()
    assert "nothing here refuses the slot" in render_diff(diff)


def test_a_cpu_dump_parses_with_the_same_parser():
    """A laptop dump and a rental dump differ in the kernel prefix and nothing else."""
    report = parse_output_code(
        "cpp_fused___rshift____to_copy_mul_0 = async_compile.cpp_pybinding(['const uint8_t*'], r'''\n"
        "    buf0 = empty_strided_cpu((64, 64), (64, 1), torch.float32)\n"
        "    extern_kernels.mm(arg2_1, buf0, out=buf1)\n"
    )

    assert report.kernel_counts == {"cpp": 1}
    assert report.kernels[0].mentions("__rshift__")
    assert report.extern_calls == ("extern_kernels.mm",)


def test_a_symbolic_dimension_does_not_lose_the_buffer():
    report = parse_output_code("    buf0 = empty_strided_cuda((s0, 2560), (2560, 1), torch.bfloat16)\n")

    assert [a.name for a in report.allocations] == ["buf0"]


# -- the static half ------------------------------------------------------------------


def test_opaque_kernels_names_the_registrations_inductor_cannot_fuse_across():
    """Read off the registry, not off a dump: a `CustomOpDef` is a barrier anywhere.

    Both kernels this project retired are on the left, and both champions it ships are on
    the right — which is the finding, stated as a property of the registration.
    """
    from .fusion import opaque_kernels

    assert opaque_kernels(["fused_causal_conv", "tiled_int4_head"]) == (
        "fused_causal_conv",
        "tiled_int4_head",
    )
    assert opaque_kernels(["inline_causal_conv", "int4_head_torch_dequant", "static_decode_cache"]) == ()


def test_barrier_preflight_passes_a_batch_that_measures_both_sides():
    from .batch import Batch, Hypothesis
    from .fusion import barrier_preflight

    def hyp(slug, kernels, contrast=None):
        return Hypothesis(
            slug=slug,
            kernels=kernels,
            category="A",
            byte_share=0.1,
            mechanism="m",
            prediction="win",
            rationale="r" * 90,
            correctness="approximate",
            top1_threshold=1.0,
            kl_threshold=1e-6,
            contrast_with=contrast,
        )

    paired = Batch(
        batch_id="010-x",
        hypotheses=(
            hyp("a-custom-op", ("fused_causal_conv",)),
            hyp("b-inline", ("inline_causal_conv",), contrast="a-custom-op"),
        ),
    )
    unpaired = Batch(batch_id="010-y", hypotheses=(hyp("a-custom-op", ("fused_causal_conv",)),))

    assert barrier_preflight(paired) == ()
    assert barrier_preflight(unpaired) == ("a-custom-op",)
