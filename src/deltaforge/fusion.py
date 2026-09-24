"""What the compiler did with the program, read out of `TORCH_LOGS=output_code`.

Every finding this project holds since rental 43 came out of generated code rather than
out of a ratio: that an opaque custom op costs 21% of the decode step by splitting a chain
inductor had been fusing (`044` against `045`), and that inductor writes the grouped
nibble unpack inline and welds the final RMSNorm into the same reduction (`054` against
`056`). Both were read by hand, on a rented card, after the rental had been paid for.

This module makes that reading mechanical, so it can happen on a laptop before a slot is
filled. It has two halves, and they are separate on purpose:

* `parse_output_code` turns a dump into a `FusionReport`. It imports nothing, needs no
  GPU, and works on a CUDA dump pulled off a rental as well as on a CPU one generated
  here — the naming scheme is the same (`triton_red_fused_add_mul_7`, `cpp_fused_add_mul_0`).
* `diff_reports` says what a candidate did to the reference's graph, and turns that into
  verdicts a manifest can act on.

**What a CPU dump can and cannot tell you.** It answers, reliably and device-independently:

* **Does this registration put an opaque dispatch in the graph?** A `torch.ops.*` call in
  the wrapper is a fusion barrier wherever it runs — inductor must materialise its inputs
  and outputs and cannot fuse across it. That is a property of the dispatcher, not of the
  backend, and it is the mechanism that cost `044` 21% and `054` 3.2%.
* **Did the rewrite split the neighbourhood?** Kernel counts compared against the *same
  backend's* reference say whether a chain that was fusing still fuses.

It does **not** answer whether inductor will fuse a dequantise prologue into a matmul: the
CPU backend sends `mm` to MKL through `extern_kernels` and materialises the dequantised
weight into a buffer, where the CUDA backend generated a Triton template that fused it and
allocated nothing. A CPU dump of `056-int4-head-torch-dequant` predicts the loss that
rental 46 refuted. Read the barrier verdict here; read the prologue question off a GPU dump.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Allocation",
    "barrier_preflight",
    "FusionDiff",
    "FusionReport",
    "Kernel",
    "diff_reports",
    "opaque_kernels",
    "parse_output_code",
    "render_diff",
    "render_report",
]

#: `V0924 16:57:37.436000 118349 torch/_inductor/codecache.py:2145] [0/0] [__output_code] `
_LOG_PREFIX = re.compile(r"^[VIWED]\d{4} .*?\[__output_code\] ?")
_KERNEL_DEF = re.compile(r"^\s*((?:triton|cpp)_\w*?fused_\w*?_?\d+)\s*=\s*async_compile\.")
_KERNEL_CLASS = re.compile(r"^(triton_(?:poi|red|per|tem|mm|unk)|cpp)_")
_ALLOCATION = re.compile(r"^\s*(\w+)\s*=\s*empty_strided_\w+\(\(([^)]*)\),\s*\([^)]*\),\s*torch\.(\w+)")
_EXTERN = re.compile(r"extern_kernels\.(\w+)\(")
_CUSTOM_OP = re.compile(r"torch\.ops\.(?!aten|inductor|_quantized|prims|_c10d)(\w+)\.(\w+)[.(]")
_LAUNCH = re.compile(r"^\s*((?:triton|cpp)_\w*?fused_\w*?_?\d+)[.(]")


@dataclass(frozen=True)
class Kernel:
    """One generated kernel, named after everything inductor fused into it."""

    name: str
    #: `triton_poi`, `triton_red`, `triton_per`, `triton_tem` or `cpp`.
    kind: str
    #: The op-name blob between `fused_` and the trailing index, kept whole.
    #:
    #: Inductor joins op names with `_` and several of them contain `_` themselves
    #: (`__rshift__`, `_to_copy`, `unsafe_view`), so splitting it into a list would invent
    #: boundaries that are not there. `mentions` asks the only question worth asking of it.
    ops: str

    def mentions(self, op: str) -> bool:
        return op in self.ops


@dataclass(frozen=True)
class Allocation:
    """A buffer the graph allocates. ``elements`` is what makes a weight-sized one visible."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def elements(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count


@dataclass(frozen=True)
class FusionReport:
    kernels: tuple[Kernel, ...]
    allocations: tuple[Allocation, ...]
    #: `extern_kernels.convolution`, `extern_kernels.mm` — one entry per call site.
    extern_calls: tuple[str, ...]
    #: Dispatches to a custom op namespace: opaque to inductor, and therefore a barrier.
    custom_op_dispatches: tuple[str, ...]
    #: Kernel call sites, which is what the launch census counts. Not the same as
    #: `len(kernels)`: a fully unrolled decode graph calls one kernel many times.
    launches: int

    @property
    def kernel_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for kernel in self.kernels:
            counts[kernel.kind] = counts.get(kernel.kind, 0) + 1
        return counts

    @property
    def largest_allocation(self) -> Allocation | None:
        return max(self.allocations, key=lambda a: a.elements, default=None)

    def kernels_mentioning(self, op: str) -> tuple[Kernel, ...]:
        return tuple(k for k in self.kernels if k.mentions(op))

    def fused_together(self, *ops: str) -> bool:
        """True when one kernel carries all of ``ops``.

        This is the question `056`'s dump answered: `__rshift__`, `mm`, `mean` and `add`
        in one reduction means the unpack, the matmul, the RMSNorm and the residual are one
        pass over the hidden state.
        """
        return any(all(kernel.mentions(op) for op in ops) for kernel in self.kernels)


def parse_output_code(text: str) -> FusionReport:
    """Parse a `TORCH_LOGS=output_code` dump, with or without its log-line prefixes."""
    kernels: list[Kernel] = []
    allocations: list[Allocation] = []
    externs: list[str] = []
    custom_ops: list[str] = []
    launches = 0
    seen_kernels: set[str] = set()

    for raw in text.splitlines():
        line = _LOG_PREFIX.sub("", raw)

        definition = _KERNEL_DEF.match(line)
        if definition:
            name = definition.group(1)
            if name not in seen_kernels:
                seen_kernels.add(name)
                kernels.append(Kernel(name=name, kind=_kind_of(name), ops=_ops_of(name)))
            continue

        launch = _LAUNCH.match(line)
        if launch:
            launches += 1

        allocation = _ALLOCATION.match(line)
        if allocation:
            name, dims, dtype = allocation.groups()
            allocations.append(Allocation(name=name, shape=_shape_of(dims), dtype=dtype))

        externs.extend(f"extern_kernels.{name}" for name in _EXTERN.findall(line))
        custom_ops.extend(f"torch.ops.{ns}.{op}" for ns, op in _CUSTOM_OP.findall(line))

    return FusionReport(
        kernels=tuple(kernels),
        allocations=tuple(allocations),
        extern_calls=tuple(externs),
        custom_op_dispatches=tuple(custom_ops),
        launches=launches,
    )


def _kind_of(name: str) -> str:
    match = _KERNEL_CLASS.match(name)
    return match.group(1) if match else name.split("_fused")[0]


def _ops_of(name: str) -> str:
    _, _, rest = name.partition("fused_")
    return rest.rsplit("_", 1)[0] if rest else ""


def _shape_of(dims: str) -> tuple[int, ...]:
    out: list[int] = []
    for piece in dims.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.append(int(piece))
        except ValueError:
            # A symbolic dimension (`s0`) is not a size we can multiply. Treated as 1 so
            # the rest of the shape is still readable rather than dropping the buffer.
            out.append(1)
    return tuple(out)


@dataclass(frozen=True)
class FusionDiff:
    reference: FusionReport
    candidate: FusionReport

    @property
    def kernel_delta(self) -> dict[str, int]:
        kinds = set(self.reference.kernel_counts) | set(self.candidate.kernel_counts)
        return {
            kind: self.candidate.kernel_counts.get(kind, 0) - self.reference.kernel_counts.get(kind, 0)
            for kind in sorted(kinds)
        }

    @property
    def introduced_barriers(self) -> tuple[str, ...]:
        """Custom-op dispatches the candidate has and the reference does not."""
        introduced = set(self.candidate.custom_op_dispatches) - set(self.reference.custom_op_dispatches)
        return tuple(sorted(introduced))

    @property
    def allocation_delta(self) -> int:
        return len(self.candidate.allocations) - len(self.reference.allocations)

    @property
    def extern_delta(self) -> int:
        return len(self.candidate.extern_calls) - len(self.reference.extern_calls)

    def verdicts(self) -> tuple[str, ...]:
        """Sentences a manifest can act on, worst first. Empty means nothing was detected."""
        out: list[str] = []
        barriers = self.introduced_barriers
        if barriers:
            out.append(
                f"BARRIER: the candidate dispatches {', '.join(barriers)}, which inductor cannot "
                "fuse across. It pays its own kernel plus whatever the compiler can no longer "
                "fuse in the kernels next door -- 21% of the whole decode step on rental 45. "
                "Express it as torch operations and measure the pair."
            )
        largest_ref = self.reference.largest_allocation
        largest_cand = self.candidate.largest_allocation
        if largest_ref and largest_cand and largest_cand.elements > largest_ref.elements:
            out.append(
                f"BUFFER: the candidate's largest allocation is {largest_cand.name} at "
                f"{largest_cand.elements} elements against the reference's {largest_ref.elements}. "
                "On a CPU dump this is expected for anything feeding `mm` and says nothing about "
                "CUDA; on a GPU dump a weight-sized buffer is the materialisation that would sink "
                "the candidate by arithmetic alone."
            )
        splits = {kind: delta for kind, delta in self.kernel_delta.items() if delta > 0}
        if splits:
            out.append(
                "SPLIT: the candidate defines more kernels than the reference "
                f"({', '.join(f'{k} {v:+d}' for k, v in splits.items())}). A rewrite that adds "
                "kernels has taken work out of a fusion the compiler was already doing."
            )
        return tuple(out)


def diff_reports(reference: FusionReport, candidate: FusionReport) -> FusionDiff:
    return FusionDiff(reference=reference, candidate=candidate)


def render_report(report: FusionReport, label: str = "report") -> str:
    counts = ", ".join(f"{kind} {n}" for kind, n in sorted(report.kernel_counts.items())) or "none"
    largest = report.largest_allocation
    unique_ops = sorted(set(report.custom_op_dispatches))
    named = f" ({', '.join(unique_ops)})" if unique_ops else ""
    lines = [
        f"{label}:",
        f"  kernels defined   {counts}",
        f"  kernel launches   {report.launches}",
        f"  extern calls      {len(report.extern_calls)}",
        f"  allocations       {len(report.allocations)}"
        + (f" (largest {largest.name}, {largest.elements} elements)" if largest else ""),
        f"  custom-op calls   {len(report.custom_op_dispatches)}" + named,
    ]
    return "\n".join(lines)


def render_diff(diff: FusionDiff) -> str:
    lines = [
        render_report(diff.reference, "reference"),
        render_report(diff.candidate, "candidate"),
        "delta:",
        "  kernels           " + ", ".join(f"{k} {v:+d}" for k, v in diff.kernel_delta.items()),
        f"  extern calls      {diff.extern_delta:+d}",
        f"  allocations       {diff.allocation_delta:+d}",
    ]
    verdicts = diff.verdicts()
    lines.append("")
    lines.extend(verdicts or ("no barrier, no new buffer, no split: nothing here refuses the slot.",))
    return "\n".join(lines)


# -- the static half: a registration that is a barrier by construction -----------------


def opaque_kernels(names: tuple[str, ...] | list[str], source: object | None = None) -> tuple[str, ...]:
    """Which of ``names`` reach inductor as an opaque dispatch, checked not guessed.

    A kernel registered through `torch.library.custom_op` is a `CustomOpDef`, and inductor
    cannot fuse across one: it materialises the op's inputs and outputs and splits whatever
    producer chain ran through it. That cost is paid in the *neighbouring* kernels, where
    nobody looks — 21% of the whole decode step for the causal conv (`044` against `045`)
    and 3.2% for the int4 head (`054` against `056`), which is more than either kernel was
    ever going to win.

    `torch.library.triton_op` is the registration that is meant to be visible, and it is
    reported here too: on torch 2.11 it does not trace this project's kernels at all
    (`055` died under `FakeTensorMode` reaching `.data_ptr()`), so on this toolchain it is
    not the escape from the barrier that it looks like.

    This needs no dump, no GPU and no run: it reads what the registry holds.
    """
    import torch  # noqa: PLC0415

    from .kernels import REGISTRY  # noqa: PLC0415

    registry = REGISTRY if source is None else source
    opaque: list[str] = []
    for name in names:
        impl = registry.get(name).impl
        if isinstance(impl, torch._library.custom_ops.CustomOpDef):
            opaque.append(name)
    return tuple(opaque)


def barrier_preflight(batch) -> tuple[str, ...]:
    """Slots that install a fusion barrier without the batch measuring the alternative.

    The experiment that settled both of this project's champions is a pair of slots in one
    process that differ only in whether the arithmetic reaches inductor as an opaque op or
    as torch it may fuse. A batch that installs a barrier and never measures the same
    program without one cannot tell "this kernel is slow" from "this registration costs its
    neighbours", which is three rentals of this project's history.

    Returns the slugs that install an opaque kernel and are neither half of a declared
    contrast: they do not name an earlier slot, and no later slot names them.
    """
    partnered = {hyp.contrast_with for hyp in batch if hyp.contrast_with}
    unpartnered: list[str] = []
    for hyp in batch:
        if not opaque_kernels(hyp.kernels):
            continue
        if hyp.contrast_with is None and hyp.slug not in partnered:
            unpartnered.append(hyp.slug)
    return tuple(unpartnered)
