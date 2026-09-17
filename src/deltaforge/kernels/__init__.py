"""The kernel registry: ``name -> (impl, replaces, status)``.

**This package currently ships no champion.** Every registered kernel is ``RETIRED``, so
``apply_champions`` installs nothing and the ``candidate`` column is bit-identical to
``eager``. That is deliberate: it is the *identity champion* a first GPU session uses to
calibrate the harness, at zero kernel-writing risk. See ``AGENT.md``.

Each kernel module exports a callable with the *exact signature* of the reference
operation it replaces, and registers itself here. The registry's single hard invariant:
**at most one champion per replaceable reference operation.** Two champions for the same
operation would make "what does `model.py` assemble?" ambiguous, and an ambiguous
champion makes the leaderboard a lie.

Retired kernels stay registered on purpose. A retired entry plus its graveyard entry in
``docs/HYPOTHESES.md`` is what stops a later session paying to rediscover a dead end.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from enum import Enum

__all__ = [
    "CHECK_BUILDERS",
    "REGISTRY",
    "REPLACEABLE_OPS",
    "KernelEntry",
    "KernelRegistry",
    "KernelStatus",
    "RegistryError",
    "build_kernel_checks",
    "register_checks",
]


class KernelStatus(str, Enum):
    """Where a kernel stands.

    ``CANDIDATE`` — registered and correct, but not the current best.
    ``CHAMPION``  — the one implementation `model.py` assembles for this operation.
    ``RETIRED``   — beaten, or superseded. Kept registered on purpose: a retired kernel
                    plus its graveyard entry is what stops a later session re-running a
                    dead end.
    """

    CANDIDATE = "candidate"
    CHAMPION = "champion"
    RETIRED = "retired"


#: The reference operations a Triton kernel is allowed to replace. Each name corresponds
#: to a callable boundary inside `deltaforge.reference`. Adding a name here is a
#: deliberate act: it widens what a candidate may swap out, and therefore what the
#: correctness gate must cover.
REPLACEABLE_OPS = frozenset(
    {
        "rms_norm",
        "rms_norm_residual",
        "swiglu_mlp",
        "qkv_projection_rope",
        "gated_delta_rule",
        "gqa_attention",
        "kv_cache_update",
        "decode_step",
    }
)


class RegistryError(RuntimeError):
    """Raised when a registration would violate a registry invariant."""


@dataclass(frozen=True)
class KernelEntry:
    name: str
    impl: Callable[..., object]
    replaces: str
    status: KernelStatus
    hypothesis: str | None = None
    notes: str = ""


class KernelRegistry:
    """A mapping of kernel name to entry, with the champion invariant enforced.

    Constructed per-instance rather than only as a module global so tests can work on an
    isolated registry instead of mutating shared state.
    """

    def __init__(self, replaceable_ops: frozenset[str] = REPLACEABLE_OPS) -> None:
        self._replaceable_ops = replaceable_ops
        self._entries: dict[str, KernelEntry] = {}

    # -- registration ---------------------------------------------------------

    def register(
        self,
        name: str,
        impl: Callable[..., object],
        *,
        replaces: str,
        status: KernelStatus | str = KernelStatus.CANDIDATE,
        hypothesis: str | None = None,
        notes: str = "",
    ) -> KernelEntry:
        if name in self._entries:
            raise RegistryError(f"kernel {name!r} is already registered")
        if replaces not in self._replaceable_ops:
            raise RegistryError(
                f"kernel {name!r} replaces unknown operation {replaces!r}; "
                f"known operations: {sorted(self._replaceable_ops)}"
            )
        status = KernelStatus(status)
        if status is KernelStatus.CHAMPION:
            incumbent = self.champion(replaces)
            if incumbent is not None:
                raise RegistryError(
                    f"cannot register {name!r} as champion of {replaces!r}: "
                    f"{incumbent.name!r} already holds it. Use promote() instead, which "
                    "demotes the incumbent in the same step."
                )
        entry = KernelEntry(
            name=name,
            impl=impl,
            replaces=replaces,
            status=status,
            hypothesis=hypothesis,
            notes=notes,
        )
        self._entries[name] = entry
        return entry

    def promote(self, name: str) -> KernelEntry:
        """Make ``name`` the champion of its operation, retiring the incumbent.

        Atomic by construction: there is never a moment with two champions, and never a
        moment with none if there was one before.
        """
        entry = self.get(name)
        if entry.status is KernelStatus.CHAMPION:
            return entry
        incumbent = self.champion(entry.replaces)
        if incumbent is not None:
            self._entries[incumbent.name] = replace(incumbent, status=KernelStatus.RETIRED)
        promoted = replace(entry, status=KernelStatus.CHAMPION)
        self._entries[name] = promoted
        return promoted

    def retire(self, name: str) -> KernelEntry:
        entry = self.get(name)
        retired = replace(entry, status=KernelStatus.RETIRED)
        self._entries[name] = retired
        return retired

    def unregister(self, name: str) -> None:
        del self._entries[name]

    def clear(self) -> None:
        self._entries.clear()

    # -- queries --------------------------------------------------------------

    def get(self, name: str) -> KernelEntry:
        try:
            return self._entries[name]
        except KeyError:
            raise RegistryError(f"no kernel registered as {name!r}") from None

    def champion(self, replaces: str) -> KernelEntry | None:
        """The single champion for an operation, or ``None`` if it has none."""
        found = [
            entry
            for entry in self._entries.values()
            if entry.replaces == replaces and entry.status is KernelStatus.CHAMPION
        ]
        if len(found) > 1:  # pragma: no cover - register/promote make this unreachable
            raise RegistryError(f"registry invariant violated: {len(found)} champions for {replaces!r}")
        return found[0] if found else None

    def champions(self) -> dict[str, KernelEntry]:
        """Operation name -> champion entry, for every operation that has one."""
        return {op: entry for op in sorted(self._replaceable_ops) if (entry := self.champion(op)) is not None}

    def for_op(self, replaces: str) -> tuple[KernelEntry, ...]:
        return tuple(e for e in self._entries.values() if e.replaces == replaces)

    def check_invariants(self) -> None:
        """Raise if anything is inconsistent. Cheap enough to call at assembly time."""
        for op in self._replaceable_ops:
            self.champion(op)  # raises on a duplicate champion
        for name, entry in self._entries.items():
            if entry.name != name:
                raise RegistryError(f"entry keyed {name!r} carries name {entry.name!r}")
            if entry.replaces not in self._replaceable_ops:
                raise RegistryError(f"kernel {name!r} replaces unknown operation {entry.replaces!r}")

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[KernelEntry]:
        return iter(self._entries.values())

    def __repr__(self) -> str:
        return f"KernelRegistry({len(self._entries)} kernels, {len(self.champions())} champions)"


#: The process-wide registry. Kernel modules register into this on import.
REGISTRY = KernelRegistry()


#: **Kernel name** -> a builder returning that kernel's layer-1 `KernelCheck`s.
#:
#: Mirrors `model.INSTALLERS` and is keyed the same way, for the same reason: several
#: kernels replace the same reference operation — `rmsnorm_hidden` and `rmsnorm_qk` both
#: replace `rms_norm`, `gqa_decode` and `flash_decode_splitkv` both replace
#: `gqa_attention` — and an operation-keyed table would silently run one kernel's checks
#: against another kernel. A gate that checks the wrong thing is worse than no gate,
#: because it reports `pass`.
#:
#: A kernel with no checks is allowed — the end-to-end gate still covers it — but it is
#: reported as unchecked rather than as passing.
CHECK_BUILDERS: dict[str, Callable[..., tuple]] = {}


def register_checks(kernel_name: str, builder: Callable[..., tuple]) -> None:
    if kernel_name in CHECK_BUILDERS:
        raise ValueError(f"checks for kernel {kernel_name!r} are already registered")
    CHECK_BUILDERS[kernel_name] = builder


def build_kernel_checks(model, *, registry: KernelRegistry | None = None, **kwargs) -> tuple:
    """Run every champion's layer-1 checks against ``model``'s reference operations."""
    registry = REGISTRY if registry is None else registry
    checks: list = []
    for entry in registry.champions().values():
        builder = CHECK_BUILDERS.get(entry.name)
        if builder is not None:
            checks.extend(builder(model, **kwargs))
    return tuple(checks)


# -- registered kernels ------------------------------------------------------------
#
# Imported for their registration side effect. A kernel module must import cleanly with
# no Triton present, so that the CPU test suite and CI can still import the registry.

from . import fused_rmsnorm_residual as _fused_rmsnorm_residual  # noqa: E402

REGISTRY.register(
    "fused_rmsnorm_residual",
    impl=_fused_rmsnorm_residual.add_rms_norm,
    replaces="rms_norm_residual",
    status=KernelStatus.RETIRED,
    hypothesis="001-fused-rmsnorm-residual",
    notes=(
        "Fuses the residual add with the RMSNorm that follows it, and replaces the "
        "layer's other hidden-size norm with a single-pass Triton RMSNorm. "
        "RETIRED, never measured: the operations it fuses move 0.018% of per-token "
        "bytes at batch-1 decode, so its ceiling is below the harness's own noise band "
        "and no measurement could have shown a win. Kept registered, with its installer "
        "and its tests, so a later session finds the dead end already explored rather "
        "than rediscovering it. See docs/HYPOTHESES.md and docs/roofline.py."
    ),
)
register_checks("fused_rmsnorm_residual", _fused_rmsnorm_residual.correctness_checks)

# -- batch 001 kernels --------------------------------------------------------------
#
# All RETIRED, like everything else here: status in this registry means "is this the
# shipped champion", and batch mode does not read it. `batch.scoped_registry` promotes
# exactly the kernels a hypothesis names into a registry of its own, so the shipped
# default stays the identity champion and `apply_champions(model, REGISTRY)` still
# installs nothing.

from . import flash_decode_splitkv as _flash_decode_splitkv  # noqa: E402
from . import fused_rope as _fused_rope  # noqa: E402
from . import fused_swiglu as _fused_swiglu  # noqa: E402
from . import gated_delta_step as _gated_delta_step  # noqa: E402
from . import gqa_decode as _gqa_decode  # noqa: E402
from . import rmsnorm_placements as _rmsnorm_placements  # noqa: E402

REGISTRY.register(
    "rmsnorm_hidden",
    impl=_fused_rmsnorm_residual.rms_norm,
    replaces="rms_norm",
    status=KernelStatus.RETIRED,
    hypothesis="002-rmsnorm-only",
    notes=(
        "The same single-pass Triton RMSNorm as 001, installed on the three "
        "hidden_size-wide norms *without* fusing the residual add. 001 changes two things "
        "at once; this changes one, so 001 minus 002 isolates the fusion."
    ),
)
register_checks("rmsnorm_hidden", _rmsnorm_placements.standalone_correctness_checks)
register_checks("rmsnorm_qk", _rmsnorm_placements.qk_correctness_checks)

REGISTRY.register(
    "rmsnorm_qk",
    impl=_fused_rmsnorm_residual.rms_norm,
    replaces="rms_norm",
    status=KernelStatus.RETIRED,
    hypothesis="003-qk-norm-triton",
    notes=(
        "The same kernel on q_norm/k_norm: 256-wide rows instead of 2560-wide, ten times "
        "as many of them, and only in the 8 full-attention layers. A different shape "
        "regime for both inductor and the kernel, which 001 explicitly declined to enter."
    ),
)

REGISTRY.register(
    "fused_swiglu",
    impl=_fused_swiglu.silu_mul,
    replaces="swiglu_mlp",
    status=KernelStatus.RETIRED,
    hypothesis="004-fused-swiglu",
    notes=(
        "Fuses the SiLU and the gate multiply into one pass over the two 9216-wide "
        "intermediates. Ceiling 0.026% of per-token bytes; graveyarded on that arithmetic "
        "before batching made measuring it cheap."
    ),
)
register_checks("fused_swiglu", _fused_swiglu.correctness_checks)

REGISTRY.register(
    "fused_rope",
    impl=_fused_rope.apply_partial_rope,
    replaces="qkv_projection_rope",
    status=KernelStatus.RETIRED,
    hypothesis="005-fused-qkv-rope",
    notes=(
        "Partial mRoPE in one pass: rotates 64 of 256 head dims in registers and passes "
        "the other 192 through, replacing a slice/negate/cat/cat chain. Ceiling 0.004%, "
        "the smallest in the backlog."
    ),
)
register_checks("fused_rope", _fused_rope.correctness_checks)

REGISTRY.register(
    "gqa_decode",
    impl=_gqa_decode.gqa_decode_attention,
    replaces="gqa_attention",
    status=KernelStatus.RETIRED,
    hypothesis="006-gqa-no-expand",
    notes=(
        "Decode attention that indexes the unexpanded KV cache instead of materialising "
        "repeat_interleave's 4x copy: 6.23% of per-token bytes at context 2048, growing "
        "with context. The only slot in batch 001 predicted to win, conditional on "
        "inductor not already folding the expansion."
    ),
)
register_checks("gqa_decode", _gqa_decode.correctness_checks)

REGISTRY.register(
    "flash_decode_splitkv",
    impl=_flash_decode_splitkv.split_kv_decode_attention,
    replaces="gqa_attention",
    status=KernelStatus.RETIRED,
    hypothesis="008-flash-decode-splitkv",
    notes=(
        "006's kernel with the KV scan split across 8 programs and merged by the "
        "online-softmax rescaling identity. At batch 1 with 16 query heads that is 16 "
        "programs becoming 128. Predicted inconclusive at context 2048 — it needs a "
        "long-context workload to express itself — and it is here as the control for 006."
    ),
)

REGISTRY.register(
    "gated_delta_step",
    impl=_gated_delta_step.delta_rule_step,
    replaces="gated_delta_rule",
    status=KernelStatus.RETIRED,
    hypothesis="007-gated-delta-fused-step",
    notes=(
        "One pass over the (128, 128) fp32 recurrent state instead of five: decay, "
        "recall, rank-1 correction and read-out all in registers. 24 of 32 layers. Not "
        "the chunked scan, which is a different and larger claim that cannot express "
        "itself at single-token decode."
    ),
)
register_checks("gated_delta_step", _gated_delta_step.correctness_checks)
register_checks("flash_decode_splitkv", _flash_decode_splitkv.correctness_checks)

# -- batch 003 kernels: weight-only quantisation ------------------------------------
#
# All six install across whole regions of the model rather than swapping one operation,
# so five of them declare `decode_step` — the only replaceable op that means "the step
# itself". `int8_mlp` declares `swiglu_mlp` because it genuinely touches nothing else,
# which is what makes it the diagnostic that separates the MLP's 53.9% of weight bytes
# from the rest.

from . import quantised_linear as _quantised_linear  # noqa: E402

REGISTRY.register(
    "gemv_bf16",
    impl=_quantised_linear.bf16_gemv,
    replaces="decode_step",
    status=KernelStatus.RETIRED,
    hypothesis="009-gemv-bf16-control",
    notes=(
        "The control, not the hypothesis. The same hand-written GEMV as the int8 kernels "
        "reading the reference's own bf16 weights, so it moves exactly the bytes cuBLAS "
        "moves. Its ratio is the divisor that separates 'a hand-written GEMV is "
        "competitive' from 'int8 moves fewer bytes' — without it a win at int8 cannot say "
        "which of the two it is."
    ),
)
register_checks("gemv_bf16", _quantised_linear.bf16_correctness_checks)

REGISTRY.register(
    "int8_dequant_torch",
    impl=_quantised_linear.Int8DequantLinear,
    replaces="decode_step",
    status=KernelStatus.RETIRED,
    hypothesis="010-int8-dequant-torch",
    notes=(
        "Weight-only int8 expressed the only way PyTorch can say it — materialise the "
        "dequantised bf16 weight, then call cuBLAS — and handed to max-autotune. "
        "docs/HYPOTHESES.md asserts inductor cannot fuse this and that the extra write "
        "makes it SLOWER than bf16. That assertion has never been measured; this slot is "
        "the measurement, and it is predicted to lose."
    ),
)
register_checks("int8_dequant_torch", _quantised_linear.int8_dequant_correctness_checks)

REGISTRY.register(
    "int8_mlp",
    impl=_quantised_linear.int8_gemv,
    replaces="swiglu_mlp",
    status=KernelStatus.RETIRED,
    hypothesis="011-int8-mlp",
    notes=(
        "int8 weight-only on the three MLP projections only: 53.9% of the model's weight "
        "bytes, and the cleanest large share in the model. Diagnostic for 012 — the "
        "difference between them is what the attention and linear-attention projections "
        "are worth."
    ),
)
register_checks("int8_mlp", _quantised_linear.int8_mlp_correctness_checks)

REGISTRY.register(
    "int8_all_linear",
    impl=_quantised_linear.int8_gemv,
    replaces="decode_step",
    status=KernelStatus.RETIRED,
    hypothesis="012-int8-all-linear",
    notes=(
        "int8 weight-only on every projection inside a decoder layer: MLP, attention and "
        "linear attention, 84.9% of weight bytes. The tied LM head is deliberately left "
        "in bf16 so that 013 minus 012 is exactly what the head is worth."
    ),
)
register_checks("int8_all_linear", _quantised_linear.int8_correctness_checks)

REGISTRY.register(
    "int8_full",
    impl=_quantised_linear.int8_gemv,
    replaces="decode_step",
    status=KernelStatus.RETIRED,
    hypothesis="013-int8-full",
    notes=(
        "012 plus the tied LM head, which is 15.1% of weight bytes and the single largest "
        "GEMV in the model. 100% of the weight stream at 8 bits: the roofline ceiling is "
        "1.85x. The embedding *lookup* stays bf16 — it reads one row, not the table."
    ),
)
register_checks("int8_full", _quantised_linear.int8_full_correctness_checks)

REGISTRY.register(
    "int4_full",
    impl=_quantised_linear.int4_gemv,
    replaces="decode_step",
    status=KernelStatus.RETIRED,
    hypothesis="014-int4-full",
    notes=(
        "The same sites as 013 at 4 bits with per-group (128) scales, two values packed "
        "per byte and unpacked in registers. Ceiling 3.21x, and the largest accuracy risk "
        "in the batch: in_proj_a and in_proj_b feed an exponential, and 32 output channels "
        "of 4-bit weights is where this breaks if it breaks."
    ),
)
register_checks("int4_full", _quantised_linear.int4_correctness_checks)

# Batch 004. Registered under its own name rather than replacing `gemv_bf16`: 009's
# measurement is the control this one is read against, and `CHECK_BUILDERS` keys by kernel
# name because both replace `decode_step` and an op-keyed table would run one's checks
# against the other's weights and report `pass`.

from . import tiled_gemv as _tiled_gemv  # noqa: E402

REGISTRY.register(
    "tiled_gemv_bf16",
    impl=_tiled_gemv.tiled_gemv_bf16,
    replaces="decode_step",
    hypothesis="015-tiled-gemv-bf16",
    notes=(
        "009's kernel rewritten around a tl.dot accumulator with a K-major weight layout "
        "and split-K, which removes the per-K-iteration cross-lane reduction that took "
        "batch 003 from ~1200 GB/s to 332. Moves exactly cuBLAS's bytes, so its ratio is "
        "the divisor for every quantised slot behind it and 0.2801 is the number it has "
        "to beat. Tying is the honest expectation; its job is to be the divisor."
    ),
)
register_checks("tiled_gemv_bf16", _tiled_gemv.tiled_bf16_correctness_checks)

REGISTRY.register(
    "tiled_fp8_mlp",
    impl=_tiled_gemv.tiled_gemv_fp8,
    replaces="swiglu_mlp",
    hypothesis="018-fp8-mlp",
    notes=(
        "e4m3 on the three MLP projections only: 53.9% of the model's weight bytes and "
        "49.5% of per-token traffic. The low rung of the dose-response ladder -- a smaller "
        "share buying a smaller win is what distinguishes 'the mechanism works' from "
        "'something else moved'."
    ),
)
register_checks("tiled_fp8_mlp", _tiled_gemv.tiled_fp8_mlp_correctness_checks)

REGISTRY.register(
    "tiled_fp8_all_linear",
    impl=_tiled_gemv.tiled_gemv_fp8,
    replaces="decode_step",
    hypothesis="016-fp8-all-linear",
    notes=(
        "e4m3 on every projection inside a decoder layer: 77.9% of per-token bytes. The "
        "direct fp8 counterpart of batch 003's 012, which returned 0.1962 on a kernel that "
        "could not collect the saving. Every finite e4m3 value is exactly a bf16 value, so "
        "the conversion this kernel performs is lossless."
    ),
)
register_checks("tiled_fp8_all_linear", _tiled_gemv.tiled_fp8_correctness_checks)

REGISTRY.register(
    "tiled_fp8_full",
    impl=_tiled_gemv.tiled_gemv_fp8,
    replaces="decode_step",
    hypothesis="017-fp8-full",
    notes=(
        "016 plus the tied LM head, which is 15.1% of weight bytes and the largest single "
        "GEMV in the model: 91.8% of per-token traffic at 8 bits, a 1.85x roofline ceiling. "
        "The embedding *lookup* stays bf16 -- it reads one row, not the table."
    ),
)
register_checks("tiled_fp8_full", _tiled_gemv.tiled_fp8_full_correctness_checks)

REGISTRY.register(
    "tiled_int8_all_linear",
    impl=_tiled_gemv.tiled_gemv_int8,
    replaces="decode_step",
    hypothesis="019-int8-all-linear",
    notes=(
        "The same sites and the same bit width as 016, stored int8 instead of e4m3. Batch "
        "003 measured int8 at 1.438x the time of bf16 in the same kernel because int8->fp32 "
        "is an ALU instruction on the critical path, where sm_120 converts e4m3 inside the "
        "MMA pipeline. Adjacent to 016 so the difference between them is that tax alone, "
        "and it is also the direct re-run of 012: 0.1962 against whatever it now returns is "
        "the value of the kernel rewrite, isolated."
    ),
)
register_checks("tiled_int8_all_linear", _tiled_gemv.tiled_int8_correctness_checks)

REGISTRY.register(
    "tiled_int4_full",
    impl=_tiled_gemv.tiled_gemv_int4,
    replaces="decode_step",
    hypothesis="020-int4-full",
    notes=(
        "The same sites as 017 at 4 bits with per-group (128) scales, two values packed per "
        "byte and unpacked in registers. Ceiling 3.21x and the largest accuracy risk in the "
        "batch: batch 003's int4 measured 0.0919 nats and 38/264 flips. The group scale "
        "varies along K, so it is applied to the weight tile before the dot rather than to "
        "a partial sum -- no cross-lane reduction enters the loop."
    ),
)
register_checks("tiled_int4_full", _tiled_gemv.tiled_int4_correctness_checks)
