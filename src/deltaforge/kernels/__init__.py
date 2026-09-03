"""The kernel registry: ``name -> (impl, replaces, status)``.

**This package ships with no kernels.** The bootstrap session builds the harness that
measures kernels; writing one in the same session as the harness that measures it
produces a broken harness and a meaningless number. The registry contract, its
invariants and its tests are here so the first hypothesis session has somewhere to
land.

Each kernel module exports a callable with the *exact signature* of the reference
operation it replaces, and registers itself here. The registry's single hard invariant:
**at most one champion per replaceable reference operation.** Two champions for the same
operation would make "what does `model.py` assemble?" ambiguous, and an ambiguous
champion makes the leaderboard a lie.
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


#: Operation name -> a builder returning that kernel's layer-1 `KernelCheck`s.
#:
#: Mirrors `model.INSTALLERS`: a kernel declares how it is *installed* there and how it is
#: *checked* here, both next to the kernel itself. A champion with no checks is allowed —
#: the end-to-end gate still covers it — but it is reported as unchecked rather than as
#: passing.
CHECK_BUILDERS: dict[str, Callable[..., tuple]] = {}


def register_checks(op: str, builder: Callable[..., tuple]) -> None:
    if op in CHECK_BUILDERS:
        raise ValueError(f"checks for {op!r} are already registered")
    CHECK_BUILDERS[op] = builder


def build_kernel_checks(model, *, registry: KernelRegistry | None = None, **kwargs) -> tuple:
    """Run every champion's layer-1 checks against ``model``'s reference operations."""
    registry = REGISTRY if registry is None else registry
    checks: list = []
    for op in registry.champions():
        builder = CHECK_BUILDERS.get(op)
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
    status=KernelStatus.CHAMPION,
    hypothesis="001-fused-rmsnorm-residual",
    notes=(
        "Fuses the residual add with the RMSNorm that follows it, and replaces the "
        "layer's other hidden-size norm with a single-pass Triton RMSNorm. Champion of "
        "an operation with no incumbent: being champion is what puts it in the candidate "
        "column, which is how it gets measured at all. Retired to the graveyard if the "
        "measurement does not clear the noise band."
    ),
)
register_checks("rms_norm_residual", _fused_rmsnorm_residual.correctness_checks)
