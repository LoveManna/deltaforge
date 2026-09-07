"""Assembles the reference model, optionally with the registry's champion kernels.

`reference.py` is the baseline and never changes to accommodate a kernel. This module is
where a champion gets installed on top of it, which keeps the baseline's "no custom
kernels, ever" invariant mechanically true rather than merely intended.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import torch

from .config import ModelConfig, model_config
from .kernels import REGISTRY, KernelEntry, KernelRegistry
from .reference import DecodeCache, ReferenceModel

__all__ = [
    "INSTALLERS",
    "InstallerMissing",
    "apply_champions",
    "build_model",
    "greedy_decode",
    "register_installer",
]


class InstallerMissing(RuntimeError):
    """A champion kernel exists for an operation, but nothing knows how to install it."""


#: **Kernel name** -> a function that splices that kernel into a built model.
#:
#: Keyed by kernel rather than by the operation it replaces. Batch mode routinely runs
#: several kernels against the same reference operation — two different attacks on
#: `gqa_attention`, say — and an operation-keyed table can only name one of them. Keying
#: by kernel gives "which code installs this candidate?" exactly one answer.
#:
#: An installer is added in the same commit as the kernel it installs, and stays
#: registered when that kernel is retired so a retired entry can still be promoted for a
#: one-off comparison. `apply_champions` fails loudly rather than silently running the
#: reference if a champion has no installer — silently benchmarking the baseline as if it
#: were the candidate is the worst failure this harness could have.
INSTALLERS: dict[str, Callable[[ReferenceModel, KernelEntry], None]] = {}


def register_installer(kernel_name: str, installer: Callable[[ReferenceModel, KernelEntry], None]) -> None:
    if kernel_name in INSTALLERS:
        raise ValueError(f"installer for kernel {kernel_name!r} is already registered")
    INSTALLERS[kernel_name] = installer


def apply_champions(
    model: ReferenceModel,
    registry: KernelRegistry = REGISTRY,
    installers: Mapping[str, Callable[[ReferenceModel, KernelEntry], None]] | None = None,
) -> tuple[str, ...]:
    """Install every champion kernel into ``model``. Returns the names installed.

    With no champions registered this is a no-op and returns ``()``, which makes the
    ``candidate`` column bit-identical to ``eager`` — the *identity champion* a first GPU
    session uses to calibrate the harness. That is the repo's current shipped state.
    """
    registry.check_invariants()
    table = INSTALLERS if installers is None else installers
    applied: list[str] = []
    for op, entry in registry.champions().items():
        installer = table.get(entry.name)
        if installer is None:
            raise InstallerMissing(
                f"{entry.name!r} is champion of {op!r} but no installer is registered for "
                f"it. Add one to deltaforge.model.INSTALLERS alongside the kernel; "
                "refusing to run, because falling back to the reference here would "
                "benchmark the baseline while labelling it the candidate."
            )
        installer(model, entry)
        applied.append(entry.name)
    return tuple(applied)


def build_model(
    config: ModelConfig | None = None,
    weights_path: str | Path | None = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    registry: KernelRegistry | None = None,
    verbose: bool = True,
) -> ReferenceModel:
    """Build a model, load weights if given, and install champions if a registry is given.

    ``registry=None`` means "pure reference" — that is what the ``eager`` and ``compiled``
    benchmark columns use. Pass ``REGISTRY`` to build the ``candidate`` column.
    """
    config = config or model_config()
    model = ReferenceModel(config).to(device=device, dtype=dtype).eval()
    if weights_path is not None:
        from .weights import load_weights  # noqa: PLC0415 - keeps safetensors optional

        load_weights(model, weights_path, dtype=dtype, verbose=verbose)
    if registry is not None:
        apply_champions(model, registry)
    return model


@torch.no_grad()
def greedy_decode(
    model: ReferenceModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    cache: DecodeCache | None = None,
) -> torch.Tensor:
    """Greedy-decode ``max_new_tokens`` and return only the generated ids.

    Used by the layer-2 correctness gate, where the candidate's token sequence must
    match eager's exactly, and by the benchmark's decode workload.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    batch, prompt_len = input_ids.shape
    if cache is None:
        cache = model.new_cache(batch, prompt_len + max_new_tokens)

    logits, _ = model(input_ids, cache, num_logits_to_keep=1)
    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        logits, _ = model(next_token, cache, num_logits_to_keep=1)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
    return torch.cat(generated, dim=1)


# -- installers --------------------------------------------------------------------
#
# One entry per replaceable operation, registered next to the kernel that needs it.


def _install_fused_rmsnorm_residual(model: ReferenceModel, entry: KernelEntry) -> None:
    from .kernels.fused_rmsnorm_residual import install  # noqa: PLC0415 - keeps Triton optional

    install(model, entry)


register_installer("fused_rmsnorm_residual", _install_fused_rmsnorm_residual)


def _install_rmsnorm_hidden(model: ReferenceModel, entry: KernelEntry) -> None:
    from .kernels.rmsnorm_placements import install_hidden_norms  # noqa: PLC0415

    install_hidden_norms(model, entry)


def _install_rmsnorm_qk(model: ReferenceModel, entry: KernelEntry) -> None:
    from .kernels.rmsnorm_placements import install_qk_norms  # noqa: PLC0415

    install_qk_norms(model, entry)


def _install_fused_swiglu(model: ReferenceModel, entry: KernelEntry) -> None:
    from .kernels.fused_swiglu import install  # noqa: PLC0415

    install(model, entry)


def _install_fused_rope(model: ReferenceModel, entry: KernelEntry) -> None:
    from .kernels.fused_rope import install  # noqa: PLC0415

    install(model, entry)


def _install_gqa_decode(model: ReferenceModel, entry: KernelEntry) -> None:
    from .kernels.gqa_decode import install  # noqa: PLC0415

    install(model, entry)


def _install_flash_decode_splitkv(model: ReferenceModel, entry: KernelEntry) -> None:
    from .kernels.flash_decode_splitkv import install  # noqa: PLC0415

    install(model, entry)


def _install_gated_delta_step(model: ReferenceModel, entry: KernelEntry) -> None:
    from .kernels.gated_delta_step import install  # noqa: PLC0415

    install(model, entry)


register_installer("rmsnorm_hidden", _install_rmsnorm_hidden)
register_installer("rmsnorm_qk", _install_rmsnorm_qk)
register_installer("fused_swiglu", _install_fused_swiglu)
register_installer("fused_rope", _install_fused_rope)
register_installer("gqa_decode", _install_gqa_decode)
register_installer("flash_decode_splitkv", _install_flash_decode_splitkv)
register_installer("gated_delta_step", _install_gated_delta_step)
