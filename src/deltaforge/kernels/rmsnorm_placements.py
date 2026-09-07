"""Hypotheses 002 and 003 — the same RMSNorm kernel, in two different places.

Both reuse `fused_rmsnorm_residual.rms_norm` verbatim. Neither introduces new numerics.
What they vary is **where** the kernel is installed, and each isolates one variable that
hypothesis 001 confounded.

**002 — hidden-size norms only, no fusion.** 001 installs a fused add+norm *and* a
standalone norm at the same time, so a result from it cannot say whether any effect came
from the Triton norm or from the fusion of the residual add. 002 installs only the
standalone norm on the same three sites, changing nothing else. **001 minus 002 is the
fusion**, and that subtraction is the only way either number means anything.

**003 — the head-dim norms.** `q_norm` and `k_norm` are also RMSNorm but run over
``head_dim`` = 256 rather than ``hidden_size`` = 2560, on ``(batch, seq, heads, 256)``
rather than ``(batch, seq, 2560)``. 001's docstring explicitly declined them to keep its
measurement attributable to one thing. That leaves an open question worth three minutes:
a 256-element row is a tenth of the reduction width and ten times the row count, which is
a different regime for both inductor and the hand-written kernel. Only the 8 full-attention
layers of 32 have these norms at all.

Both are predicted `inconclusive` on the arithmetic — 0.018% and ~0.001% of per-token
bytes. They are here because at three minutes a slot, measuring is cheaper than arguing.
"""

from __future__ import annotations

from ..reference import RMSNorm
from .fused_rmsnorm_residual import rms_norm

__all__ = [
    "install_hidden_norms",
    "install_qk_norms",
    "qk_correctness_checks",
    "standalone_correctness_checks",
]


_PATCHED: dict[str, type] = {}


def _triton_rmsnorm_class() -> type:
    """The patched `RMSNorm` subclass, built once.

    Swapping an instance's `__class__` rather than assigning a bound method: Dynamo then
    sees an ordinary `nn.Module` with an ordinary `forward`, which is what keeps the
    compiled candidate column free of graph breaks.
    """
    if "norm" not in _PATCHED:

        class TritonRMSNorm(RMSNorm):
            def forward(self, x):
                return rms_norm(x, self.weight, self.eps)

        _PATCHED["norm"] = TritonRMSNorm
    return _PATCHED["norm"]


def install_hidden_norms(model, entry=None) -> None:
    """Hypothesis 002: every ``hidden_size``-wide RMSNorm, and nothing else.

    The three sites per layer that 001 also touches — `input_layernorm`,
    `post_attention_layernorm` and the model's final norm — but each as a standalone norm
    with the residual add left to the reference.
    """
    if getattr(model, "_deltaforge_hidden_norms", False):
        return
    hidden = model.config.hidden_size
    patched = _triton_rmsnorm_class()
    for layer in model.layers:
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            # Guarded by width rather than by attribute name: a norm that is not
            # hidden-size wide is a different shape regime and belongs to 003, and
            # silently including it would make neither hypothesis mean what it says.
            if norm.weight.shape[0] == hidden:
                norm.__class__ = patched
    if model.norm.weight.shape[0] == hidden:
        model.norm.__class__ = patched
    model._deltaforge_hidden_norms = True


def install_qk_norms(model, entry=None) -> None:
    """Hypothesis 003: only ``q_norm`` and ``k_norm``, in the full-attention layers."""
    if getattr(model, "_deltaforge_qk_norms", False):
        return
    patched = _triton_rmsnorm_class()
    installed = 0
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:  # a linear-attention layer has no q_norm/k_norm
            continue
        attn.q_norm.__class__ = patched
        attn.k_norm.__class__ = patched
        installed += 1
    if installed == 0:
        raise RuntimeError(
            "no full-attention layers found, so this hypothesis installed nothing. A "
            "candidate identical to the reference would be recorded as a null result "
            "rather than as the bug it is."
        )
    model._deltaforge_qk_norms = True


def _checks_for(model, sites, *, device, dtype, seed, label_prefix):
    """`allclose` against the *actual reference module*, not a re-derived formula.

    A re-derivation could agree with the kernel and both be wrong about what
    `reference.py` does, which is the one thing this gate exists to rule out.
    """
    import torch

    from ..harness.correctness import check_kernel

    generator = torch.Generator(device=device).manual_seed(seed)
    checks = []
    for norm, shapes, note in sites:
        width = int(norm.weight.shape[0])
        target_dtype = dtype if dtype is not None else norm.weight.dtype
        for shape in shapes:
            x = torch.randn((*shape, width), device=device, dtype=target_dtype, generator=generator)
            checks.append(
                check_kernel(
                    f"{label_prefix}.rms_norm",
                    lambda t, n=norm: n(t),
                    lambda t, n=norm: rms_norm(t, n.weight, n.eps),
                    args=(x,),
                    replaces="rms_norm",
                    note=f"{note}, shape {tuple(x.shape)}",
                )
            )
    return tuple(checks)


def standalone_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    layer = model.layers[0]
    sites = [
        (layer.input_layernorm, [(1, 1), (32, 1), (1, 2048)], "hidden-size norm"),
        (model.norm, [(1, 1), (1, 2048)], "final norm"),
    ]
    return _checks_for(model, sites, device=device, dtype=dtype, seed=seed, label_prefix="rmsnorm_hidden")


def qk_correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    attn = None
    for layer in model.layers:
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            break
    if attn is None:  # pragma: no cover - every real config has full-attention layers
        return ()
    heads = attn.config.num_attention_heads
    kv_heads = attn.config.num_key_value_heads
    sites = [
        # The shapes q_norm/k_norm actually see: (batch, seq, heads, head_dim).
        (attn.q_norm, [(1, 1, heads), (32, 1, heads), (1, 2048, heads)], "q_norm, head-dim wide"),
        (attn.k_norm, [(1, 1, kv_heads), (32, 1, kv_heads), (1, 2048, kv_heads)], "k_norm, head-dim wide"),
    ]
    return _checks_for(model, sites, device=device, dtype=dtype, seed=seed, label_prefix="rmsnorm_qk")
