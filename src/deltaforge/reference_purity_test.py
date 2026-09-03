"""Mechanically enforce the baseline's central invariant.

`reference.py` is the definition of the baseline, and the project's headline claim is
"hand-written Triton beats what the compiler generates". If the baseline itself called a
hand-written kernel, the claim would compare hand-tuned to hand-tuned and would be false.

That invariant is easy to state and easy to violate by accident — one convenient import,
one `F.scaled_dot_product_attention` for speed, and the number stops meaning what the
README says it means. So it is checked here rather than trusted to reviewers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REFERENCE = Path(__file__).with_name("reference.py")

#: Import roots the baseline may never touch.
FORBIDDEN_IMPORT_ROOTS = {
    "triton",
    "flash_attn",
    "flash_attn_2_cuda",
    "fla",
    "flash_linear_attention",
    "transformers",
    "xformers",
    "apex",
    "liger_kernel",
    "mamba_ssm",
    "causal_conv1d",
    "deltaforge.kernels",
}

#: Call targets that *are* the fused algorithm under test. Using one of these would mean
#: benchmarking hand-written Triton against hand-written CUDA. `scaled_dot_product_attention`
#: dispatches to FlashAttention, which is precisely hypothesis 5's target.
FORBIDDEN_CALLS = {
    "scaled_dot_product_attention",
    "flash_attn_func",
    "flash_attn_varlen_func",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
}

#: Dynamic-import machinery. `_imported_roots` walks only `ast.Import`/`ast.ImportFrom`,
#: so `importlib.import_module("triton")` is an `ast.Call` that every import check above
#: would wave through. These are checked structurally in the same AST call walk, with a
#: source-level backstop below for forms an attribute-call check can still miss.
FORBIDDEN_DYNAMIC_IMPORT_CALLS = {
    "import_module",
    "__import__",
}


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(REFERENCE.read_text(), filename=str(REFERENCE))


def _imported_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module)
            roots.add(node.module.split(".")[0])
    return roots


def test_reference_imports_no_hand_written_kernel_library(tree):
    offending = _imported_roots(tree) & FORBIDDEN_IMPORT_ROOTS
    assert not offending, (
        f"reference.py imports {sorted(offending)}. The baseline must contain no custom "
        "kernels of any kind; see the module docstring for where the line is drawn."
    )


def test_reference_imports_only_torch_and_local_config(tree):
    """Positive form: an unexpected new dependency has to be a deliberate decision."""
    allowed = {
        "torch",
        "math",
        "dataclasses",
        "__future__",
        "typing",
        "collections",
        "collections.abc",
        "torch.nn",
        "torch.nn.functional",
        ".config",
        "config",
    }
    roots = {r for r in _imported_roots(tree) if r not in allowed}
    assert not roots, f"reference.py gained unexpected imports: {sorted(roots)}"


def _called_names(tree: ast.Module) -> set[str]:
    """Every callee name in the module, whether called bare or through an attribute."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name:
            names.add(name)
    return names


def test_reference_never_calls_a_fused_attention_or_scan_primitive(tree):
    found = _called_names(tree) & FORBIDDEN_CALLS
    assert not found, (
        f"reference.py calls {sorted(found)}. These are the fused algorithms the project "
        "exists to hand-write; the baseline must spell them out instead."
    )


def test_reference_never_reaches_a_kernel_through_a_dynamic_import(tree):
    """A statically clean import list proves nothing if the module can import at runtime.

    Enforced structurally on the AST call graph, so it does not depend on how the source
    happens to be spelled or formatted.
    """
    found = _called_names(tree) & FORBIDDEN_DYNAMIC_IMPORT_CALLS
    assert not found, (
        f"reference.py calls {sorted(found)}. The baseline's dependencies must be "
        "statically visible; a runtime import can pull in a hand-written kernel and "
        "defeat every import check in this file."
    )


def test_the_check_would_actually_catch_a_violation(tree):
    """A guard test that never fails is worthless. Prove the detector detects."""
    violating = ast.parse(
        "import triton\n"
        "import torch.nn.functional as F\n"
        "def f(q, k, v):\n"
        "    return F.scaled_dot_product_attention(q, k, v)\n"
    )
    assert _imported_roots(violating) & FORBIDDEN_IMPORT_ROOTS == {"triton"}
    assert _called_names(violating) & FORBIDDEN_CALLS == {"scaled_dot_product_attention"}

    # The dynamic-import path is the one the import walk cannot see: no ast.Import node
    # appears anywhere in this snippet, yet it loads Triton.
    dynamic = ast.parse(
        "import importlib\n"
        "def f():\n"
        "    tl = importlib.import_module('triton.language')\n"
        "    return __import__('triton')\n"
    )
    assert not _imported_roots(dynamic) & FORBIDDEN_IMPORT_ROOTS
    assert _called_names(dynamic) & FORBIDDEN_DYNAMIC_IMPORT_CALLS == {
        "import_module",
        "__import__",
    }

    # And the real module is clean by all three detectors.
    assert not _imported_roots(tree) & FORBIDDEN_IMPORT_ROOTS
    assert not _called_names(tree) & (FORBIDDEN_CALLS | FORBIDDEN_DYNAMIC_IMPORT_CALLS)


def test_no_dynamic_import_machinery_appears_in_the_source_either():
    """Source-level backstop for the check above, not a search for the word "triton".

    The AST call check is the primary enforcement. This keeps a second, dumber reading of
    the same invariant so that a spelling the attribute-call walk mishandles — an aliased
    `importlib`, a getattr indirection, a call built up in a string — still trips."""
    source = REFERENCE.read_text()
    code_lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("*"):
            continue
        code_lines.append(line)
    body = "\n".join(code_lines)
    # The module docstring legitimately explains why Triton is excluded, so only
    # executable-looking occurrences matter.
    assert "import_module" not in body
    assert "importlib" not in body
    assert "__import__" not in body
