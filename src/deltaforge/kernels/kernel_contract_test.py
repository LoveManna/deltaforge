"""Invariants every Triton kernel in this package must hold, checked without a GPU.

Rental 40 paid for this file. `_reduce_partials_kernel` took `SCALE` and `HAS_SCALE` in
its signature and never read either, so `tiled_gemv_int8` and `tiled_gemv_fp8` returned
unscaled integer dot products — layer-1 relative error **4511**, and 9473 nats at layer 2.
The epilogue that applied the scale had always existed in `quantised_linear` (line 355);
the batch-004 rewrite moved it into this kernel and dropped the multiply.

**It survived a whole rental because nothing executed it.** Batch 004 built five slots on
that kernel and its preconditions declined all five, so the code shipped, passed CI and was
never run. A declined slot saves money and leaves an unexecuted code path behind it, and
the next batch that picks one up starts from "never run" rather than "known good".

The checks here are source-level, because that is the only kind available without a card.
A Triton body is a string compiled on a GPU, so neither the type checker, the linter nor
the CPU suite can see into it — but the AST can.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

KERNELS = pathlib.Path(__file__).parent

#: ``(kernel, parameter)`` pairs that are declared and legitimately never read.
#:
#: Every one of these is a **grid bound**: the launcher uses it to size `program_id`
#: dimensions and the body indexes with `pid_m` instead. They are dead weight rather than
#: dropped computation, and they are listed one by one rather than exempted by name so a
#: seventh has to be added deliberately. `quantised_linear`'s kernels are kept exactly as
#: batch 003 measured them — their numbers are the control every later GEMV is read
#: against — so the fix for those is not to edit them.
GRID_ONLY_PARAMETERS = frozenset(
    {
        ("_bf16_gemv_kernel", "M"),
        ("_int8_gemv_kernel", "M"),
        ("_int4_gemv_kernel", "M"),
        ("_int4_gemv_kernel", "K"),
        ("_tiled_gemv_bf16_kernel", "M"),
        ("_tiled_gemv_scaled_kernel", "M"),
        ("_tiled_gemv_int4_kernel", "M"),
    }
)


def _jit_kernels() -> list[tuple[str, str, ast.FunctionDef]]:
    """Every ``@triton.jit`` function in this package, as (module, name, node)."""
    found = []
    for path in sorted(KERNELS.glob("*.py")):
        if path.name.endswith("_test.py"):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if any(ast.unparse(d).endswith("triton.jit") for d in node.decorator_list):
                found.append((path.name, node.name, node))
    return found


def test_the_package_actually_has_jit_kernels_to_check():
    """A collector that silently finds nothing passes every test below."""
    assert len(_jit_kernels()) >= 5


@pytest.mark.parametrize("module,name,node", _jit_kernels(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_kernel_parameter_is_read_or_declared_grid_only(module, name, node):
    """A declared-but-unread parameter is a dropped computation until it is named as not.

    `_reduce_partials_kernel` declared `SCALE` and `HAS_SCALE`, read neither, and returned
    int8 dot products scaled by nothing. Every caller passed the scale correctly; the kernel
    simply did not use it, and the callers had no way to tell.
    """
    declared = {arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
    read = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    unread = sorted(param for param in declared - read if (name, param) not in GRID_ONLY_PARAMETERS)

    assert not unread, (
        f"{module}:{name} declares {unread} and never reads "
        f"{'it' if len(unread) == 1 else 'them'}. Either the kernel is dropping a "
        "computation its callers think it performs — rental 40: a per-channel scale, worth "
        "a layer-1 relative error of 4511 — or the parameter is a grid bound and belongs "
        "in GRID_ONLY_PARAMETERS with the others."
    )


def test_the_dtype_polymorphic_load_uses_a_literal_both_dtypes_accept():
    """`other=0` is an int32 literal, and it does not cast to e4m3.

    `_tiled_gemv_scaled_kernel` serves **both** int8 and fp8 from one body, so its masked
    load cannot use a literal only one of them accepts. On rental 40 the int8 slot compiled
    and ran while `024-fp8-head` died at compile time — `cannot cast int32[64, 64] to
    fp8e4nv` — from that one character.

    Scoped to this kernel on purpose: `_tiled_gemv_int4_kernel` loads packed nibbles out of
    an int8 tensor and is right to use an integer `other`. The rule is not "always a float",
    it is "a kernel reused across storage dtypes cannot assume either".
    """
    source = (KERNELS / "tiled_gemv.py").read_text()
    node = next(n for n in ast.walk(ast.parse(source)) if _is_named(n, "_tiled_gemv_scaled_kernel"))

    others = [
        keyword.value
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "tl.load"
        for keyword in call.keywords
        if keyword.arg == "other"
    ]

    assert others, "the kernel does a masked load; if it stopped, this test should have gone too"
    for value in others:
        assert isinstance(value, ast.Constant) and isinstance(value.value, float), (
            f"other={ast.unparse(value)} is not castable to every dtype this kernel loads"
        )


def _is_named(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.FunctionDef) and node.name == name
