"""Shared Triton plumbing: the optional import, launch sizing, and the no-fallback rule.

Every kernel module in this package repeats the same three things, and repeating them by
hand is how one of them ends up subtly different from the others. In particular the
**no silent fallback** rule below is load-bearing and must be spelled the same way
everywhere: a candidate column that quietly ran PyTorch would benchmark the baseline
while the harness recorded it as the kernel under test.

The CPU development environment and CI have no Triton, so importing any kernel module —
and therefore the registry — must work without it. `HAS_TRITON` guards the `@triton.jit`
definitions; the wrappers around them raise on a non-CUDA tensor rather than falling back.
"""

from __future__ import annotations

from torch import Tensor

__all__ = [
    "HAS_TRITON",
    "launch_config",
    "next_power_of_two",
    "require_cuda",
    "tl",
    "triton",
]

try:  # pragma: no cover - the CPU environment takes the except branch
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised by CI, which has no Triton
    triton = None
    tl = None
    HAS_TRITON = False


def next_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError(f"no power of two covers {n}")
    return 1 << (n - 1).bit_length()


def launch_config(n: int, *, min_block: int = 128, max_block: int = 65536) -> tuple[int, int]:
    """``(BLOCK, num_warps)`` for a one-row-per-program pass over ``n`` elements.

    One program per row keeps a reduction in registers with no cross-block communication.
    A 2560-element bf16 row is 5 KB, comfortably register-resident.
    """
    block = min(max(next_power_of_two(n), min_block), max_block)
    if next_power_of_two(n) > max_block:
        raise ValueError(f"a row of {n} elements is too wide for a single-program pass")
    return block, min(32, max(4, block // 256))


def require_cuda(tensor: Tensor, op: str) -> None:
    """Refuse to run on CPU rather than falling back to PyTorch.

    There is no fallback anywhere in this package, deliberately. A kernel that quietly ran
    the reference implementation on an unsupported input would produce a plausible number
    for the wrong thing, and the harness would record it as the candidate. Failing loudly
    turns that into an errored slot, which is a result; falling back turns it into a lie.
    """
    if not tensor.is_cuda:
        raise RuntimeError(
            f"deltaforge::{op} requires a CUDA tensor and has no PyTorch fallback, on "
            "purpose: falling back here would run the reference while the harness "
            "recorded it as the candidate."
        )
