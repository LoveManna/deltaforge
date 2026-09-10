"""A wall-clock cap for one unit of work.

`SlotBudget.can_start` decides whether a slot may *begin*. Nothing bounded what it could
then do: rental 22 began slot 0, spent ~40 minutes inside a cold `max-autotune` compile,
and the session gate ended the run with nothing measured.

The work to interrupt is mostly inside inductor — C extensions, a subprocess pool, and a
Python-level autotuning loop. A cooperative flag would never be read, so this uses
`_thread.interrupt_main()`, which raises `KeyboardInterrupt` in the main thread at the next
bytecode boundary. Autotuning reaches that boundary constantly.

**Nothing here imports torch.** It is a timer, and it is tested on a CPU.
"""

from __future__ import annotations

import _thread
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = ["SlotTimeout", "slot_deadline"]


class SlotTimeout(Exception):
    """The capped work did not finish in time.

    Deliberately an `Exception` and not a `BaseException`: `BatchRunner.run_slot` isolates
    a slot by catching `Exception`, and a cap that escaped that handler would take the
    rental down — which is the failure it exists to prevent.
    """


@dataclass
class _Armed:
    """Whether the timer actually fired, so an unrelated Ctrl-C is not relabelled."""

    fired: bool = False


@contextmanager
def slot_deadline(seconds: float) -> Iterator[_Armed]:
    """Interrupt the body after ``seconds``, raising `SlotTimeout`.

    A non-positive cap does not arm: a slot with no time left is refused by
    `SlotBudget.can_start`, and a timer that fired instantly would turn a clean stop into
    an error record.
    """
    armed = _Armed()
    if seconds <= 0:
        yield armed
        return

    def fire() -> None:
        armed.fired = True
        _thread.interrupt_main()

    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    try:
        yield armed
    except KeyboardInterrupt:
        if armed.fired:
            raise SlotTimeout(f"exceeded its {seconds:.1f}s cap") from None
        raise
    finally:
        timer.cancel()
