"""CPU tests for the command-line entry points.

`_load_models` itself needs CUDA and a 9 GB checkpoint, so what is tested here is the
*mechanism* it relies on — that two module trees can share one set of parameter tensors,
that installing a kernel into one does not disturb the other, and that the guard which
checks all this actually fires. Those are the parts that would fail silently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from .cli import DEFAULT_WORKLOADS, _assert_parameters_are_shared, _selected_columns, build_parser
from .config import tiny_config
from .kernels import KernelRegistry, KernelStatus
from .model import apply_champions
from .reference import ReferenceModel


@pytest.fixture
def pair():
    """A reference and a candidate built the way `_load_models` builds them."""
    config = tiny_config()
    torch.manual_seed(0)
    reference = ReferenceModel(config).eval()
    candidate = ReferenceModel(config).eval()
    candidate.load_state_dict(reference.state_dict(), assign=True)
    return reference, candidate


# -- shared parameters ------------------------------------------------------------------


def test_assign_true_shares_storage_rather_than_copying(pair):
    """The whole point: one set of weights, not two.

    Loading twice cost a second 9.3 GB read from disk and held 16.8 GB on a 32 GB card.
    `assign=True` is what makes `load_state_dict` rebind tensors instead of copying into
    the destination's existing storage.
    """
    reference, candidate = pair
    ref = dict(reference.named_parameters())

    for name, param in candidate.named_parameters():
        assert param.data_ptr() == ref[name].data_ptr(), f"{name} was copied, not shared"


def test_a_copying_load_is_caught_by_the_guard(pair):
    """`assign=False` is the default and silently doubles memory, so the guard must fire."""
    reference, _ = pair
    copied = ReferenceModel(tiny_config()).eval()
    copied.load_state_dict(reference.state_dict())  # no assign=True

    with pytest.raises(SystemExit, match="does not share storage"):
        _assert_parameters_are_shared(reference, copied)


def test_the_guard_passes_on_a_correctly_shared_pair(pair):
    reference, candidate = pair

    _assert_parameters_are_shared(reference, candidate)


def test_installing_a_kernel_into_the_candidate_leaves_the_reference_alone(pair):
    """Separate module trees are the reason the candidate can be modified at all.

    Installation swaps `__class__` on the candidate's modules. If the trees were shared,
    the reference would be modified too and the benchmark would compare the candidate
    against itself — the failure this harness exists to prevent.
    """
    reference, candidate = pair
    reference_classes = [type(layer) for layer in reference.layers]

    registry = KernelRegistry()
    registry.register(
        "spy",
        impl=lambda *a, **k: None,
        replaces="rms_norm_residual",
        status=KernelStatus.CHAMPION,
    )
    installed = []

    def installer(model, entry):
        installed.append(entry.name)
        for layer in model.layers:
            layer.__class__ = type("Fused", (type(layer),), {})

    apply_champions(candidate, registry, installers={"spy": installer})

    assert installed == ["spy"]
    assert [type(layer) for layer in reference.layers] == reference_classes
    assert [type(layer) for layer in candidate.layers] != reference_classes
    # And the weights are still the same tensors after installation.
    _assert_parameters_are_shared(reference, candidate)


def test_sharing_survives_a_forward_pass(pair):
    """Nothing writes to a parameter, which is what makes sharing safe in the first place."""
    reference, candidate = pair
    ids = torch.randint(0, tiny_config().vocab_size, (1, 4))

    with torch.no_grad():
        reference(ids)
        candidate(ids)

    _assert_parameters_are_shared(reference, candidate)


# -- column selection -------------------------------------------------------------------


def _bench_args(**overrides) -> argparse.Namespace:
    args = build_parser().parse_args(
        ["bench", "--weights", "/nonexistent", *[str(x) for x in overrides.pop("argv", [])]]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_the_default_columns_omit_the_nocudagraphs_diagnostic():
    """Three max-autotune compilations per run is the single largest fixed cost.

    `compiled_nocudagraphs` existed to show a win was not merely CUDA-graph launch-overhead
    removal. That confound is gone by construction now that the scoring pair —
    `compiled` and `candidate_compiled` — both have CUDA graphs. It stays available, but
    not on every run.
    """
    labels = _selected_columns(_bench_args())

    assert "compiled_nocudagraphs" not in labels
    assert {"compiled", "candidate_compiled"} <= set(labels)


def test_the_scoring_pair_can_never_be_dropped():
    """`compiled` vs `candidate_compiled` is the claim. Without both there is no result."""
    for dropped in ("compiled", "candidate_compiled"):
        keep = [c for c in ("eager", "compiled", "candidate", "candidate_compiled") if c != dropped]
        with pytest.raises(SystemExit, match="scoring"):
            _selected_columns(_bench_args(columns=",".join(keep)))


def test_all_columns_can_be_requested_for_a_calibration_run():
    labels = _selected_columns(_bench_args(columns="all"))

    assert "compiled_nocudagraphs" in labels
    assert len(labels) == 5


def test_an_unknown_column_is_refused_rather_than_ignored():
    with pytest.raises(SystemExit, match="unknown benchmark column"):
        _selected_columns(_bench_args(columns="compiled,candidate_compiled,teleport"))


# -- fetching the checkpoint ------------------------------------------------------------


def _fetch_args(tmp_path, **overrides) -> argparse.Namespace:
    args = build_parser().parse_args(["fetch-weights", "--dest", str(tmp_path), "--model", "Qwen/Qwen3.5-4B"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_the_fast_download_is_used_when_available(tmp_path, monkeypatch):
    from . import cli

    calls = []

    def fake_snapshot(repo, dest, *, use_xet):
        calls.append((repo, use_xet))
        return True

    monkeypatch.setattr(cli, "_snapshot_download", fake_snapshot)
    monkeypatch.setattr(cli, "_download", lambda *a: pytest.fail("fell back unnecessarily"))

    assert cli.cmd_fetch_weights(_fetch_args(tmp_path)) == 0
    # Xet off unless asked for: the only two silent stalls this project has recorded
    # (rentals 47 and 55, both at 4/6 files) were CAS retry storms, and CAS is Xet.
    assert calls == [("Qwen/Qwen3.5-4B", False)]
    assert cli.cmd_fetch_weights(_fetch_args(tmp_path, xet=True)) == 0
    assert calls[-1] == ("Qwen/Qwen3.5-4B", True)


def test_a_missing_huggingface_hub_falls_back_rather_than_failing(tmp_path, monkeypatch):
    """The fallback is slow, not broken. A box without the library must still work."""
    from . import cli

    monkeypatch.setattr(cli, "_snapshot_download", lambda repo, dest, **kw: False)
    fetched = []

    def fake_download(url, path):
        fetched.append(path.name)
        if path.name == "model.safetensors.index.json":
            path.write_text('{"weight_map": {"a": "model-00001.safetensors"}}')
        else:
            path.write_text("x")

    monkeypatch.setattr(cli, "_download", fake_download)

    assert cli.cmd_fetch_weights(_fetch_args(tmp_path)) == 0
    assert "model-00001.safetensors" in fetched


def test_a_failing_fast_download_falls_back_instead_of_losing_the_rental(tmp_path, monkeypatch):
    """An exception mid-download must not end the run: the box is billed by the minute and
    the slow path still works."""
    from . import cli

    def boom(repo, dest):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(cli, "_snapshot_download", boom)
    monkeypatch.setattr(
        cli,
        "_download",
        lambda url, path: path.write_text(
            '{"weight_map": {"a": "s.safetensors"}}' if path.name == "model.safetensors.index.json" else "x"
        ),
    )

    assert cli.cmd_fetch_weights(_fetch_args(tmp_path)) == 0


def test_no_hf_transfer_skips_the_fast_path_entirely(tmp_path, monkeypatch):
    from . import cli

    monkeypatch.setattr(cli, "_snapshot_download", lambda repo, dest: pytest.fail("should not be called"))
    monkeypatch.setattr(
        cli,
        "_download",
        lambda url, path: path.write_text(
            '{"weight_map": {"a": "s.safetensors"}}' if path.name == "model.safetensors.index.json" else "x"
        ),
    )

    assert cli.cmd_fetch_weights(_fetch_args(tmp_path, no_hf_transfer=True)) == 0


def test_the_default_columns_are_the_two_that_score():
    """Rental 21 OOMed at 30.71 GiB of 31.36 during warmup with four columns.

    Rental 22 measured construction at 0.11 GiB on top of the weights, so the memory went
    to columns being resident through warmup — and two of the four score nothing.
    `--columns all` still asks for the diagnostics when a result is confusing enough to be
    worth the memory and the compile.
    """
    from .cli import DEFAULT_COLUMNS, SCORING_COLUMNS

    assert DEFAULT_COLUMNS == SCORING_COLUMNS


# -- what a rental is allowed to tell the next one's pre-flight gate -----------------------


def _slot(slug: str, duration_s: float, outcome: str = "inconclusive", error: str | None = None):
    from .batch import Hypothesis
    from .batch_run import SlotResult

    return SlotResult(
        hypothesis=Hypothesis(
            slug=slug,
            kernels=("k",),
            category="A",
            byte_share=0.01,
            mechanism="does a thing",
            prediction="inconclusive",
            rationale="a rationale long enough to be a claim rather than a label, stated up front",
            # A kernel cannot be bit-identical, so `exact` is not available to it.
            correctness="approximate",
            top1_threshold=0.9,
            kl_threshold=0.01,
        ),
        outcome=outcome,
        duration_s=duration_s,
        error=error,
    )


def test_a_slot_that_hit_its_cap_is_not_reported_as_what_a_slot_costs(tmp_path):
    """A cut-off slot is a lower bound, and the gate it feeds decides whether to rent.

    Rental 32's slot 0 was killed at its 6980.9s cap without finishing, and `max()` over
    slot durations wrote that cap to `phases.env` as the measured cost of a slot. The next
    pre-flight then needs 310 minutes against a 180-minute gate and refuses — so one
    timeout silently ends every future rental on that card, and the number doing it was
    never a measurement of anything.
    """
    from .cli import _write_phases_env

    path = tmp_path / "phases.env"
    results = [
        _slot("000-identity", 6980.9, outcome="error", error="SlotTimeout: exceeded its 6980.9s cap"),
        _slot("001-real", 240.0),
    ]

    _write_phases_env(path, {"000-identity.compile_compiled": 120.0}, results)

    assert "DF_PHASE_SLOT_S=240.0" in path.read_text()


def test_a_slot_that_failed_on_its_own_terms_still_counts(tmp_path):
    """Only a cap is excluded. A kernel that raised still ran, and its time was real —
    dropping every error would make the estimate optimistic, which costs a rental."""
    from .cli import _write_phases_env

    path = tmp_path / "phases.env"
    results = [
        _slot("000-identity", 200.0),
        _slot("001-bad-kernel", 900.0, outcome="error", error="ImportError: no such name"),
    ]

    _write_phases_env(path, {"000-identity.compile_compiled": 120.0}, results)

    assert "DF_PHASE_SLOT_S=900.0" in path.read_text()


def test_a_rental_whose_every_slot_was_cut_off_records_nothing(tmp_path):
    """With no slot that finished there is no measurement, and the cold estimates in
    `batch.COLD_PHASE_ESTIMATES` are a better input to the gate than a cap."""
    from .cli import _write_phases_env

    path = tmp_path / "phases.env"
    results = [_slot("000-identity", 6980.9, outcome="error", error="SlotTimeout: exceeded its 6980.9s cap")]

    _write_phases_env(path, {}, results)

    assert not path.exists()


# -- workloads ------------------------------------------------------------------


def test_a_text_workload_exists_with_the_headline_shape():
    """Acceptance on random token ids is not acceptance on text: the model's continuation
    of noise is its own distribution, and an n-gram drafter can look far better or far
    worse there than it ever will in use. Every hypothesis before this one was indifferent
    to the prompt."""
    text = DEFAULT_WORKLOADS["headline_text"]
    headline = DEFAULT_WORKLOADS["headline"]

    assert text["batch_size"] == headline["batch_size"]
    assert text["context_length"] == headline["context_length"]
    assert text["decode_tokens"] == headline["decode_tokens"]
    assert text["prompt"] == "text"
    assert headline.get("prompt", "random") == "random"


def test_workload_prompt_refuses_an_unrecognised_prompt_kind():
    """I1's required test: the silent-substitution failure made structurally impossible.

    Before `_workload_prompt` existed, `cmd_batch` built its prompt with `torch.randint`
    unconditionally and never looked at `workload["prompt"]` at all, so
    `--batch 010-speculative-verify --workload headline_text` silently ran on random token
    ids -- exactly the failure this asserts can no longer happen: an unrecognised prompt
    kind must refuse loudly rather than fall back to noise. Chosen over a full
    `cmd_batch`-builds-text-prompt test because `cmd_batch` needs CUDA and a checkpoint to
    run at all; this is what is honestly testable on a CPU.
    """
    from .cli import _workload_prompt

    with pytest.raises(SystemExit, match="unknown workload prompt"):
        _workload_prompt(
            weights=None,
            workload={"batch_size": 1, "context_length": 8, "prompt": "surprise"},
            vocab_size=32,
            device="cpu",
        )


def test_workload_prompt_routes_a_text_workload_to_the_text_prompt(monkeypatch):
    """The mechanism `cmd_batch` now shares with `_build_columns`: a `prompt: "text"`
    workload must reach `_text_prompt`, not `torch.randint`. `_text_prompt` itself needs a
    real tokenizer file, so it is stubbed here -- what this proves is the routing, which is
    exactly what was missing from `cmd_batch` before this fix."""
    import deltaforge.cli as cli_module

    calls = []

    def fake_text_prompt(weights, batch, context, device):
        calls.append((weights, batch, context, device))
        return torch.zeros((batch, context), dtype=torch.long)

    monkeypatch.setattr(cli_module, "_text_prompt", fake_text_prompt)

    prompt = cli_module._workload_prompt(
        weights=Path("/weights"),
        workload={"batch_size": 2, "context_length": 8, "prompt": "text"},
        vocab_size=32,
        device="cpu",
    )

    assert calls == [(Path("/weights"), 2, 8, "cpu")]
    assert prompt.shape == (2, 8)


def test_workload_prompt_defaults_to_random_ids_when_unspecified():
    """Unspecified (the `headline`/`batch32` workloads) and explicit `"random"` must both
    keep behaving exactly as `torch.randint` always did -- this refactor changes which path
    reaches `cmd_batch`, not what either existing workload measures."""
    from .cli import _workload_prompt

    torch.manual_seed(0)
    prompt = _workload_prompt(
        weights=None, workload={"batch_size": 3, "context_length": 5}, vocab_size=17, device="cpu"
    )

    assert prompt.shape == (3, 5)
    assert prompt.dtype == torch.long
    assert int(prompt.max()) < 17


def test_cmd_batch_builds_its_prompt_through_the_shared_workload_helper():
    """Static, not behavioural: `cmd_batch` needs CUDA and a checkpoint to actually run, so
    this is the honest way to assert the fix is wired in rather than merely available.
    Before this, `cmd_batch`'s own body called `torch.randint` unconditionally and never
    referenced `_workload_prompt` or `workload["prompt"]` at all."""
    import inspect

    from .cli import cmd_batch

    source = inspect.getsource(cmd_batch)
    assert "_workload_prompt(" in source
    assert "torch.randint(" not in source


# -- fetch-weights: the disk progress watchdog -----------------------------------------
#
# Rental 55 (2026-09-30) is why this exists. `fetch-weights` sat at `4/6` for 40 minutes
# while `huggingface_hub` retried a CAS error internally and silently; the step guard in
# `remote/run_remote.sh` bounds *silence*, not *stalling*, so it waited out its whole
# 2400 s budget and the rental was destroyed having measured nothing, for $0.378. Rental 47
# sat 8:07 at the same `4/6`. The guard's premise -- a healthy step prints something -- is
# false for exactly this step, because `snapshot_download`'s bar ticks once per completed
# file and the last two files are the safetensors shards.


def test_a_growing_download_is_never_reported_as_stalled():
    from .cli import DiskProgress

    now = [0.0]
    size = [0]
    progress = DiskProgress(300.0, lambda: size[0], lambda: now[0])

    for _ in range(40):
        now[0] += 30.0
        size[0] += 1
        line, stalled = progress.poll()
        assert not stalled, line

    assert "GB" in line


def test_a_download_that_stops_growing_stalls_at_the_registered_bound():
    """Exactly at the bound, not after it: the bound is what the caller paid to choose."""
    from .cli import DiskProgress

    now = [0.0]
    progress = DiskProgress(300.0, lambda: 4_600_000_000, lambda: now[0])

    # The first poll is where these bytes are first *observed*, so that is where the clock
    # starts -- not at construction. A fetch whose first heartbeat lands 30 s in has not
    # been stalled for 30 s; it has been running for 30 s.
    now[0] = 30.0
    assert progress.poll() == ("  weights on disk: 4.60 GB (static 0s)", False)

    now[0] = 329.0
    assert progress.poll()[1] is False
    now[0] = 330.0
    assert progress.poll()[1] is True


def test_a_directory_total_that_dips_is_not_read_as_a_stall():
    """A resumable download renames `.part` files into place, so the total can dip.

    Reading a dip as absence of progress would abort a healthy fetch — which would make
    this watchdog the cause of the failure it exists to bound. `_largest` is what makes
    that impossible.
    """
    from .cli import DiskProgress

    now = [0.0]
    size = [1_000]
    progress = DiskProgress(100.0, lambda: size[0], lambda: now[0])

    now[0] = 10.0
    progress.poll()
    size[0] = 900  # a .part file renamed away mid-walk
    now[0] = 20.0
    line, stalled = progress.poll()

    assert not stalled
    assert "0.00 GB" in line
    # and the clock still runs from the last real growth, so a genuine stall after the dip
    # is still caught on time
    now[0] = 110.0
    assert progress.poll()[1] is True


def test_the_directory_sizer_counts_part_files_because_they_are_the_download(tmp_path):
    from .cli import _directory_bytes

    (tmp_path / "config.json").write_bytes(b"x" * 10)
    nested = tmp_path / "blobs"
    nested.mkdir()
    (nested / "model-00001-of-00002.safetensors.part").write_bytes(b"y" * 500)

    assert _directory_bytes(tmp_path) == 510


def test_the_stall_exit_code_is_distinct_from_a_crash():
    """A log reader must never have to guess whether a fetch crashed or went quiet."""
    from .cli import STALLED_EXIT_CODE

    assert STALLED_EXIT_CODE not in (0, 1, 2)


def test_the_watchdog_stops_when_the_download_does(tmp_path):
    """A watchdog that can call `os._exit` must not outlive its subject.

    It watches a directory for growth and kills the process when growth stops -- which is
    exactly what a *finished* download looks like. Left running, it would sit in whatever
    the process does next (a pytest session, say) and eventually exit it.
    """
    from . import cli

    stop = cli._watch_disk(tmp_path, stall_seconds=0.01, interval=3600.0)

    assert not stop.is_set()
    stop.set()
    assert stop.is_set()


def test_fetch_weights_stops_watching_even_when_the_download_raises(tmp_path, monkeypatch):
    from . import cli

    events = []
    monkeypatch.setattr(cli, "_watch_disk", lambda *a, **kw: events.append("start") or _Recorder(events))

    def boom(args, repo_id, dest):
        raise RuntimeError("hub exploded")

    monkeypatch.setattr(cli, "_fetch_weights", boom)

    with pytest.raises(RuntimeError, match="hub exploded"):
        cli.cmd_fetch_weights(_fetch_args(tmp_path))

    assert events == ["start", "stop"]


class _Recorder:
    def __init__(self, events):
        self._events = events

    def set(self):
        self._events.append("stop")
