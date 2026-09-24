"""Command line entry points, invoked by ``remote/run_remote.sh`` on the rented box.

Everything here except ``fetch-weights`` needs a CUDA device and the full checkpoint, so
none of it has ever run. The code paths are wired and the shapes are fixed; the numbers
arrive in the first session that gets a working GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .config import DEFAULT_REPO_ID, MODELS


def _hf_base(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main"


#: Fetched to the instance. Ungated, so no HuggingFace token is required.
AUXILIARY_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

DEFAULT_WORKLOADS = {
    # The headline: single-stream latency in the memory-bound regime, where fusion wins
    # are real.
    "headline": {"batch_size": 1, "context_length": 2048, "decode_tokens": 128},
    # Recorded to show behaviour as the workload becomes compute-bound.
    "batch32": {"batch_size": 32, "context_length": 2048, "decode_tokens": 128},
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    print(f"  fetching {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as response, tmp.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    tmp.replace(dest)


def _snapshot_download(repo_id: str, dest: Path) -> bool:
    """Fetch the checkpoint with ``huggingface_hub``. Returns False if it is unavailable.

    Worth the dependency: the fallback below is a single HTTP connection, sequential, with
    no resume, so an interruption 8 GB into a 9.3 GB shard starts that shard again. On a
    rented box the download is billed wall-clock time, and ``hf_transfer`` opens many
    connections at once — typically several times faster on the 300-1200 Mbit links the
    offer filter selects for.

    Two environment variables, one deprecated, because which one is honoured depends on the
    installed version and the box gets to decide: ``HF_HUB_ENABLE_HF_TRANSFER`` was the knob
    until huggingface_hub moved to Xet, which ignores it and reads
    ``HF_XET_HIGH_PERFORMANCE`` instead. Both are set before the import, because the library
    reads them at import time. If neither accelerator is present, ``huggingface_hub`` warns
    and uses its own (still parallel, still resumable) downloader, which is why this is not
    fatal.
    """
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415 - optional dependency
    except ImportError:
        return False

    # Only the decode path. The vision tower and MTP head are excluded from the benchmark
    # (see the README), and on Qwen3.5-4B they are ~0.9 GB of the checkpoint.
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=[*AUXILIARY_FILES, "*.safetensors"],
        max_workers=8,
    )
    return True


def cmd_fetch_weights(args: argparse.Namespace) -> int:
    """Download the checkpoint. The only command here that does not need a GPU."""
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    repo_id = args.model
    print(f"fetching {repo_id} into {dest}")

    if not args.no_hf_transfer:
        try:
            if _snapshot_download(repo_id, dest):
                shards = sorted(dest.glob("*.safetensors"))
                total = sum(f.stat().st_size for f in shards)
                print(f"done: {len(shards)} shards, {total / 1e9:.2f} GB in {dest}")
                return 0
            print("  huggingface_hub not installed; falling back to urllib")
        except Exception as exc:  # noqa: BLE001 - any failure here is recoverable
            print(f"  huggingface_hub download failed ({exc}); falling back to urllib")

    base = _hf_base(repo_id)
    for name in AUXILIARY_FILES:
        try:
            _download(f"{base}/{name}", dest / name)
        except OSError as exc:
            print(f"  skipping {name}: {exc}")

    index_path = dest / "model.safetensors.index.json"
    if not index_path.exists():
        print("no safetensors index; cannot enumerate shards", file=sys.stderr)
        return 1
    shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    for shard in shards:
        _download(f"{base}/{shard}", dest / shard)
    print(f"done: {len(shards)} shards in {dest}")
    return 0


def _load_models(args: argparse.Namespace) -> tuple[object, object, object]:
    """Build (config, eager reference, candidate). Requires CUDA and the checkpoint.

    The two models are **separate module trees that share one set of parameter tensors.**

    Separate trees are required: installing a kernel swaps ``__class__`` on the candidate's
    modules, and a shared tree would silently alter the reference — benchmarking the
    candidate against itself.

    Shared parameters are safe and worth a lot. Nothing writes to a parameter here: every
    forward runs under ``no_grad`` and the harness only ever reads them. Loading twice
    instead cost a second 9.3 GB read from disk in each of the three commands that build
    models, and held 16.8 GB on a 32 GB card where 8.4 GB does. ``assign=True`` is what
    makes ``load_state_dict`` rebind the tensors rather than copy into new storage.

    A kernel that genuinely needs different weights — a quantised candidate, say — should
    replace them explicitly after this returns, and say so in its writeup.
    """
    import torch

    from .config import from_hf_config
    from .model import apply_champions, build_model
    from .reference import ReferenceModel

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device available. Every command except fetch-weights is deferred to "
            "a funded GPU session by design; nothing here fabricates a number without one."
        )

    weights = Path(args.weights)
    config = from_hf_config(weights / "config.json")
    reference = build_model(config, weights, device="cuda", dtype=torch.bfloat16, registry=None)

    candidate = ReferenceModel(config).to(device="cuda", dtype=torch.bfloat16).eval()
    candidate.load_state_dict(reference.state_dict(), assign=True)
    apply_champions(candidate, _candidate_registry(args))
    _assert_parameters_are_shared(reference, candidate)
    return config, reference, candidate


def _candidate_registry(args: argparse.Namespace):
    """Which kernels the candidate column installs: `--install`, or the global champions.

    `--install` exists for the `output_code` dump. Without it the dump renders whatever is
    registered CHAMPION, which is one kernel — and the composition this project cannot
    explain is two. Rental 38 dumped a candidate with no kernel in it at all and entry 5 of
    `docs/HYPOTHESES.md` has been open ever since asking for the dump to be pointed at a
    real slot; naming the kernels is how a step does that without being a batch.
    """
    from .batch import registry_for  # noqa: PLC0415
    from .kernels import REGISTRY  # noqa: PLC0415

    requested = [name.strip() for name in (getattr(args, "install", "") or "").split(",") if name.strip()]
    if not requested:
        return REGISTRY
    print(f"[bench] installing {requested} instead of the registered champions")
    return registry_for(requested, label="cli --install")


def _assert_parameters_are_shared(reference, candidate) -> None:
    """Fail loudly if the candidate quietly stopped sharing the reference's weights.

    Checked rather than assumed, because the failure is invisible: the run would still
    produce a plausible number, on twice the memory, having silently loaded a second copy.
    """
    ref = dict(reference.named_parameters())
    for name, param in candidate.named_parameters():
        target = ref.get(name)
        if target is None:
            raise SystemExit(f"candidate parameter {name!r} has no counterpart in the reference")
        if param.data_ptr() != target.data_ptr():
            raise SystemExit(
                f"candidate parameter {name!r} does not share storage with the reference. "
                "The two models must differ only in which kernels they call."
            )


def _tokenize_prompts(weights: Path, prompts: tuple[str, ...]) -> list[list[int]]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))
    return [tokenizer.encode(prompt).ids for prompt in prompts]


def cmd_correctness(args: argparse.Namespace) -> int:
    from .harness.correctness import CorrectnessReport, check_end_to_end
    from .harness.prompts import CORRECTNESS_PROMPTS, PROMPT_DIGEST
    from .harness.report import ResultRecord, render_markdown, write_record
    from .kernels import build_kernel_checks

    _config, reference, candidate = _load_models(args)
    prompt_ids = _tokenize_prompts(Path(args.weights), CORRECTNESS_PROMPTS)

    # Layer 1: every champion kernel against the reference operation it replaces, on real
    # shapes. Checks run against `reference`, whose modules are untouched by any install.
    kernel_checks = build_kernel_checks(reference, device="cuda")

    end_to_end = check_end_to_end(
        reference,
        candidate,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        prompt_digest=PROMPT_DIGEST,
    )
    report = CorrectnessReport(kernel_checks=kernel_checks, end_to_end=end_to_end)

    record = ResultRecord(
        kind="hypothesis" if args.hypothesis else "baseline",
        outcome="baseline" if report.passed else "incorrect",
        config_name=args.model,
        hypothesis={"slug": args.hypothesis} if args.hypothesis else None,
        correctness=report.to_dict(),
    )
    out = Path(args.output or _default_output(args, "correctness"))
    write_record(record, out)
    print(render_markdown(record))
    print(f"wrote {out}")
    return 0 if report.passed else 1


#: label -> (which model, torch.compile mode or None).
#:
#: `compiled` vs `candidate_compiled` is the claim: identical treatment, the only
#: difference being who wrote the kernel. The two eager columns are nearly free — they add
#: timing runs, not compilations — and each earns its place. `eager` is the sanity check
#: that the compiler did anything at all; the gap between `candidate` and
#: `candidate_compiled` attributes a win between the compiler and the kernel.
BENCH_COLUMNS: dict[str, tuple[str, str | None]] = {
    "eager": ("reference", None),
    "compiled": ("reference", "max-autotune"),
    "compiled_nocudagraphs": ("reference", "max-autotune-no-cudagraphs"),
    "candidate": ("candidate", None),
    "candidate_compiled": ("candidate", "max-autotune"),
}

#: Without both of these there is no result, so they cannot be dropped.
SCORING_COLUMNS = ("compiled", "candidate_compiled")

#: What runs unless `--columns` says otherwise: the two that score, and nothing else.
#:
#: `eager` and `candidate` are diagnostics — the gap between them is the compiler's
#: contribution — but every column is a live model state on the card with its own KV cache
#: and, when compiled, its own CUDA-graph pool, and all of them stay resident through
#: warmup. Rental 21 OOMed there at 30.71 GiB of 31.36 while construction accounted for
#: 0.11. Two of those four columns scored nothing.
#:
#: `compiled_nocudagraphs` is in `all` and not here for a second reason: it costs a full
#: `max-autotune` compilation, and the confound it existed to rule out disappeared when the
#: scoring column became `candidate_compiled`, since both sides now have CUDA graphs.
#:
#: `--columns all` asks for the full set when a result is confusing enough to be worth the
#: memory and the compile, or when a hypothesis is itself about launch overhead.
#: Benchmark rounds a *batch* runs, against `BenchConfig`'s default of 7 for a single
#: hypothesis. Two are discarded as warmup, so this is **15 scoring rounds**. See the
#: comment on ``--rounds`` in `build_parser`: rental 45 lost six slots to a band it could
#: have narrowed for 20 seconds each.
BATCH_ROUNDS = 17

DEFAULT_COLUMNS = SCORING_COLUMNS


def _selected_columns(args: argparse.Namespace) -> tuple[str, ...]:
    """Which benchmark columns this run measures. Recorded alongside the numbers."""
    requested = getattr(args, "columns", "") or ""
    if requested == "all":
        labels = tuple(BENCH_COLUMNS)
    elif requested:
        labels = tuple(label.strip() for label in requested.split(",") if label.strip())
    else:
        labels = DEFAULT_COLUMNS

    unknown = [label for label in labels if label not in BENCH_COLUMNS]
    if unknown:
        raise SystemExit(f"unknown benchmark column(s): {unknown}; known: {sorted(BENCH_COLUMNS)}")
    missing = [label for label in SCORING_COLUMNS if label not in labels]
    if missing:
        raise SystemExit(
            f"refusing to run without the scoring column(s) {missing}. "
            f"{SCORING_COLUMNS[0]!r} against {SCORING_COLUMNS[1]!r} is the measurement; "
            "a run without both produces no result."
        )
    return labels


def _build_columns(args: argparse.Namespace) -> tuple[dict[str, Callable], dict[str, Callable], dict]:
    """The four columns of spec section 5, plus their untimed per-round setup.

    Each timed callable runs only the decode steps. Prefill and cache restoration happen
    in the setup, outside the measured region.
    """
    import torch

    from .model import greedy_decode, prefill_setup

    _config, reference, candidate = _load_models(args)
    workload = DEFAULT_WORKLOADS[args.workload]
    batch = workload["batch_size"]
    context = workload["context_length"]
    tokens = workload["decode_tokens"]

    prompt = torch.randint(0, reference.config.vocab_size, (batch, context), device="cuda", dtype=torch.long)

    def make(model, compile_mode: str | None):
        runnable = model
        if compile_mode is not None:
            runnable = torch.compile(model, mode=compile_mode)
        cache = model.new_cache(batch, context + tokens)
        # Prefilled once and restored per round; `model.prefill_setup` says what that saves
        # and why it is sound.
        setup = prefill_setup(model, prompt, cache)

        def run() -> None:
            greedy_decode(runnable, prompt[:, -1:], tokens, cache=cache)

        return setup, run

    columns: dict[str, Callable] = {}
    setups: dict[str, Callable] = {}
    models = {"reference": reference, "candidate": candidate}
    for label in _selected_columns(args):
        which, mode = BENCH_COLUMNS[label]
        setups[label], columns[label] = make(models[which], mode)
    return columns, setups, workload


def cmd_bench(args: argparse.Namespace) -> int:
    from .harness.bench import BenchConfig, CudaEventTimer, run_interleaved
    from .harness.report import ResultRecord, render_markdown, write_record

    columns, setups, workload = _build_columns(args)
    result = run_interleaved(
        columns,
        BenchConfig(rounds=args.rounds, warmup_rounds=args.warmup_rounds),
        timer=CudaEventTimer(),
        setups=setups,
        metadata={"workload": args.workload, "columns": list(_selected_columns(args))},
    )

    cost = None
    if args.instance_id:
        cost = {
            "instance_id": args.instance_id,
            "hourly_rate_usd": args.hourly_rate,
            "gpu_model": None,
            "actual_minutes": None,
            "actual_cost_usd": None,
        }

    record = ResultRecord(
        kind="hypothesis" if args.hypothesis else "baseline",
        # A bootstrap run records the baseline; a hypothesis run's outcome is decided by
        # the session against the incumbent and the noise band, not asserted here.
        outcome="baseline" if not args.hypothesis else "inconclusive",
        config_name=args.model,
        hypothesis={"slug": args.hypothesis} if args.hypothesis else None,
        workload=workload,
        bench=result.to_dict(),
        cost=cost,
    )
    out = Path(args.output or _default_output(args, "bench"))
    write_record(record, out)
    print(render_markdown(record))
    print(f"wrote {out}")
    return 0


def _default_output(args: argparse.Namespace, kind: str) -> Path:
    root = Path(__file__).resolve().parents[2] / "results"
    session = getattr(args, "session_id", "unknown") or "unknown"
    if getattr(args, "hypothesis", ""):
        return root / "hypotheses" / f"{args.hypothesis}-{kind}-{session}.json"
    return root / "baseline" / f"{kind}-{session}.json"


# --------------------------------------------------------------------------------------
# Batch mode
# --------------------------------------------------------------------------------------


def _load_reference_only(args):
    """The reference model alone. Batch mode builds candidates one at a time instead.

    `_load_models` builds a reference *and* one candidate, which is the right shape for a
    single-hypothesis run and the wrong one for a batch: there are N candidates and only
    one of them may be resident at a time.
    """
    import torch

    from .config import from_hf_config
    from .model import build_model

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device available. Every command except fetch-weights is deferred to "
            "a funded GPU session by design; nothing here fabricates a number without one."
        )
    weights = Path(args.weights)
    config = from_hf_config(weights / "config.json")
    reference = build_model(config, weights, device="cuda", dtype=torch.bfloat16, registry=None)
    return config, reference


def _write_phases_env(path: Path, phases: dict[str, float], results) -> None:
    """Record what this rental actually cost, for the next one's pre-flight check.

    Shell `KEY=VALUE` lines rather than JSON: `run_remote.sh` sources this file, and it has
    to work before any python environment exists. The keys match
    `batch.COLD_PHASE_ESTIMATES`, which is what they replace.

    `slot_s` is the **longest** slot rather than the median. The check it feeds decides
    whether to rent at all, and being wrong in the optimistic direction costs a rental that
    buys nothing — which is the failure this whole change exists to stop.

    **A slot that hit its cap is excluded**, because it did not finish and so measures
    nothing about what finishing costs. It is the maximum by construction, so including it
    writes the cap itself into the gate: rental 32 recorded 6983.3s that way, which needs
    310 minutes against a 180-minute session and refuses every subsequent rental on that
    card. A slot that *failed* still counts — a kernel that raised did its work first, and
    dropping every error would bias the estimate the optimistic way.
    """
    from .slot_timer import SlotTimeout

    cut_off = f"{SlotTimeout.__name__}:"
    reference_compile_s = sum(
        seconds for name, seconds in phases.items() if name.endswith(".compile_compiled")
    )
    slot_times = [
        r.duration_s
        for r in results
        if r.duration_s and r.outcome != "not_run" and not (r.error or "").startswith(cut_off)
    ]
    measured = {
        "DF_PHASE_REFERENCE_COMPILE_S": reference_compile_s,
        "DF_PHASE_SLOT_S": max(slot_times) if slot_times else 0.0,
    }
    lines = [f"{key}={value:.1f}\n" for key, value in measured.items() if value > 0]
    if not lines:
        print(f"[batch] no phase timings to record; leaving {path} alone")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    print(f"[batch] wrote {path}")


def cmd_batch(args: argparse.Namespace) -> int:
    """Measure a whole batch of hypotheses on one rental."""
    import time

    import torch

    from .batch import SlotBudget
    from .batch_run import BatchRunner, run_batch, slot_record_path
    from .batches import get_batch
    from .harness.bench import BenchConfig
    from .harness.prompts import CORRECTNESS_PROMPTS
    from .harness.report import (
        BatchRecord,
        ResultRecord,
        render_batch_markdown,
        write_batch_record,
        write_record,
    )

    batch = get_batch(args.batch)
    columns = _selected_columns(args)
    workload = DEFAULT_WORKLOADS[args.workload]

    deadline = args.deadline_epoch if args.deadline_epoch > 0 else time.time() + args.batch_minutes * 60
    budget = SlotBudget(deadline_epoch=deadline)
    print(
        f"[batch] {batch.batch_id}: {len(batch)} hypotheses, "
        f"{budget.remaining_s() / 60:.1f} minutes until the deadline"
    )

    config, reference = _load_reference_only(args)
    prompt_ids = _tokenize_prompts(Path(args.weights), CORRECTNESS_PROMPTS)
    prompt = torch.randint(
        0,
        reference.config.vocab_size,
        (workload["batch_size"], workload["context_length"]),
        device="cuda",
        dtype=torch.long,
    )

    runner = BatchRunner(
        config=config,
        reference=reference,
        prompt=prompt,
        prompt_ids=prompt_ids,
        workload=workload,
        weights_dtype=torch.bfloat16,
        bench_config=BenchConfig(rounds=args.rounds, warmup_rounds=args.warmup_rounds),
        max_new_tokens=args.max_new_tokens,
        columns=columns,
    )

    root = Path(args.output or Path(__file__).resolve().parents[2] / "results")

    def write_slot(result) -> None:
        """One record per slot, written the moment the slot finishes.

        A hard crash at slot 8 must leave slots 0-7 on disk. Writing at the end of the
        batch would put a whole rental of paid measurement behind a single point of
        failure.
        """
        record = ResultRecord(
            kind="hypothesis",
            outcome=result.outcome,
            config_name=args.model,
            hypothesis={
                "slug": result.hypothesis.slug,
                "batch": batch.batch_id,
                "statement": result.hypothesis.mechanism,
                "prediction": result.hypothesis.prediction,
                "rationale": result.hypothesis.rationale,
                "category": result.hypothesis.category,
                "byte_share": result.hypothesis.byte_share,
                "kernels": list(result.hypothesis.kernels),
            },
            workload=workload,
            bench=result.bench,
            correctness=result.correctness,
            phases=result.phases_s,
            notes=result.error or "",
        )
        path = slot_record_path(root, batch.batch_id, result.hypothesis.slug)
        write_record(record, path)
        print(f"[batch] wrote {path}")

    results, calibrated, scores = run_batch(runner, batch, budget=budget, on_slot=write_slot)

    score_by_slug = {s.slug: s.correct for s in scores}
    slots = []
    for result in results:
        slot = result.to_slot_dict()
        slot["prediction_correct"] = score_by_slug.get(result.hypothesis.slug)
        slots.append(slot)

    cost = None
    if args.instance_id:
        cost = {
            "instance_id": args.instance_id,
            "hourly_rate_usd": args.hourly_rate,
            "gpu_model": None,
            "actual_minutes": None,
            "actual_cost_usd": None,
        }

    record = BatchRecord(
        batch_id=batch.batch_id,
        session_id=args.session_id,
        config_name=args.model,
        description=batch.description,
        slots=slots,
        calibrated=calibrated,
        predictions=[s.to_dict() for s in scores],
        workload=workload,
        cost=cost,
        # Flattened `<slug>.<phase>` so one file answers "where did the rental go" without
        # opening nine slot records. The batch's cost arithmetic was an estimate that a
        # rental contradicted; this is the measurement that replaces it.
        phases={
            f"{r.hypothesis.slug}.{name}": seconds for r in results for name, seconds in r.phases_s.items()
        },
    )
    summary = root / "batches" / batch.batch_id / "summary.json"
    write_batch_record(record, summary)
    print(render_batch_markdown(record))
    print(f"[batch] wrote {summary}")

    if args.phases_env:
        _write_phases_env(Path(args.phases_env), record.phases, results)

    # Exit 0 whenever the batch ran to its own conclusion. An errored slot is a recorded
    # result, not a failed run, and failing the process here would make `run_remote.sh`
    # tear down as if the rental had gone wrong — losing the slots that did succeed.
    if calibrated is False:
        print(
            "[batch] CALIBRATION FAILED: the identity champion did not measure 1.00. "
            "Every other number in this batch is void; see the summary."
        )
    return 0


def compile_and_capture(model, prompt, cache) -> str:
    """Compile one decode step and return the generated code inductor logged.

    In-process rather than through a subprocess so the caller can hold two variants of one
    model in one interpreter, which is the whole point: the comparison is between two
    programs, and anything that differs between two runs of the same process is noise in it.
    """
    import io  # noqa: PLC0415
    import logging  # noqa: PLC0415

    import torch  # noqa: PLC0415

    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    logger = logging.getLogger("torch._inductor")
    previous = logger.level
    torch._logging.set_logs(output_code=True)
    logger.addHandler(handler)
    try:
        with torch.no_grad():
            model(prompt, cache, num_logits_to_keep=1)
            step = prompt[:, -1:]
            torch.compile(model)(step, cache, num_logits_to_keep=1)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
        torch._logging.set_logs()
    return captured.getvalue()


def cmd_fusion(args: argparse.Namespace) -> int:
    """Read what the compiler does with a candidate, on this machine, before renting one.

    Three of the last four rentals were decided by generated code rather than by a ratio,
    and both of this project's champions were found by reading it. This runs the same
    diagnostic on the tiny CPU config, where a slot costs seconds instead of ~400 s of
    billed time. `deltaforge.fusion` documents what a CPU dump can and cannot settle: it is
    the barrier question, not the matmul-prologue question.
    """
    import torch  # noqa: PLC0415

    from .batch import registry_for  # noqa: PLC0415
    from .batches import get_batch  # noqa: PLC0415
    from .config import tiny_config  # noqa: PLC0415
    from .fusion import (  # noqa: PLC0415
        barrier_preflight,
        diff_reports,
        opaque_kernels,
        parse_output_code,
        render_diff,
    )
    from .model import apply_champions  # noqa: PLC0415
    from .reference import ReferenceModel  # noqa: PLC0415

    if args.batch:
        # The static half, which needs neither a compile nor a card: a `CustomOpDef` is a
        # barrier wherever it runs, and a batch that installs one without measuring the same
        # program without it cannot tell a slow kernel from an expensive registration.
        batch = get_batch(args.batch)
        for hyp in batch:
            opaque = opaque_kernels(hyp.kernels)
            if opaque:
                partner = hyp.contrast_with or next(
                    (o.slug for o in batch if o.contrast_with == hyp.slug), None
                )
                against = f"contrasted with {partner}" if partner else "NOT CONTRASTED"
                print(f"[fusion] {hyp.slug}: opaque {list(opaque)} -- {against}")
        unpartnered = barrier_preflight(batch)
        if unpartnered:
            print(
                f"[fusion] {len(unpartnered)} slot(s) install a fusion barrier with nothing "
                f"measuring the alternative: {list(unpartnered)}. That is the 21% the conv "
                "lost and the 3.2% the head lost, and neither was visible from its own slot."
            )
            return 1
        print("[fusion] no unpartnered fusion barrier in this batch")
        return 0

    kernels = [name.strip() for name in args.install.split(",") if name.strip()]
    if not kernels:
        raise SystemExit("--install names the kernels to read; with none there is nothing to compare")

    config = tiny_config()
    torch.manual_seed(0)
    prompt = torch.randint(0, config.vocab_size, (1, args.context), dtype=torch.long)

    dumps: dict[str, str] = {}
    for label, names in (("reference", []), ("candidate", kernels)):
        torch._dynamo.reset()
        model = ReferenceModel(config).eval()
        if names:
            applied = apply_champions(model, registry_for(names, label=f"cli fusion {label}"))
            print(f"[fusion] candidate installs {list(applied)}")
        dumps[label] = compile_and_capture(model, prompt, model.new_cache(1, args.context + 1))

    if args.save:
        for label, text in dumps.items():
            path = Path(args.save).with_suffix("").parent / f"{Path(args.save).name}-{label}.txt"
            path.write_text(text)
            print(f"[fusion] wrote {path}")

    diff = diff_reports(parse_output_code(dumps["reference"]), parse_output_code(dumps["candidate"]))
    print(render_diff(diff))
    # Exit 1 on a barrier so a pre-rental check can be wired into a manifest test or a
    # shell script without parsing prose. Everything else is advisory: a split on CPU is
    # worth reading, but only a dispatch is a barrier wherever it runs.
    return 1 if diff.introduced_barriers else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deltaforge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    model_arg = argparse.ArgumentParser(add_help=False)
    model_arg.add_argument(
        "--model",
        default=DEFAULT_REPO_ID,
        choices=sorted(MODELS),
        help=f"published checkpoint to benchmark (default: {DEFAULT_REPO_ID})",
    )

    fetch = sub.add_parser("fetch-weights", parents=[model_arg], help="download a published checkpoint")
    fetch.add_argument("--dest", required=True)
    fetch.add_argument(
        "--no-hf-transfer",
        action="store_true",
        help="skip huggingface_hub and use the single-connection urllib fallback",
    )
    fetch.set_defaults(func=cmd_fetch_weights)

    common = argparse.ArgumentParser(add_help=False, parents=[model_arg])
    common.add_argument("--weights", required=True, help="checkpoint directory")
    common.add_argument("--session-id", default="")
    common.add_argument("--hypothesis", default="")
    common.add_argument("--output", default="")

    correctness = sub.add_parser(
        "correctness", parents=[common], help="run both correctness gates (needs CUDA)"
    )
    correctness.add_argument("--max-new-tokens", type=int, default=128)
    correctness.set_defaults(func=cmd_correctness)

    bench = sub.add_parser("bench", parents=[common], help="run the interleaved benchmark (needs CUDA)")
    bench.add_argument(
        "--install",
        default="",
        help=(
            "comma-separated kernel names for the candidate column, in place of whatever "
            "the registry holds as champion. The dump step uses it to render the code "
            "inductor generates for a composition rather than for one kernel."
        ),
    )
    bench.add_argument("--workload", choices=sorted(DEFAULT_WORKLOADS), default="headline")
    bench.add_argument(
        "--columns",
        default="",
        help=(
            "comma-separated benchmark columns, or 'all'. Default: "
            f"{','.join(DEFAULT_COLUMNS)}. Each max-autotune column costs a full "
            "compilation, so 'all' is for calibration runs."
        ),
    )
    bench.add_argument("--rounds", type=int, default=7)
    bench.add_argument("--warmup-rounds", type=int, default=2)
    bench.add_argument("--instance-id", default="")
    bench.add_argument("--hourly-rate", type=float, default=0.0)
    bench.set_defaults(func=cmd_bench)

    batch = sub.add_parser(
        "batch",
        parents=[common],
        help="measure a whole batch of hypotheses on one rental (needs CUDA)",
    )
    batch.add_argument(
        "--batch",
        default="001-calibration",
        help="batch id from deltaforge.batches (default: 001-calibration)",
    )
    batch.add_argument("--workload", choices=sorted(DEFAULT_WORKLOADS), default="headline")
    batch.add_argument(
        "--columns",
        default="",
        help=(
            "comma-separated benchmark columns, or 'all'. Default: "
            f"{','.join(DEFAULT_COLUMNS)}. The reference columns are compiled once for the "
            "whole batch; each candidate column costs one compilation per hypothesis."
        ),
    )
    # 17 rather than the single-hypothesis default of 7, because rental 45 showed what 5
    # scoring rounds buy on a card that drifts. Its SM clock fell 2910 -> 2400 MHz mid-batch,
    # the reference column moved 7.17 -> 7.92 ms/token, and **six of eleven slots came back
    # `inconclusive` at IQRs of 0.019-0.151** -- on effects that are probably real. A round
    # is ~2 s of wall clock for both columns against slots that cost 130-330 s, so tripling
    # the sample costs about 20 s a slot and 4 minutes a batch. That is the cheapest
    # resolution this project can buy, and `SlotBudget` already prices slots generously
    # enough to absorb it.
    batch.add_argument("--rounds", type=int, default=BATCH_ROUNDS)
    batch.add_argument("--warmup-rounds", type=int, default=2)
    batch.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help=(
            "tokens for the layer-2 exact-match gate (default: 32). The single-hypothesis "
            "path uses 128; batch mode trades that for slots and records the number it used "
            "in every result. Re-check anything being promoted at 128."
        ),
    )
    batch.add_argument(
        "--deadline-epoch",
        type=float,
        default=0.0,
        help=(
            "absolute unix time the batch must stop by, passed down by run_remote.sh from "
            "the session gate. The batch stops itself before a slot it cannot finish, so "
            "the watchdog never has to — a watchdog firing is a reportable fault."
        ),
    )
    batch.add_argument(
        "--phases-env",
        default="",
        help=(
            "write this rental's measured phase costs here as shell KEY=VALUE lines. "
            "run_remote.sh sources the file before the next rental, so the pre-flight "
            "check uses measurements rather than the cold estimates in batch.py."
        ),
    )
    batch.add_argument(
        "--batch-minutes",
        type=float,
        default=60.0,
        help="fallback deadline, in minutes from now, when --deadline-epoch is not given",
    )
    batch.add_argument("--instance-id", default="")
    batch.add_argument("--hourly-rate", type=float, default=0.0)
    batch.set_defaults(func=cmd_batch)

    fusion = sub.add_parser(
        "fusion",
        help="compile the tiny config on CPU and diff what inductor generated (no GPU, no weights)",
    )
    fusion.add_argument(
        "--batch",
        default="",
        help="read a manifest's registrations instead of compiling: which slots install a barrier",
    )
    fusion.add_argument(
        "--install",
        default="",
        help="comma-separated kernel names to install on the candidate, as `bench --install` takes",
    )
    fusion.add_argument(
        "--context",
        type=int,
        default=8,
        help="prompt length for the prefill that precedes the compiled decode step (default: 8)",
    )
    fusion.add_argument("--save", default="", help="write both raw dumps next to this path")
    fusion.set_defaults(func=cmd_fusion)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
