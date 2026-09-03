"""Command line entry points, invoked by ``remote/run_remote.sh`` on the rented box.

Everything here except ``fetch-weights`` needs a CUDA device and the full checkpoint, so
none of it ran during the bootstrap session. The code paths are wired and the shapes are
fixed; the numbers arrive in the first funded session.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

REPO_ID = "Qwen/Qwen3.5-4B"
HF_BASE = f"https://huggingface.co/{REPO_ID}/resolve/main"

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


def cmd_fetch_weights(args: argparse.Namespace) -> int:
    """Download the checkpoint. The only command here that does not need a GPU."""
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"fetching {REPO_ID} into {dest}")
    for name in AUXILIARY_FILES:
        try:
            _download(f"{HF_BASE}/{name}", dest / name)
        except OSError as exc:
            print(f"  skipping {name}: {exc}")

    index_path = dest / "model.safetensors.index.json"
    if not index_path.exists():
        print("no safetensors index; cannot enumerate shards", file=sys.stderr)
        return 1
    shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    for shard in shards:
        _download(f"{HF_BASE}/{shard}", dest / shard)
    print(f"done: {len(shards)} shards in {dest}")
    return 0


def _load_models(args: argparse.Namespace) -> tuple[object, object, object]:
    """Build (config, eager reference, candidate). Requires CUDA and the checkpoint."""
    import torch

    from .config import from_hf_config
    from .kernels import REGISTRY
    from .model import build_model

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device available. Every command except fetch-weights is deferred to "
            "a funded GPU session by design; nothing here fabricates a number without one."
        )

    weights = Path(args.weights)
    config = from_hf_config(weights / "config.json")
    reference = build_model(config, weights, device="cuda", dtype=torch.bfloat16, registry=None)
    candidate = build_model(config, weights, device="cuda", dtype=torch.bfloat16, registry=REGISTRY)
    return config, reference, candidate


def _tokenize_prompts(weights: Path, prompts: tuple[str, ...]) -> list[list[int]]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))
    return [tokenizer.encode(prompt).ids for prompt in prompts]


def cmd_correctness(args: argparse.Namespace) -> int:
    from .harness.correctness import CorrectnessReport, check_end_to_end
    from .harness.prompts import CORRECTNESS_PROMPTS, PROMPT_DIGEST
    from .harness.report import ResultRecord, render_markdown, write_record

    _config, reference, candidate = _load_models(args)
    prompt_ids = _tokenize_prompts(Path(args.weights), CORRECTNESS_PROMPTS)

    end_to_end = check_end_to_end(
        reference,
        candidate,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        prompt_digest=PROMPT_DIGEST,
    )
    # Layer 1 has nothing to check while the registry is empty: with no kernels
    # registered there is no per-kernel comparison to make. It populates itself as soon
    # as the first kernel lands.
    report = CorrectnessReport(kernel_checks=(), end_to_end=end_to_end)

    record = ResultRecord(
        kind="hypothesis" if args.hypothesis else "baseline",
        outcome="baseline" if report.passed else "incorrect",
        config_name=REPO_ID,
        hypothesis={"slug": args.hypothesis} if args.hypothesis else None,
        correctness=report.to_dict(),
    )
    out = Path(args.output or _default_output(args, "correctness"))
    write_record(record, out)
    print(render_markdown(record))
    print(f"wrote {out}")
    return 0 if report.passed else 1


def _build_columns(args: argparse.Namespace) -> tuple[dict[str, Callable], dict[str, Callable], dict]:
    """The four columns of spec section 5, plus their untimed per-round setup.

    Each timed callable runs only the decode steps. Prefill and cache restoration happen
    in the setup, outside the measured region.
    """
    import torch

    from .model import greedy_decode

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

        def setup() -> None:
            cache.reset()
            runnable(prompt, cache, num_logits_to_keep=1)

        def run() -> None:
            greedy_decode(runnable, prompt[:, -1:], tokens, cache=cache)

        return setup, run

    columns: dict[str, Callable] = {}
    setups: dict[str, Callable] = {}
    for label, model, mode in (
        ("eager", reference, None),
        ("compiled", reference, "max-autotune"),
        ("compiled_nocudagraphs", reference, "max-autotune-no-cudagraphs"),
        ("candidate", candidate, None),
    ):
        setups[label], columns[label] = make(model, mode)
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
        metadata={"workload": args.workload},
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
        config_name=REPO_ID,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deltaforge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-weights", help="download the Qwen3.5-4B checkpoint")
    fetch.add_argument("--dest", required=True)
    fetch.set_defaults(func=cmd_fetch_weights)

    common = argparse.ArgumentParser(add_help=False)
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
    bench.add_argument("--workload", choices=sorted(DEFAULT_WORKLOADS), default="headline")
    bench.add_argument("--rounds", type=int, default=7)
    bench.add_argument("--warmup-rounds", type=int, default=2)
    bench.add_argument("--instance-id", default="")
    bench.add_argument("--hourly-rate", type=float, default=0.0)
    bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
