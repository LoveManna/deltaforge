#!/usr/bin/env python
"""Where every byte goes in one decode step, and what that implies for a hypothesis.

This is the first thing to run when considering a kernel. It needs no GPU and no
checkpoint — it reads the tensor manifest in ``deltaforge/data`` — so the arithmetic that
decides whether a hypothesis is worth renting a GPU for costs nothing.

    uv run python docs/roofline.py
    uv run python docs/roofline.py --model Qwen/Qwen3.8-27B --context 32768

The ceiling of a hypothesis is the share of bytes it touches. An entry whose share is
below the harness's noise band is unmeasurable, however good the kernel is.

The arithmetic itself lives in ``deltaforge.harness.bytes_model`` so that the ceiling
printed here and the achieved bandwidth the benchmark records cannot drift apart.
"""

from __future__ import annotations

import argparse
import json

from deltaforge.config import DEFAULT_REPO_ID, MODELS, model_config
from deltaforge.harness.bytes_model import traffic, weight_bytes

#: name -> (TB/s of HBM bandwidth, GB of memory). Bandwidth is the vendor peak; real
#: kernels reach 75-90% of it, so treat these as a ceiling rather than a prediction.
GPUS = {
    "RTX 4090": (1.008, 24),
    "RTX 5090": (1.792, 32),
    "A100 80GB": (2.039, 80),
    "H100 SXM": (3.350, 80),
    "H200": (4.800, 141),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_REPO_ID, choices=sorted(MODELS))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--json", action="store_true", help="emit the table as JSON")
    args = parser.parse_args()

    config = model_config(args.model)
    groups = weight_bytes(args.model)
    context = args.context + args.decode_tokens
    rows = traffic(config, groups, args.batch, context)
    total = sum(rows.values())

    if args.json:
        print(json.dumps({"model": args.model, "total_bytes": total, "rows": rows}, indent=2))
        return 0

    print(f"{args.model} — batch {args.batch}, context {args.context}, {config.num_hidden_layers} layers\n")
    print("decode-path weights")
    for key, size in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {key:26s} {size / 1e9:8.3f} GB  {100 * size / sum(groups.values()):5.1f}%")

    print(f"\nper-token traffic{'':24s}{'MB/token':>12s}{'share = ceiling':>17s}")
    for key, size in sorted(rows.items(), key=lambda kv: -kv[1]):
        print(f"  {key:38s} {size / 1e6:11.2f} {100 * size / total:16.3f}%")
    print(f"  {'TOTAL':38s} {total / 1e6:11.2f}")

    print("\nroofline (vendor peak bandwidth; real kernels reach 75-90% of it)")
    for name, (tbs, mem) in GPUS.items():
        ms = 1000 * total / (tbs * 1e12)
        print(f"  {name:12s} {ms:7.2f} ms/token  {(tbs * 1e12) / total:6.0f} tok/s   {mem} GB")

    print("\nweight-only quantisation (hypothesis 1's ceiling)")
    weights = rows["weights, streamed once"]
    for bits, label in ((8, "fp8/int8"), (4, "int4")):
        quantised = total - weights + weights * bits / 16
        print(f"  {label:9s} total {quantised / 1e9:6.2f} GB  ->  {total / quantised:5.2f}x speedup")

    print(
        "\nA hypothesis cannot beat its share. Compare the share column against the\n"
        "harness's noise band (BenchResult.iqr_ratio) before writing a kernel."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
