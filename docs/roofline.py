#!/usr/bin/env python
"""Where every byte goes in one decode step, and what that implies for a hypothesis.

This is the first thing to run when considering a kernel. It needs no GPU and no
checkpoint — it reads the tensor manifest in ``deltaforge/data`` — so the arithmetic that
decides whether a hypothesis is worth renting a GPU for costs nothing.

    uv run python docs/roofline.py
    uv run python docs/roofline.py --model Qwen/Qwen3.8-27B --context 32768

The ceiling of a hypothesis is the share of bytes it touches. An entry whose share is
below the harness's noise band is unmeasurable, however good the kernel is.
"""

from __future__ import annotations

import argparse
import json

from deltaforge.config import DEFAULT_REPO_ID, MODELS, ModelConfig, model_config
from deltaforge.weights import load_manifest, map_checkpoint_name

BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8}

#: name -> (TB/s of HBM bandwidth, GB of memory). Bandwidth is the vendor peak; real
#: kernels reach 75-90% of it, so treat these as a ceiling rather than a prediction.
GPUS = {
    "RTX 4090": (1.008, 24),
    "RTX 5090": (1.792, 32),
    "A100 80GB": (2.039, 80),
    "H100 SXM": (3.350, 80),
    "H200": (4.800, 141),
}


def _tensor_bytes(meta: dict) -> int:
    n = 1
    for dim in meta["shape"]:
        n *= dim
    return n * BYTES.get(meta["dtype"], 2)


def weight_bytes(repo_id: str) -> dict[str, int]:
    """Decode-path weight bytes, grouped. Vision tower and MTP head are excluded."""
    groups: dict[str, int] = {}
    for name, meta in load_manifest(repo_id)["tensors"].items():
        if map_checkpoint_name(name) is None:
            continue  # vision tower or MTP head: not in the decode path
        size = _tensor_bytes(meta)
        if "embed_tokens" in name:
            key = "embeddings"
        elif name.startswith("lm_head"):
            key = "lm_head"
        elif ".mlp." in name:
            key = "mlp"
        elif "linear_attn" in name:
            key = "linear_attn"
        elif "self_attn" in name:
            key = "full_attn"
        else:
            key = "norms"
        groups[key] = groups.get(key, 0) + size
    return groups


def traffic(config: ModelConfig, groups: dict[str, int], batch: int, context: int) -> dict[str, int]:
    """Bytes moved per decoded token, by what moves them."""
    hidden, inter = config.hidden_size, config.intermediate_size
    layers = config.num_hidden_layers
    full = len(config.layer_indices("full_attention"))
    linear = len(config.layer_indices("linear_attention"))
    bf = 2

    weights = sum(groups.values())
    if config.tie_word_embeddings:
        # One tensor serves both; it is read in full for the lm_head matmul, and the
        # embedding lookup reads a single row. Counting it once is correct.
        pass
    else:
        # The embedding matrix is read one row per token, not in full.
        weights -= groups.get("embeddings", 0)

    kv = full * 2 * config.num_key_value_heads * context * config.head_dim * bf * batch
    # `repeat_interleave` to expand KV heads to query heads materialises `groups` copies,
    # written then read back by the attention matmul.
    expand = kv * config.num_key_value_groups * 2
    state = (
        linear
        * config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * 4  # fp32, fixed by mamba_ssm_dtype
        * 2  # read + write
        * batch
    )
    # Two residual+norm pairs per layer: add reads x and residual and writes residual,
    # then the norm reads and writes.
    norms = 2 * layers * (hidden * bf * 3 + hidden * bf * 2) * batch
    swiglu = layers * 4 * inter * bf * batch
    qkv = (
        full
        * 2
        * (2 * config.num_attention_heads + 2 * config.num_key_value_heads)
        * config.head_dim
        * bf
        * batch
    )
    return {
        "weights, streamed once": weights,
        "GQA repeat_interleave materialisation": expand,
        "recurrent state, read + write": state,
        "KV cache read": kv,
        "SwiGLU intermediates": swiglu,
        "norm + residual": norms,
        "QKV / RoPE intermediates": qkv,
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
