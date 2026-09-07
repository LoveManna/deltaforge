"""Build a tensor manifest from a published checkpoint's safetensors headers.

Fetches only the header of each shard over HTTP range requests — names, dtypes and
shapes, no weight data — so a 54 GB checkpoint costs a few hundred kilobytes to describe.

    python -m deltaforge.data.make_manifest Qwen/Qwen3.5-4B > \
        src/deltaforge/data/qwen3_5_4b_manifest.json
"""

from __future__ import annotations

import json
import struct
import sys
import urllib.request


def _today() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).date().isoformat()


REPO = sys.argv[1]
BASE = f"https://huggingface.co/{REPO}/resolve/main"


def get(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


index = json.loads(get(f"{BASE}/model.safetensors.index.json"))
shards = sorted(set(index["weight_map"].values()))
tensors = {}
for shard in shards:
    url = f"{BASE}/{shard}"
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    header = json.loads(get(url, 8, 8 + n - 1))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        tensors[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "shard": shard}
    print(
        f"  {shard}: {len(header) - 1 if '__metadata__' in header else len(header)} tensors", file=sys.stderr
    )

out = {
    "_comment": "safetensors headers only; no weight data. Regenerate with scratchpad/mkmanifest.py.",
    "repo_id": REPO,
    "retrieved": _today(),
    "shards": shards,
    "total_size": index["metadata"]["total_size"],
    "tensors": tensors,
}
json.dump(out, sys.stdout, indent=1, sort_keys=True)
