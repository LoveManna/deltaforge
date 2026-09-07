"""Checkpoint tensor manifests: names, dtypes and shapes, no weight data.

They let the CPU test suite prove the name map and every reference parameter shape
against a real published checkpoint without downloading it. Regenerate one with::

    python -m deltaforge.data.make_manifest Qwen/Qwen3.8-27B > \\
        src/deltaforge/data/qwen3_8_27b_manifest.json
"""
