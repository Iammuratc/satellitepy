#!/usr/bin/env python3
"""Download blanchon/FAIR1M and persist it under one destination root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("/data/fair1m_official"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "huggingface_cache"
    saved = args.output / "dataset"

    fair1m = load_dataset("blanchon/FAIR1M", cache_dir=str(cache))
    fair1m.save_to_disk(str(saved))
    summary = {
        "source": "blanchon/FAIR1M",
        "cache": str(cache),
        "saved_dataset": str(saved),
        "splits": {name: len(split) for name, split in fair1m.items()},
        "features": {name: str(split.features) for name, split in fair1m.items()},
    }
    (args.output / "download_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
