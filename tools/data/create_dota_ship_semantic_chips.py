#!/usr/bin/env python3
"""Create scene-disjoint target-prompted DOTA ship chips with iSAID masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from create_aircraft_semantic_chips import (
    SPLITS,
    collect_dataset_pool,
    has_instance_mask,
    scene_resplit,
    write_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dota-root", type=Path,
                        default=Path("/output/yolo_mtl_patches/dota_ships/rgb"))
    parser.add_argument("--output", type=Path,
                        default=Path("/output/vessel_semantic_chips/dota_ship_rgb_256_split_seed42"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--context-scale", type=float, default=2.25)
    parser.add_argument("--minimum-source-size", type=float, default=64.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-trials", type=int, default=10000)
    args = parser.parse_args()

    pool = [item for item in collect_dataset_pool("dota", args.dota_root)
            if has_instance_mask(item)]
    split_candidates = scene_resplit(pool, args.split_seed, args.split_trials)
    manifest = {
        "schema": "target_prompted_vessel_chips_v1",
        "mask_source": "native iSAID instance polygons matched to DOTA objects",
        "output_size": args.size,
        "context_scale": args.context_scale,
        "minimum_source_size": args.minimum_source_size,
        "split_seed": args.split_seed,
        "split_policy": "scene-level 80/10/10 after global object deduplication",
        "datasets": {"dota": {}},
    }
    for split in SPLITS:
        candidates = sorted(split_candidates[split],
                            key=lambda item: (item.scene, item.object_index,
                                              str(item.label_path)))
        records = [write_candidate(item, args.output, args.size,
                                   args.context_scale, args.minimum_source_size)
                   for item in candidates]
        masked = sum(record["has_target_mask"] for record in records)
        if masked != len(records):
            raise RuntimeError(f"{split}: created {masked}/{len(records)} masks")
        manifest["datasets"]["dota"][split] = {
            "samples": len(records),
            "target_masks": masked,
            "scenes": sorted({record["scene"] for record in records}),
        }
        print(f"dota/{split}: {len(records)} chips, {masked} native masks")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
