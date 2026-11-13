# Sparse multi-task YOLO

The merged datasets do not share one annotation schema. A source object may
have HBB+CGC, OBB+CGC+role, both geometries, a mask, or additional airplane
attributes. `satellitepy.models.yolo_mtl` therefore treats annotation absence
as missing supervision rather than as a negative label.

## Model

`SparseMTLYOLO` uses the pretrained YOLO11s-OBB backbone and P3/P4/P5 neck.
It has independent HBB and OBB geometry heads, one shared semantic tower, and
cheap task projections for:

- coarse class (CGC), role, fine class (FGC), very-fine class (FtGC), and
  FineAir class;
- foreground masks;
- engine count and propulsion;
- canards and fuselage length;
- wingspan, wing shape, and wing position;
- tail-fin count.

The four-channel configuration initializes the IR stem weights from the mean
RGB kernel. RGB-only images receive a zero IR channel, and random IR channel
dropout keeps the model usable when IR is unavailable.

FineAir source TIFFs contain RGB in bands 1--3 and NIR in band 4. Build their
four-channel patches with the windowed reader so the NIR band is not mistaken
for alpha and large scenes are not loaded into memory at once:

```bash
PYTHONPATH=. python tools/data/create_fineair_merged_patches.py \
  --root /data/FineAir/FR24_dataset \
  --out-folder /output/yolo_mtl_patches/fineair/rgbir
```

The builder balances FineAir classes across original-scene splits, accepts
only fully contained airplanes, and writes the exact split to
`split_manifest.json`. FineAir's balanced `fineairtype` is used as the shared
`fine-class`; a meaningful subtype is retained as `very-fine-class`.

## Geometry provenance

Label readers store `_provided_geometry` in generated JSON. HBB is never
calculated from OBB. RarePlanes and ShipImageNet use both geometries because
both are explicitly present in their originals. Potsdam/Vaihingen remains the
explicit mask-derived exception. Empty patches retain provenance so an empty
labelled patch trains objectness, while a geometry-unlabelled patch is ignored
by that head.

## Training

Regenerate patches before training; older patches may contain synthesized
HBBs:

```bash
PYTHONPATH=. python tools/data/create_original_dataset_patches.py \
  --manifest docs/orig_data_paths.csv --data-root /data \
  --out-folder /output/yolo_mtl_patches --patch-size 1024 \
  --patch-overlap 100 --seed 42 --workers 16 --keep-empty
```

When `/output/dota_v2_annotations` contains the extracted train and validation
archives, the builder automatically replaces matching DOTA-v1 annotation files
with the relabelled DOTA-v2 versions. The v1 images remain unchanged. Override
or disable that location with `--dota-v2-label-root`.

Training accepts multiple split roots. Each must contain `images/` and
`labels/`:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
  tools/train/yolo_mtl.py \
  --train-folders /output/yolo_mtl_patches/dior/rgb/train \
                  /output/yolo_mtl_patches/dota/rgb/train \
  --val-folders /output/yolo_mtl_patches/dior/rgb/val \
                /output/yolo_mtl_patches/dota/rgb/val \
  --out-folder /output/yolo11s_sparse_mtl
```

The progress bar updates every batch and validation runs at the configured
interval. `last.pt` and the lowest-validation-loss `best.pt` are maintained.

Ultralytics code and weights are AGPL-3.0 unless separately licensed.
