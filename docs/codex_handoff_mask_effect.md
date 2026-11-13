# Codex handoff: sparse MTL YOLO and partial-mask experiments

Updated: 2026-10-29 (Europe/Berlin)

## Objective

The repository now contains a sparse-supervision YOLO11s MTL model intended to
train simultaneously on heterogeneous annotation combinations. The partial-mask
experiment reproduces the setup described in `docs/IGARSS_2025___MTL.pdf` on
DOTA-derived patches and Potsdam/Vaihingen patches.

The active experiment tasks are:

- single-class oriented object detection (`obboxes`),
- coarse-grained object classification (`coarse-class`), and
- foreground segmentation (`masks`).

Only annotations explicitly supplied by the original source are used. Missing
HBB/OBB/task annotations are masked out; HBB is not fabricated from OBB.

## Important paths

### Code and configuration

- Trainer: `tools/train/yolo_mtl.py`
- Matrix scheduler: `tools/train/run_mask_effect_matrix.py`
- Model: `satellitepy/models/yolo_mtl/model.py`
- Sparse losses and adaptive balancing: `satellitepy/models/yolo_mtl/loss.py`
- Dataset/collation: `satellitepy/models/yolo_mtl/data.py`
- Seed-1 config: `configs/yolo_mtl/yolo11s_mask_effect_seed1.yaml`
- Seed-2 Potsdam config: `configs/yolo_mtl/yolo11s_mask_effect_seed2.yaml`
- Mask-effect subset builder: `tools/data/create_mask_effect_subsets.py`
- iSAID-to-DOTA mask addition: `tools/data/add_isaid_masks_to_dota_patches.py`
- Retrospective mAP evaluator: `tools/evaluate/evaluate_yolo_mtl_checkpoints.py`
- Retrospective mask-IoU evaluator: `tools/evaluate/evaluate_yolo_mtl_mask_iou.py`

### Data and outputs

- Main patches: `/output/yolo_mtl_patches`
- Experiment root: `/output/yolo_mask_effect`
- DOTA experiment data: `/output/yolo_mask_effect/dota/data`
- Potsdam experiment data: `/output/yolo_mask_effect/potsdam/data`
- Original/unseeded checkpoints: `/output/yolo_mask_effect/{dota,potsdam}/runs/<condition>`
- Explicit seed 1: `/output/yolo_mask_effect/{dota,potsdam}/runs/<condition>_seed1`
- Potsdam seed 2: `/output/yolo_mask_effect/potsdam/runs/<condition>_seed2`

## Annotation conditions

There are 14 conditions per dataset:

```text
cgc100_mask000  cgc100_mask010  cgc100_mask025  cgc100_mask050
cgc100_mask100  cgc050_mask000  cgc050_mask010  cgc050_mask025
cgc050_mask050  cgc025_mask000  cgc025_mask010  cgc025_mask025
cgc010_mask000  cgc010_mask010
```

`cgcNNN` is the retained coarse-class annotation percentage and `maskNNN` is
the retained segmentation-mask percentage.

## Model and loss behavior

- Input size: 1024 x 1024 RGB patches.
- Backbone/head initialization: `yolo11s-obb.pt`.
- Global batch size per independent one-GPU experiment: 8.
- Four conditions are scheduled concurrently, one per physical GPU.
- Each condition is an independent experiment, not DDP.
- OBB, coarse-class, and mask losses are sparse-supervision aware.
- EMA magnitude normalization balances the task losses.
- `coarse-class` has a balance target of 2.0; OBB and masks use 1.0.
- W&B logs balanced losses under `train/*` and `val/*`.
- Raw magnitudes are under `train_raw/*` and `val_raw/*`.

The coarse-class raw loss may be much smaller than OBB/mask loss. Judge task
balance from the non-raw metrics. The geometry head is single-class; mAP is
therefore class-agnostic object-detection mAP. It does not measure coarse-class
accuracy.

## Validation metrics

The trainer now calculates during every validation epoch:

- `epoch/val_map50`: rotated OBB mAP at IoU 0.50,
- `epoch/val_map50_95`: rotated OBB mAP averaged over 0.50:0.95,
- `epoch/val_mask_iou`: foreground mask micro-IoU at sigmoid threshold 0.5,
- validation loss and per-task components.

Rotated decoding, rotated NMS, probabilistic rotated IoU, and AP matching use
the installed Ultralytics implementation.

For mask IoU, intersection and union are accumulated only over validation
images whose labels explicitly provide masks.

## Checkpoints

Current trainer behavior:

- `last.pt`: most recently completed epoch,
- `best.pt`: best validation mAP@50-95,
- `best_iou.pt`: best validation mask IoU (added for seed 2).

Early stopping is disabled (`early_stopping_patience: 0`). Every current run
uses exactly 30 epochs.

The original/unseeded matrix predates online mAP. It saved only `best.pt`
(selected by validation loss) and `last.pt`; no per-epoch files or W&B model
artifacts exist. Its true historical best-mAP epoch cannot be recovered. The
retrospective comparison evaluates the better mAP of those two surviving
checkpoints only.

Seed 1 has genuine mAP-selected `best.pt` checkpoints. Seed 2 has genuine
mAP-selected `best.pt` and IoU-selected `best_iou.pt` checkpoints.

## Experiment status

### Original/unseeded DOTA + Potsdam matrix

- Complete: 28 logical conditions.
- Retrospective mAP for `best.pt` and `last.pt` is complete.
- Results: `/output/yolo_mask_effect/first_run_checkpoint_map.csv`.

### Explicit seed 1 DOTA + Potsdam matrix

- Complete: 28 conditions, 30 epochs per condition.
- True mAP-selected results:
  `/output/yolo_mask_effect/seed1_true_map_selected_results.csv`.
- Exact values used in the labeled plots:
  `/output/yolo_mask_effect/seed1_true_map_selected_plot_values.csv`.

### Explicit seed 2 Potsdam-only matrix

- Complete: all 14 conditions reached epoch 30.
- Seed: 2.
- Online mAP and mask IoU were calculated every epoch.
- Checkpoint folders end in `_seed2`.
- Scheduler log: `/output/yolo_mask_effect/matrix_seed2.log`.
- This seed has not yet been merged into the summary CSVs or plots.

The next recommended task is to extract `best.pt` mAP and `best_iou.pt` IoU
from every `_seed2` folder, decide whether the paper table should pair metrics
at the mAP-selected checkpoint or report independently selected best metrics,
then regenerate Potsdam mean/SD over the desired runs.

## Existing result tables

- First-run surviving checkpoint mAP:
  `/output/yolo_mask_effect/first_run_checkpoint_map.csv`
- Two-run mAP mean/SD:
  `/output/yolo_mask_effect/best_map_mean_std_two_runs.csv`
- Two-run combined mAP and mask-IoU mean/SD:
  `/output/yolo_mask_effect/best_metrics_mean_std_two_runs.csv`
- Per-checkpoint mask IoU:
  `/output/yolo_mask_effect/mask_iou_selected_checkpoints.csv`
- Seed-1 true mAP-selected results:
  `/output/yolo_mask_effect/seed1_true_map_selected_results.csv`

The two-run tables combine the original/unseeded run and seed 1. For the
original run, they use the better mAP of the two surviving checkpoints. Treat
that limitation explicitly in any paper text.

## Existing figures

Combined two-run figure:

- `/output/yolo_mask_effect/metrics_by_cgc_and_mask_ratio.png`
- `/output/yolo_mask_effect/metrics_by_cgc_and_mask_ratio.pdf`

Final labeled seed-1-only figures (percent scale, one decimal, grouped by CGC):

- `/output/yolo_mask_effect/dota_classification_seed1_percent_labeled.{png,pdf}`
- `/output/yolo_mask_effect/dota_mask_seed1_percent_labeled.{png,pdf}`
- `/output/yolo_mask_effect/potsdam_classification_seed1_percent_labeled.{png,pdf}`
- `/output/yolo_mask_effect/potsdam_mask_seed1_percent_labeled.{png,pdf}`

The files named `classification` plot detection mAP@50-95, not coarse-class
accuracy. Rename them if used in a publication to avoid ambiguity.

## W&B

- Project: `satellitepy-mask-effect`
- URL: `https://forge.coreweave.com/wandb/satellitepy/satellitepy-mask-effect`
- Groups: `dota` and `potsdam`
- Run suffixes distinguish explicit seeds: `_seed1`, `_seed2`.

The API key is stored locally in `docs/wandb_api.txt`. Never display it or add
it to Git. It is ignored in `.gitignore` and has mode 600. The scheduler reads
it in memory through `--wandb-key-file docs/wandb_api.txt`.

## Resume command for seed 2 (only if needed)

The seed-2 matrix is complete, so normally this command will skip all runs. It
is safe for recovering an incomplete/missing checkpoint:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/satellitepy \
python tools/train/run_mask_effect_matrix.py \
  --root /output/yolo_mask_effect \
  --patch-root /output/yolo_mtl_patches \
  --config configs/yolo_mtl/yolo11s_mask_effect_seed2.yaml \
  --domains potsdam \
  --epochs 30 \
  --max-parallel 4 \
  --seed 2 \
  --run-suffix _seed2 \
  --wandb-project satellitepy-mask-effect \
  --wandb-key-file docs/wandb_api.txt \
  >> /output/yolo_mask_effect/matrix_seed2.log 2>&1
```

## Known caveats

1. mAP is single-class, class-agnostic OBB detection AP. Coarse-class accuracy,
   macro-F1, and class-aware AP are not yet implemented.
2. The original run's true best-mAP checkpoint is irrecoverable.
3. IoU is foreground micro-IoU at threshold 0.5. Sparse-mask conditions often
   produce near-zero IoU; 50-100% mask conditions are substantially better.
4. Two-run standard deviations use sample SD (`ddof=1`). Seed 2 should be
   included before making final Potsdam statistical claims.
5. The worktree contains many user/project changes. Do not reset or discard
   unrelated files.
6. No secrets should appear in commands, logs, commits, or reports.

## Suggested next steps

1. Aggregate Potsdam seed 2 and regenerate the Potsdam tables/plots.
2. Decide metric pairing:
   - evaluate mAP and IoU at `best.pt` for a single jointly selected model, or
   - report best mAP from `best.pt` and best IoU from `best_iou.pt` separately.
3. Add coarse-class accuracy and macro-F1 during validation if semantic-class
   performance is required.
4. Save a manifest containing seed, checkpoint path, selected epoch, metrics,
   and Git revision for final thesis reproducibility.

