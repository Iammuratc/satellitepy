#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/satellitepy
fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
output=/output/yolo11_baselines/fineair_rareplanes
log="$output/resume_controls.log"

cd "$repo" || exit 1
read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.

record() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"
}

run_control() {
    local variant="$1" group="$2" run_name="$3" run_id="$4" notes="$5"
    local run_output="$output/$variant/seed0"
    local attempt=1
    while (( attempt <= 3 )); do
        record "$variant resume attempt $attempt"
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py \
            --config "configs/yolo_mtl/yolo11s_fineair_rareplanes_${variant}.yaml" \
            --train-folders "$fineair/train" "$rareplanes/train" \
            --val-folders "$fineair/val" "$rareplanes/val" \
            --out-folder "$run_output" --resume "$run_output/last.pt" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group "$group" --wandb-name "$run_name" --wandb-id "$run_id" \
            --wandb-notes "$notes" >>"$log" 2>&1; then
            record "$variant completed successfully"
            return 0
        fi
        record "$variant resume attempt $attempt failed"
        attempt=$((attempt + 1))
        sleep 30
    done
    record "$variant exhausted three resume attempts"
    return 1
}

record "control recovery queue started"
run_control detached fineair-rareplanes-detached-v1 \
    fineair-rareplanes-detached-yolo11s-seed0 fineair-rareplanes-y11s-det-s0 \
    "FineAir RGB+NIR and RarePlanes RGB (zero IR), YOLO11s OBB, shared backbone/FPN/detection head. Semantic heads use detached features. Per-dataset full-split mAP; early stopping on unweighted mean mAP50-95. NIR dropout=0.25; seed=0."
run_control detection_only fineair-rareplanes-detection-only-v1 \
    fineair-rareplanes-detection-only-yolo11s-seed0 fineair-rareplanes-y11s-detonly-s0 \
    "FineAir RGB+NIR and RarePlanes RGB (zero IR), YOLO11s OBB detection-only shared-backbone control. Per-dataset full-split mAP; early stopping on unweighted mean mAP50-95. NIR dropout=0.25; seed=0."
record "control recovery queue finished"
