#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/satellitepy
fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
output=/output/yolo11_baselines/fineair_rareplanes
log="$output/queue.log"

mkdir -p "$output"
cd "$repo" || exit 1
read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.

record() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"
}

run_baseline() {
    local variant="$1" group="$2" run_name="$3" run_id="$4" notes="$5"
    local run_output="$output/$variant/seed0"
    local attempt=1
    mkdir -p "$run_output"

    while (( attempt <= 3 )); do
        local resume_args=()
        if [[ -f "$run_output/last.pt" ]]; then
            resume_args=(--resume "$run_output/last.pt")
            record "$variant attempt $attempt resuming from last.pt"
        else
            record "$variant attempt $attempt starting from pretrained weights"
        fi
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py \
            --config "configs/yolo_mtl/yolo11s_fineair_rareplanes_${variant}.yaml" \
            --train-folders "$fineair/train" "$rareplanes/train" \
            --val-folders "$fineair/val" "$rareplanes/val" \
            --out-folder "$run_output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group "$group" --wandb-name "$run_name" --wandb-id "$run_id" \
            --wandb-notes "$notes" \
            "${resume_args[@]}" >>"$log" 2>&1; then
            record "$variant completed successfully"
            return 0
        fi
        record "$variant attempt $attempt failed"
        attempt=$((attempt + 1))
        sleep 30
    done
    record "$variant exhausted three attempts; continuing queue"
    return 1
}

record "FineAir-RarePlanes shared-backbone queue started"
run_baseline joint fineair-rareplanes-joint-v1 \
    fineair-rareplanes-joint-yolo11s-seed0 fineair-rareplanes-y11s-joint-s0 \
    "FineAir RGB+NIR and RarePlanes RGB (zero IR), YOLO11s OBB, one shared backbone/FPN/detection head. All sparse fine, very-fine, role and attribute losses backpropagate into shared features. NIR dropout=0.25; seed=0."
run_baseline detached fineair-rareplanes-detached-v1 \
    fineair-rareplanes-detached-yolo11s-seed0 fineair-rareplanes-y11s-det-s0 \
    "FineAir RGB+NIR and RarePlanes RGB (zero IR), YOLO11s OBB, one shared backbone/FPN/detection head. Sparse semantic heads train on detached features, so only detection updates shared features. NIR dropout=0.25; seed=0."
run_baseline detection_only fineair-rareplanes-detection-only-v1 \
    fineair-rareplanes-detection-only-yolo11s-seed0 fineair-rareplanes-y11s-detonly-s0 \
    "FineAir RGB+NIR and RarePlanes RGB (zero IR), YOLO11s OBB detection-only control with one shared backbone/FPN/detection head and no semantic heads. NIR dropout=0.25; seed=0."
record "FineAir-RarePlanes shared-backbone queue finished"
