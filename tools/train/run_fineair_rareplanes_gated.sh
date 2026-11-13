#!/usr/bin/env bash
set -uo pipefail

cd /workspace/satellitepy || exit 1
output=/output/yolo11_baselines/fineair_rareplanes/gated/seed0
log=/output/yolo11_baselines/fineair_rareplanes/gated_queue.log
mkdir -p "$output"
read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.

for attempt in 1 2 3; do
    resume=()
    if [[ -f "$output/last.pt" ]]; then
        resume=(--resume "$output/last.pt")
    fi
    printf '%s gated attempt %d\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$attempt" >>"$log"
    if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        tools/train/yolo_mtl.py \
        --config configs/yolo_mtl/yolo11s_fineair_rareplanes_gated.yaml \
        --train-folders /output/yolo_mtl_patches/fineair/rgbir/train \
                        /output/yolo_mtl_patches/rareplanes/rgb/train \
        --val-folders /output/yolo_mtl_patches/fineair/rgbir/val \
                      /output/yolo_mtl_patches/rareplanes/rgb/val \
        --out-folder "$output" --seed 0 \
        --wandb-project yolo11-multidataset-baselines \
        --wandb-group fineair-rareplanes-learned-gates-v1 \
        --wandb-name fineair-rareplanes-gated-yolo11s-seed0 \
        --wandb-id fineair-rareplanes-y11s-gated-s0 \
        --wandb-notes "FineAir+RarePlanes YOLO11s OBB. Detached semantic base with task-specific P3/P4/P5 residual adapters and learned gates; gate-open penalty=0.01, initial gate=0.1. Full per-dataset mAP; early stopping on unweighted mean mAP50-95. NIR dropout=0.25; seed=0." \
        "${resume[@]}" >>"$log" 2>&1; then
        printf '%s gated completed successfully\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"$log"
        PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
        exit 0
    fi
    printf '%s gated attempt %d failed\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$attempt" >>"$log"
    sleep 30
done
printf '%s gated exhausted three attempts\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"$log"
exit 1
