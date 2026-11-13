#!/usr/bin/env bash
set -uo pipefail

cd /workspace/satellitepy || exit 1
data=/output/yolo_mtl_patches/dota_ships/rgb
root=/output/yolo11_baselines
output="$root/dota_ships/detection_only/seed0"
log="$root/dota_ships/seed0_queue.log"
mkdir -p "$output" "$root/evaluations"
read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.

record() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"
}

attempt=1
while (( attempt <= 3 )); do
    resume_args=()
    if [[ -f "$output/last.pt" ]]; then
        resume_args=(--resume "$output/last.pt")
        record "DOTA-ships detection-only attempt $attempt resuming"
    else
        record "DOTA-ships detection-only attempt $attempt starting"
    fi
    if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        tools/train/yolo_mtl.py \
        --config configs/yolo_mtl/yolo11s_dota_ships_detection_only.yaml \
        --train-folders "$data/train" --val-folders "$data/val" \
        --out-folder "$output" --seed 0 \
        --wandb-project yolo11-multidataset-baselines \
        --wandb-group aircraft-ships-paper-seed0-v1 \
        --wandb-name dota-ships-detection-only-yolo11s-seed0 \
        --wandb-id dota-ships-y11s-detonly-s0 \
        --wandb-notes "Independent DOTA-v2 ship-only YOLO11s OBB baseline. 2063/239/224 ship-containing train/val/test patches; 3-channel RGB; non-ship pixels are valid negatives for the binary ship head; early stopping on ship mAP50-95; seed=0." \
        "${resume_args[@]}" >>"$log" 2>&1; then
        touch "$output/completed.ok"
        record "DOTA-ships detection-only training completed"
        break
    fi
    record "DOTA-ships detection-only attempt $attempt failed"
    attempt=$((attempt + 1))
    sleep 30
done

if [[ ! -f "$output/completed.ok" ]]; then
    record "DOTA-ships detection-only exhausted three attempts"
    exit 1
fi

evaluation="$root/evaluations/dota_ships_detection_only_seed0_test.json"
if [[ ! -s "$evaluation" ]]; then
    CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config configs/yolo_mtl/yolo11s_dota_ships_detection_only.yaml \
        --checkpoint "$output/best.pt" --split "$data/test" \
        --output "$evaluation" --device cuda:0 >>"$log" 2>&1
fi
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
record "DOTA-ships independent seed-0 queue finished"
