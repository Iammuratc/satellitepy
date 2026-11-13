#!/usr/bin/env bash
set -uo pipefail

# Queue the remaining RarePlanes controls after the currently running joint
# baseline. This script is restart-aware and reuses stable W&B run IDs.

repo=/workspace/satellitepy
data=/output/yolo_mtl_patches/rareplanes/rgb
output=/output/yolo11_baselines/rareplanes
log="$output/queue.log"
joint_worker_pid="${RAREPLANES_JOINT_WORKER_PID:-188972}"

mkdir -p "$output"
cd "$repo" || exit 1

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

record() {
    printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$log"
}

while kill -0 "$joint_worker_pid" 2>/dev/null; do
    sleep 30
done
record "joint worker $joint_worker_pid exited; starting queued controls"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_ENTITY=satellitepy
export PYTHONPATH=.

run_baseline() {
    local variant="$1"
    local group="$2"
    local run_name="$3"
    local run_id="$4"
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
            --config "configs/yolo_mtl/yolo11s_rareplanes_${variant}.yaml" \
            --train-folders "$data/train" \
            --val-folders "$data/val" \
            --out-folder "$run_output" \
            --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group "$group" \
            --wandb-name "$run_name" \
            --wandb-id "$run_id" \
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

run_baseline \
    detached \
    rareplanes-detached-tasks-v1 \
    rareplanes-detached-yolo11s-seed0 \
    rareplanes-y11s-det-s0

run_baseline \
    detection_only \
    rareplanes-detection-only-v1 \
    rareplanes-detection-only-yolo11s-seed0 \
    rareplanes-y11s-detonly-s0

record "RarePlanes baseline queue finished"
