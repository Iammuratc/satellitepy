#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/satellitepy
root=/output/yolo11_baselines
merged="$root/fineair_rareplanes"
fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
log="$merged/remaining_seed0_queue.log"

mkdir -p "$root/evaluations" "$merged"
cd "$repo" || exit 1
read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.

record() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"
}

evaluate_one() {
    local label="$1" config="$2" checkpoint="$3" split="$4" output="$5" gpu="$6"
    if [[ -s "$output" ]]; then
        record "$label evaluation already exists; skipping"
        return 0
    fi
    local attempt=1
    while (( attempt <= 3 )); do
        record "$label evaluation attempt $attempt"
        if CUDA_VISIBLE_DEVICES="$gpu" python tools/evaluate/evaluate_sparse_yolo.py \
            --config "$config" --checkpoint "$checkpoint" --split "$split" \
            --output "$output" --device cuda:0 >>"$log" 2>&1; then
            record "$label evaluation completed"
            return 0
        fi
        record "$label evaluation attempt $attempt failed"
        attempt=$((attempt + 1))
        sleep 15
    done
    record "$label evaluation exhausted three attempts; continuing"
    return 1
}

evaluate_pair() {
    local variant="$1" config="$2" checkpoint="$3" stem="$4"
    evaluate_one "$variant FineAir" "$config" "$checkpoint" "$fineair/test" \
        "$root/evaluations/${stem}_fineair_test.json" 0
    evaluate_one "$variant RarePlanes" "$config" "$checkpoint" "$rareplanes/test" \
        "$root/evaluations/${stem}_rareplanes_test.json" 0
    PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
}

train_balanced_detection() {
    local output="$merged/detection_only_balanced/seed0" attempt=1
    mkdir -p "$output"
    if [[ -f "$output/completed.ok" ]]; then
        record "balanced detection-only already completed; skipping training"
        return 0
    fi
    while (( attempt <= 3 )); do
        local resume_args=()
        if [[ -f "$output/last.pt" ]]; then
            resume_args=(--resume "$output/last.pt")
            record "balanced detection-only attempt $attempt resuming"
        else
            record "balanced detection-only attempt $attempt starting"
        fi
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py \
            --config configs/yolo_mtl/yolo11s_fineair_rareplanes_detection_only_balanced.yaml \
            --train-folders "$fineair/train" "$rareplanes/train" \
            --val-folders "$fineair/val" "$rareplanes/val" \
            --out-folder "$output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group fineair-rareplanes-paper-controls-v1 \
            --wandb-name fineair-rareplanes-detection-only-balanced-yolo11s-seed0 \
            --wandb-id fineair-rareplanes-y11s-detonly-balanced-s0 \
            --wandb-notes "FineAir+RarePlanes detection-only balanced-sampling control. Equal expected dataset probability (50%/50%); no semantic heads. Full per-dataset validation; early stopping on mean mAP50-95; NIR dropout=0.25; seed=0." \
            "${resume_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "balanced detection-only training completed"
            return 0
        fi
        record "balanced detection-only attempt $attempt failed"
        attempt=$((attempt + 1))
        sleep 30
    done
    record "balanced detection-only exhausted three attempts"
    return 1
}

record "remaining seed-0 queue started"
evaluate_pair gated configs/yolo_mtl/yolo11s_fineair_rareplanes_gated.yaml \
    "$merged/gated/seed0/best.pt" fineair_rareplanes_gated
evaluate_pair joint-corrected configs/yolo_mtl/yolo11s_fineair_rareplanes_joint.yaml \
    "$merged/joint_corrected/seed0/best.pt" fineair_rareplanes_joint_corrected
evaluate_pair strict-detached configs/yolo_mtl/yolo11s_fineair_rareplanes_strict_detached.yaml \
    "$merged/strict_detached/seed0/best.pt" fineair_rareplanes_strict_detached
evaluate_pair gated-balanced configs/yolo_mtl/yolo11s_fineair_rareplanes_gated_balanced.yaml \
    "$merged/gated_balanced/seed0/best.pt" fineair_rareplanes_gated_balanced

train_balanced_detection
evaluate_pair detection-only-balanced \
    configs/yolo_mtl/yolo11s_fineair_rareplanes_detection_only_balanced.yaml \
    "$merged/detection_only_balanced/seed0/best.pt" \
    fineair_rareplanes_detection_only_balanced
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
record "remaining seed-0 queue finished"
