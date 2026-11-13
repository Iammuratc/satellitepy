#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/satellitepy
fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
root=/output/yolo11_baselines/fineair_rareplanes
log="$root/paper_seed0_queue.log"

mkdir -p "$root"
cd "$repo" || exit 1
read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.

record() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"
}

run_job() {
    local label="$1" config="$2" output="$3" group="$4" run_name="$5" run_id="$6" notes="$7"
    local attempt=1
    mkdir -p "$output"
    if [[ -f "$output/completed.ok" ]]; then
        record "$label already completed; skipping"
        return 0
    fi
    while (( attempt <= 3 )); do
        local resume_args=()
        if [[ -f "$output/last.pt" ]]; then
            resume_args=(--resume "$output/last.pt")
            record "$label attempt $attempt resuming from last.pt"
        else
            record "$label attempt $attempt starting from pretrained weights"
        fi
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py \
            --config "$config" \
            --train-folders "$fineair/train" "$rareplanes/train" \
            --val-folders "$fineair/val" "$rareplanes/val" \
            --out-folder "$output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group "$group" --wandb-name "$run_name" --wandb-id "$run_id" \
            --wandb-notes "$notes" \
            "${resume_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "$label completed successfully"
            PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
            return 0
        fi
        record "$label attempt $attempt failed"
        attempt=$((attempt + 1))
        sleep 30
    done
    record "$label exhausted three attempts; continuing queue"
    return 1
}

record "paper seed-0 queue started; gated natural run already completed"
run_job joint-corrected configs/yolo_mtl/yolo11s_fineair_rareplanes_joint.yaml \
    "$root/joint_corrected/seed0" fineair-rareplanes-paper-controls-v1 \
    fineair-rareplanes-joint-corrected-yolo11s-seed0 fineair-rareplanes-y11s-joint-corrected-s0 \
    "FineAir+RarePlanes joint YOLO11s OBB rerun. Correct full DDP-gathered per-dataset validation; early stopping on unweighted mean FineAir/RarePlanes mAP50-95. Natural 2046:4652 patch distribution; RGB+NIR/zero-IR; NIR dropout=0.25; seed=0."
run_job strict-detached configs/yolo_mtl/yolo11s_fineair_rareplanes_strict_detached.yaml \
    "$root/strict_detached/seed0" fineair-rareplanes-paper-controls-v1 \
    fineair-rareplanes-strict-detached-yolo11s-seed0 fineair-rareplanes-y11s-strict-det-s0 \
    "FineAir+RarePlanes strict detached control. Semantic features detached; detector and semantic heads use disjoint AdamW optimizers and independent AMP scalers, preventing semantic overflow from skipping detector steps. Full per-dataset validation; stop on mean mAP50-95; natural sampling; seed=0."
run_job gated-balanced configs/yolo_mtl/yolo11s_fineair_rareplanes_gated_balanced.yaml \
    "$root/gated_balanced/seed0" fineair-rareplanes-paper-controls-v1 \
    fineair-rareplanes-gated-balanced-yolo11s-seed0 fineair-rareplanes-y11s-gated-balanced-s0 \
    "FineAir+RarePlanes learned gated residual adapters with equal-probability dataset sampling (50% FineAir, 50% RarePlanes by expected samples). Full per-dataset validation; stop on mean mAP50-95; NIR dropout=0.25; seed=0."
record "paper seed-0 queue finished"
