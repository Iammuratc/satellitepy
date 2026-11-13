#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_aircraft/rgb
root=/output/yolo11_baselines
controls="$root/dota_aircraft_controls"
log="$root/dota_adapter_and_independent_masks_seed0_queue.log"
mkdir -p "$controls" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

train_run() {
    local label="$1" config="$2" output="$3" run_id="$4" run_name="$5" notes="$6"
    shift 6
    local attempt resume_args
    mkdir -p "$output"
    [[ -f "$output/completed.ok" ]] && { record "$label already completed"; return 0; }
    for attempt in 1 2 3; do
        resume_args=()
        [[ -s "$output/last.pt" ]] && resume_args=(--resume "$output/last.pt")
        record "$label attempt $attempt"
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py --config "$config" "$@" \
            --out-folder "$output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group mask-routing-controls-seed0-v1 \
            --wandb-name "$run_name" --wandb-id "$run_id" --wandb-notes "$notes" \
            "${resume_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "$label training completed"
            return 0
        fi
        record "$label attempt $attempt failed"
        sleep 30
    done
    return 1
}

evaluate_detection() {
    local config="$1" checkpoint="$2" split="$3" result="$4"
    [[ -s "$result" ]] && return 0
    CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$checkpoint" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1
}

evaluate_mask() {
    local config="$1" checkpoint="$2" result="$3"
    [[ -s "$result" ]] && return 0
    CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo_mask.py \
        --config "$config" --checkpoint "$checkpoint" --split "$dota/val" \
        --output "$result" --device cuda:0 >>"$log" 2>&1
}

adapter_config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_dota_adapter_masks_isolated.yaml
adapter_output="$controls/dota_adapter_masks_isolated/seed0"
train_run adapter-isolated-masks "$adapter_config" "$adapter_output" \
    aircraft-dota-aircraft-y11s-dota-adapter-masks-isolated-s0 \
    aircraft-dota-aircraft-dota-adapter-masks-isolated-seed0 \
    "Selected DOTA-only residual adapter with DOTA aircraft-mask auxiliary. Mask gradients update the DOTA adapter and mask head but stop before the shared backbone/FPN; OBB detection gradients remain unchanged; semantics detached; equal three-dataset sampling; stop on macro mAP50-95; seed=0." \
    --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
    --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" || exit 1
for spec in "fineair:$fineair/test" "rareplanes:$rareplanes/test" "dota_aircraft:$dota/test"; do
    name=${spec%%:*}; split=${spec#*:}
    evaluate_detection "$adapter_config" "$adapter_output/best.pt" "$split" \
        "$root/evaluations/aircraft_dota_aircraft_dota_adapter_masks_isolated_${name}_test.json" || exit 1
done
evaluate_mask "$adapter_config" "$adapter_output/best.pt" \
    "$root/evaluations/aircraft_dota_aircraft_dota_adapter_masks_isolated_dota_aircraft_val_mask.json" || exit 1

independent_config=configs/yolo_mtl/yolo11s_dota_aircraft_detection_masks.yaml
independent_output="$controls/independent_detection_masks/seed0"
train_run independent-detection-masks "$independent_config" "$independent_output" \
    dota-aircraft-y11s-detection-masks-s0 dota-aircraft-detection-masks-seed0 \
    "Independent DOTA-v2 aircraft YOLO11s OBB+foreground-mask baseline. RGB; joint detection and iSAID-derived mask gradients update the complete backbone/FPN; unmatched mask instances ignored within their OBBs; stop on DOTA mAP50-95; seed=0." \
    --train-folders "$dota/train" --val-folders "$dota/val" || exit 1
evaluate_detection "$independent_config" "$independent_output/best.pt" "$dota/test" \
    "$root/evaluations/dota_aircraft_detection_masks_seed0_test.json" || exit 1
evaluate_mask "$independent_config" "$independent_output/best.pt" \
    "$root/evaluations/dota_aircraft_detection_masks_seed0_val_mask.json" || exit 1

PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
record "DOTA adapter and independent mask seed-0 queue finished"
