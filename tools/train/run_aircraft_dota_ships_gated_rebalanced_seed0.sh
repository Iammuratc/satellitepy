#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_ships/rgb
root=/output/yolo11_baselines
base="$root/aircraft_dota_ships"
source_checkpoint="$base/conditional_detection_gated/seed0/best.pt"
output="$base/conditional_detection_gated_rebalanced/seed0"
config=configs/yolo_mtl/yolo11s_aircraft_dota_ships_gated_rebalanced.yaml
log="$output/run.log"
mkdir -p "$output" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

[[ -s "$source_checkpoint" ]] || { record "missing source checkpoint: $source_checkpoint"; exit 1; }

if [[ ! -f "$output/completed.ok" ]]; then
    for attempt in 1 2 3; do
        init_args=(--init-weights "$source_checkpoint")
        if [[ -s "$output/last.pt" ]]; then
            init_args=(--resume "$output/last.pt")
        fi
        record "warm-started 25/25/50 rebalancing attempt $attempt"
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py --config "$config" \
            --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
            --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" \
            --out-folder "$output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group aircraft-ships-paper-seed0-v1 \
            --wandb-name aircraft-dota-ships-detection-gated-rebalanced-seed0 \
            --wandb-id aircraft-dota-ships-y11s-detection-gated-rebalanced-s0 \
            --wandb-notes "Warm-started from the seed-0 detection-gated best checkpoint. Fresh optimizer/scalers; LR=5e-5; 25% FineAir, 25% RarePlanes, 50% DOTA-ships sampling; detached semantics; stop on three-dataset mean mAP50-95." \
            "${init_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "rebalanced training completed"
            break
        fi
        record "attempt $attempt failed"
        sleep 30
    done
fi

[[ -f "$output/completed.ok" ]] || exit 1
for spec in "fineair:$fineair/test" "rareplanes:$rareplanes/test" "dota_ships:$dota/test"; do
    name=${spec%%:*}; split=${spec#*:}
    result="$root/evaluations/aircraft_dota_ships_detection_gated_rebalanced_${name}_test.json"
    [[ -s "$result" ]] || CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$output/best.pt" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1
done
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
record "rebalanced evaluation and spreadsheet update completed"
