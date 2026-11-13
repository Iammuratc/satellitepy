#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_aircraft/rgb
root=/output/yolo11_baselines
output="$root/dota_aircraft_controls/pcgrad/seed0"
config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_pcgrad.yaml
log="$root/dota_aircraft_controls/pcgrad_seed0.log"
mkdir -p "$output" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

if [[ ! -f "$output/completed.ok" ]]; then
    for attempt in 1 2 3; do
        resume_args=()
        [[ -s "$output/last.pt" ]] && resume_args=(--resume "$output/last.pt")
        record "PCGrad seed=0 attempt $attempt"
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py --config "$config" \
            --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
            --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" \
            --out-folder "$output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group gradient-conflict-controls-v1 \
            --wandb-name aircraft-dota-aircraft-pcgrad-seed0 \
            --wandb-id aircraft-dota-aircraft-y11s-pcgrad-s0 \
            --wandb-notes "Strict-detached shared YOLO11 aircraft detector with balanced FineAir/RarePlanes/DOTA sampling. PCGrad projects conflicting per-dataset detection gradients before updating the shared backbone, FPN, and aircraft head; semantic heads remain isolated; global batch=4 for PCGrad memory safety; stop on three-dataset macro mAP50-95; seed=0." \
            "${resume_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "PCGrad training completed"
            break
        fi
        record "PCGrad attempt $attempt failed"
        sleep 30
    done
fi

[[ -f "$output/completed.ok" ]] || exit 1
for spec in "fineair:$fineair/test" "rareplanes:$rareplanes/test" "dota_aircraft:$dota/test"; do
    name=${spec%%:*}; split=${spec#*:}
    result="$root/evaluations/aircraft_dota_aircraft_pcgrad_${name}_test.json"
    [[ -s "$result" ]] || CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$output/best.pt" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1
done
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
record "PCGrad evaluation and spreadsheet update completed"
