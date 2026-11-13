#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_aircraft/rgb
root=/output/yolo11_baselines
base="$root/dota_aircraft_controls"
output="$base/dota_adapter/seed0"
config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_dota_adapter.yaml
log="$base/dota_adapter_seed0.log"
mkdir -p "$output" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

if [[ ! -f "$output/completed.ok" ]]; then
    for attempt in 1 2 3; do
        resume_args=()
        [[ -s "$output/last.pt" ]] && resume_args=(--resume "$output/last.pt")
        record "DOTA-only adapter attempt $attempt"
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py --config "$config" \
            --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
            --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" \
            --out-folder "$output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group dota-aircraft-domain-controls-seed0-v1 \
            --wandb-name aircraft-dota-aircraft-dota-adapter-seed0 \
            --wandb-id aircraft-dota-aircraft-y11s-dota-adapter-s0 \
            --wandb-notes "FineAir+RarePlanes+DOTA-aircraft DOTA-only adapter control. FineAir/RarePlanes retain the shared FPN identity path; only DOTA uses a gated P3/P4/P5 residual adapter; one shared aircraft OBB head; detached semantics; equal sampling; stop on macro mAP50-95; seed=0." \
            "${resume_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "DOTA-only adapter training completed"
            break
        fi
        record "DOTA-only adapter attempt $attempt failed"
        sleep 30
    done
fi

[[ -f "$output/completed.ok" ]] || exit 1
for spec in "fineair:$fineair/test" "rareplanes:$rareplanes/test" "dota_aircraft:$dota/test"; do
    name=${spec%%:*}; split=${spec#*:}
    result="$root/evaluations/aircraft_dota_aircraft_dota_adapter_${name}_test.json"
    [[ -s "$result" ]] || CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$output/best.pt" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1
done
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
record "DOTA-only adapter evaluation and spreadsheet update completed"
