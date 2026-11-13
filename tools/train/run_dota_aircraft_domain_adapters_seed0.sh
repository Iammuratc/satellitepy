#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_aircraft/rgb
root=/output/yolo11_baselines
base="$root/dota_aircraft_controls"
current="$base/detection_gated/seed0"
output="$base/domain_adapters/seed0"
config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_domain_adapters.yaml
log="$base/domain_adapters_seed0.log"
mkdir -p "$output" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

record "waiting for detection-gated training and evaluations"
while [[ ! -f "$current/completed.ok" \
      || ! -s "$root/evaluations/aircraft_dota_aircraft_detection_gated_fineair_test.json" \
      || ! -s "$root/evaluations/aircraft_dota_aircraft_detection_gated_rareplanes_test.json" \
      || ! -s "$root/evaluations/aircraft_dota_aircraft_detection_gated_dota_aircraft_test.json" ]]; do
    sleep 30
done

if [[ ! -f "$output/completed.ok" ]]; then
    for attempt in 1 2 3; do
        resume_args=()
        [[ -s "$output/last.pt" ]] && resume_args=(--resume "$output/last.pt")
        record "domain-adapter attempt $attempt"
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py --config "$config" \
            --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
            --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" \
            --out-folder "$output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group dota-aircraft-domain-controls-seed0-v1 \
            --wandb-name aircraft-dota-aircraft-domain-adapters-seed0 \
            --wandb-id aircraft-dota-aircraft-y11s-domain-adapters-s0 \
            --wandb-notes "FineAir+RarePlanes+DOTA-aircraft domain-conditioned detector. Shared YOLO11 backbone/FPN and one aircraft OBB head; dataset-specific gated P3/P4/P5 residual adapters; semantics strictly detached; equal dataset sampling; stop on macro mAP50-95; seed=0." \
            "${resume_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "domain-adapter training completed"
            break
        fi
        record "domain-adapter attempt $attempt failed"
        sleep 30
    done
fi

[[ -f "$output/completed.ok" ]] || exit 1
for spec in "fineair:$fineair/test" "rareplanes:$rareplanes/test" "dota_aircraft:$dota/test"; do
    name=${spec%%:*}; split=${spec#*:}
    result="$root/evaluations/aircraft_dota_aircraft_domain_adapters_${name}_test.json"
    [[ -s "$result" ]] || CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$output/best.pt" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1
done
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
record "domain-adapter evaluation and spreadsheet update completed"
