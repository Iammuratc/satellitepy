#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_aircraft/rgb
root=/output/yolo11_baselines
base="$root/dota_aircraft_controls/dynamic_dota_adapter/seed0"
log="$root/dota_aircraft_controls/dynamic_dota_adapter_seed0.log"
config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_dynamic_adapter.yaml
mkdir -p "$base" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

for attempt in 1 2 3; do
    resume_args=()
    [[ -s "$base/last.pt" ]] && resume_args=(--resume "$base/last.pt")
    record "dynamic-dota-adapter seed=0 attempt $attempt"
    if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        tools/train/yolo_mtl.py --config "$config" \
        --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
        --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" \
        --out-folder "$base" --seed 0 \
        --wandb-project yolo11-multidataset-baselines \
        --wandb-group dynamic-dota-adapter-seed0-v1 \
        --wandb-name aircraft-dota-dynamic-adapter-seed0 \
        --wandb-id aircraft-dota-aircraft-y11s-dynamic-adapter-s0 \
        --wandb-notes "Shared YOLO11 backbone/FPN/head with a DOTA-only dynamic spatial-channel residual expert at P3/P4/P5: 1x1 bottleneck reduction, depthwise 3x3 correction, zero-initialized expansion, and per-image channel gates; bottleneck ratio=4; strictly detached semantics; balanced sampling; stop on macro mAP50-95; seed=0." \
        "${resume_args[@]}" >>"$log" 2>&1; then
        touch "$base/completed.ok"
        record "training completed"
        break
    fi
    record "attempt $attempt failed"
    sleep 30
done
[[ -f "$base/completed.ok" ]] || exit 1

for spec in "fineair:$fineair/test" "rareplanes:$rareplanes/test" "dota_aircraft:$dota/test"; do
    name=${spec%%:*}; split=${spec#*:}
    result="$root/evaluations/aircraft_dota_aircraft_dynamic_adapter_${name}_test.json"
    [[ -s "$result" ]] && continue
    CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$base/best.pt" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1 || exit 1
done
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
touch "$root/dota_aircraft_controls/dynamic_dota_adapter/queue.completed.ok"
record "dynamic DOTA adapter experiment completed"
