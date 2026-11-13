#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_aircraft/rgb
root=/output/yolo11_baselines
base="$root/dota_aircraft_controls/conditional_heads/seed1"
log="$root/dota_aircraft_controls/conditional_heads_seed1.log"
config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_conditional_heads.yaml
mkdir -p "$base" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

for attempt in 1 2 3; do
    resume_args=()
    [[ -s "$base/last.pt" ]] && resume_args=(--resume "$base/last.pt")
    record "conditional-heads seed=1 attempt $attempt"
    if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        tools/train/yolo_mtl.py --config "$config" \
        --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
        --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" \
        --out-folder "$base" --seed 1 \
        --wandb-project yolo11-multidataset-baselines \
        --wandb-group conditional-aircraft-heads-seeds-v1 \
        --wandb-name aircraft-dota-conditional-heads-seed1 \
        --wandb-id aircraft-dota-aircraft-y11s-conditional-heads-s1 \
        --wandb-notes "Shared YOLO11 backbone/FPN with conditional aircraft OBB heads: FineAir+RarePlanes share one head and DOTA-aircraft uses a private head; strictly detached semantic heads; balanced sampling; native imagery; stop on macro per-dataset mAP50-95; seed=1." \
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
    result="$root/evaluations/aircraft_dota_aircraft_conditional_heads_seed1_${name}_test.json"
    [[ -s "$result" ]] && continue
    CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$base/best.pt" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1 || exit 1
done
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
touch "$root/dota_aircraft_controls/conditional_heads/seed1_queue.completed.ok"
record "conditional aircraft-head seed 1 completed"
