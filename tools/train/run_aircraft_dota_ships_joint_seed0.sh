#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1
fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_ships/rgb
root=/output/yolo11_baselines
output="$root/aircraft_dota_ships/conditional_joint/seed0"
log="$root/aircraft_dota_ships/joint_seed0_queue.log"
config=configs/yolo_mtl/yolo11s_aircraft_dota_ships_joint.yaml
mkdir -p "$output" "$root/evaluations"
read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }
for attempt in 1 2 3; do
    resume_args=(); [[ -f "$output/last.pt" ]] && resume_args=(--resume "$output/last.pt")
    record "conditional joint attempt $attempt"
    if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        tools/train/yolo_mtl.py --config "$config" \
        --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
        --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" \
        --out-folder "$output" --seed 0 \
        --wandb-project yolo11-multidataset-baselines \
        --wandb-group aircraft-ships-paper-seed0-v1 \
        --wandb-name aircraft-dota-ships-conditional-joint-seed0 \
        --wandb-id aircraft-dota-ships-y11s-conditional-joint-s0 \
        --wandb-notes "FineAir+RarePlanes+DOTA-ships conditional joint YOLO11s. Separate aircraft/ship OBB heads with annotation-scope routing; all available sparse semantic losses update shared features. Equal dataset sampling; stop on three-dataset mean mAP50-95; seed=0." \
        "${resume_args[@]}" >>"$log" 2>&1; then touch "$output/completed.ok"; break; fi
    record "conditional joint attempt $attempt failed"; sleep 30
done
[[ -f "$output/completed.ok" ]] || exit 1
for spec in "fineair:$fineair/test" "rareplanes:$rareplanes/test" "dota_ships:$dota/test"; do
    name=${spec%%:*}; split=${spec#*:}; result="$root/evaluations/aircraft_dota_ships_joint_${name}_test.json"
    [[ -s "$result" ]] || CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$output/best.pt" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1
done
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
record "conditional joint seed-0 queue finished"
