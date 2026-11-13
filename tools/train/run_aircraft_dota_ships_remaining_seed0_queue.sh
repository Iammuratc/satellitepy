#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1
fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_ships/rgb
root=/output/yolo11_baselines
base="$root/aircraft_dota_ships"
log="$base/remaining_seed0_queue.log"
mkdir -p "$base" "$root/evaluations"
read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

record "waiting for conditional joint training and evaluations"
while [[ ! -f "$base/conditional_joint/seed0/completed.ok" \
      || ! -s "$root/evaluations/aircraft_dota_ships_joint_fineair_test.json" \
      || ! -s "$root/evaluations/aircraft_dota_ships_joint_rareplanes_test.json" \
      || ! -s "$root/evaluations/aircraft_dota_ships_joint_dota_ships_test.json" ]]; do
    sleep 30
done
record "joint artifacts complete; starting successor runs"

run_variant() {
    local variant="$1" config="$2" run_id="$3" notes="$4"
    local output="$base/conditional_${variant}/seed0" attempt
    mkdir -p "$output"
    if [[ ! -f "$output/completed.ok" ]]; then
        for attempt in 1 2 3; do
            local resume_args=(); [[ -f "$output/last.pt" ]] && resume_args=(--resume "$output/last.pt")
            record "$variant attempt $attempt"
            if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
                tools/train/yolo_mtl.py --config "$config" \
                --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
                --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" \
                --out-folder "$output" --seed 0 \
                --wandb-project yolo11-multidataset-baselines \
                --wandb-group aircraft-ships-paper-seed0-v1 \
                --wandb-name "aircraft-dota-ships-conditional-${variant}-seed0" \
                --wandb-id "$run_id" --wandb-notes "$notes" \
                "${resume_args[@]}" >>"$log" 2>&1; then
                touch "$output/completed.ok"; record "$variant training completed"; break
            fi
            record "$variant attempt $attempt failed"; sleep 30
        done
    fi
    [[ -f "$output/completed.ok" ]] || return 1
    for spec in "fineair:$fineair/test" "rareplanes:$rareplanes/test" "dota_ships:$dota/test"; do
        local name=${spec%%:*} split=${spec#*:}
        local result="$root/evaluations/aircraft_dota_ships_${variant}_${name}_test.json"
        [[ -s "$result" ]] || CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
            --config "$config" --checkpoint "$output/best.pt" --split "$split" \
            --output "$result" --device cuda:0 >>"$log" 2>&1
    done
    PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
}

run_variant strict_detached configs/yolo_mtl/yolo11s_aircraft_dota_ships_strict_detached.yaml \
    aircraft-dota-ships-y11s-conditional-strict-det-s0 \
    "Conditional aircraft+ship strict-detached control. Separate OBB heads; semantic features detached; disjoint detector/semantic optimizers and AMP scalers. Equal three-dataset sampling; stop on macro mAP50-95; seed=0."
run_variant detection_gated configs/yolo_mtl/yolo11s_aircraft_dota_ships_gated.yaml \
    aircraft-dota-ships-y11s-conditional-detection-gated-s0 \
    "Conditional aircraft+ship detection-gated model. Semantic features remain strictly detached with isolated optimization. Aircraft/ship detection losses train family-specific P3/P4/P5 residual adapters and gates over the shared FPN. Equal three-dataset sampling; stop on macro mAP50-95; seed=0."
record "aircraft+ship remaining seed-0 queue finished"
