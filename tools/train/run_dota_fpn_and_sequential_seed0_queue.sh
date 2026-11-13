#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_aircraft/rgb
root=/output/yolo11_baselines
controls="$root/dota_aircraft_controls"
sequential="$root/sequential_transfer"
log="$root/dota_fpn_and_sequential_seed0_queue.log"
mkdir -p "$controls" "$sequential" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

train_run() {
    local label="$1" config="$2" output="$3" run_id="$4" run_name="$5" notes="$6" init="$7"
    shift 7
    local attempt resume_args
    mkdir -p "$output"
    [[ -f "$output/completed.ok" ]] && { record "$label already completed"; return 0; }
    for attempt in 1 2 3; do
        if [[ -s "$output/last.pt" ]]; then
            resume_args=(--resume "$output/last.pt")
        elif [[ -n "$init" ]]; then
            resume_args=(--init-weights "$init")
        else
            resume_args=()
        fi
        record "$label attempt $attempt"
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py --config "$config" "$@" \
            --out-folder "$output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group domain-and-sequential-controls-seed0-v1 \
            --wandb-name "$run_name" --wandb-id "$run_id" --wandb-notes "$notes" \
            "${resume_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"; record "$label training completed"; return 0
        fi
        record "$label attempt $attempt failed"; sleep 30
    done
    return 1
}

evaluate_one() {
    local config="$1" checkpoint="$2" split="$3" result="$4"
    [[ -s "$result" ]] && return 0
    CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$checkpoint" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1
}

fpn_config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_dota_fpn.yaml
fpn_output="$controls/dota_fpn/seed0"
train_run dota-fpn "$fpn_config" "$fpn_output" \
    aircraft-dota-aircraft-y11s-dota-fpn-s0 aircraft-dota-aircraft-dota-fpn-seed0 \
    "FineAir+RarePlanes+DOTA-aircraft partial-sharing control. YOLO11 backbone modules 0-10 and one aircraft head are shared; DOTA has a separate FPN/neck; semantics detached; equal dataset sampling; stop on macro mAP50-95; seed=0." "" \
    --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
    --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" || exit 1
for spec in "fineair:$fineair/test" "rareplanes:$rareplanes/test" "dota_aircraft:$dota/test"; do
    name=${spec%%:*}; split=${spec#*:}
    evaluate_one "$fpn_config" "$fpn_output/best.pt" "$split" \
        "$root/evaluations/aircraft_dota_aircraft_dota_fpn_${name}_test.json" || exit 1
done

rp_to_fa_config=configs/yolo_mtl/yolo11s_fineair_detection_only.yaml
rp_to_fa_output="$sequential/rareplanes_to_fineair/seed0"
train_run rareplanes-to-fineair "$rp_to_fa_config" "$rp_to_fa_output" \
    sequential-rareplanes-to-fineair-y11s-s0 sequential-rareplanes-to-fineair-seed0 \
    "Sequential transfer control: initialize from the independent RarePlanes detection-only best checkpoint, adapt RGB stem to RGB+IR, reset optimizer/scalers, then train only on FineAir. Evaluate final checkpoint on both datasets; seed=0." \
    "$root/rareplanes/detection_only/seed0/best.pt" \
    --train-folders "$fineair/train" --val-folders "$fineair/val" || exit 1
evaluate_one "$rp_to_fa_config" "$rp_to_fa_output/best.pt" "$fineair/test" \
    "$root/evaluations/sequential_rareplanes_to_fineair_fineair_test.json" || exit 1
evaluate_one "$rp_to_fa_config" "$rp_to_fa_output/best.pt" "$rareplanes/test" \
    "$root/evaluations/sequential_rareplanes_to_fineair_rareplanes_test.json" || exit 1

fa_to_rp_config=configs/yolo_mtl/yolo11s_rareplanes_detection_only.yaml
fa_to_rp_output="$sequential/fineair_to_rareplanes/seed0"
train_run fineair-to-rareplanes "$fa_to_rp_config" "$fa_to_rp_output" \
    sequential-fineair-to-rareplanes-y11s-s0 sequential-fineair-to-rareplanes-seed0 \
    "Sequential transfer control: initialize from the independent FineAir detection-only best checkpoint, drop the IR stem channel, reset optimizer/scalers, then train only on RarePlanes. Evaluate final checkpoint on both datasets; seed=0." \
    "$root/detection_only/fineair_seed0/best.pt" \
    --train-folders "$rareplanes/train" --val-folders "$rareplanes/val" || exit 1
evaluate_one "$fa_to_rp_config" "$fa_to_rp_output/best.pt" "$fineair/test" \
    "$root/evaluations/sequential_fineair_to_rareplanes_fineair_test.json" || exit 1
evaluate_one "$fa_to_rp_config" "$fa_to_rp_output/best.pt" "$rareplanes/test" \
    "$root/evaluations/sequential_fineair_to_rareplanes_rareplanes_test.json" || exit 1

PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
record "DOTA-FPN and sequential-transfer seed-0 queue finished"
