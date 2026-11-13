#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_aircraft/rgb
root=/output/yolo11_baselines
output="$root/dota_aircraft_controls/dota_late_backbone_fpn_masks/seed0"
config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_late_backbone_fpn_masks.yaml
log="$root/dota_late_backbone_fpn_masks_seed0.log"
mkdir -p "$output" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

train_run() {
    local attempt resume_args
    [[ -f "$output/completed.ok" ]] && { record "training already completed"; return 0; }
    for attempt in 1 2 3; do
        if [[ -s "$output/last.pt" ]]; then
            resume_args=(--resume "$output/last.pt")
        else
            resume_args=()
        fi
        record "late-backbone-fpn-masks attempt $attempt"
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py --config "$config" \
            --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
            --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" \
            --out-folder "$output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group domain-routing-controls-seed0-v1 \
            --wandb-name aircraft-dota-aircraft-late-backbone-fpn-masks-seed0 \
            --wandb-id aircraft-dota-aircraft-y11s-late-backbone-fpn-masks-s0 \
            --wandb-notes "Mask-auxiliary ablation of the late-backbone/FPN route. YOLO11 modules 0-7 and one aircraft OBB head are shared; DOTA has modules 8-22 plus a DOTA-only foreground-mask head trained from iSAID polygons; mask gradients update the detector route; other semantics detached; stop on three-dataset macro mAP50-95; seed=0." \
            "${resume_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "training completed"
            return 0
        fi
        record "attempt $attempt failed"
        sleep 30
    done
    return 1
}

evaluate_one() {
    local name="$1" split="$2"
    local result="$root/evaluations/aircraft_dota_aircraft_late_backbone_fpn_masks_${name}_test.json"
    [[ -s "$result" ]] && return 0
    CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$output/best.pt" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1
}

train_run || exit 1
evaluate_one fineair "$fineair/test" || exit 1
evaluate_one rareplanes "$rareplanes/test" || exit 1
evaluate_one dota_aircraft "$dota/test" || exit 1
CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo_mask.py \
    --config "$config" --checkpoint "$output/best.pt" --split "$dota/val" \
    --output "$root/evaluations/aircraft_dota_aircraft_late_backbone_fpn_masks_dota_aircraft_val_mask.json" \
    --device cuda:0 >>"$log" 2>&1 || exit 1
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
record "late-backbone/FPN mask-auxiliary seed-0 experiment finished"
