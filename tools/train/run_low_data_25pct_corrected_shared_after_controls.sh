#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

controls=/output/yolo11_baselines/low_data_25pct/seed0/queue.completed.ok
root=/output/yolo11_baselines/low_data_25pct/seed0
subset=/output/yolo_mtl_patches/low_data_25pct_seed0_corrected
fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_aircraft/rgb
output="$root/shared_strict_detached_corrected"
log="$root/corrected_shared_queue.log"

# The extra modality directory is intentional: yolo_mtl.py identifies a
# dataset using train_folder.parents[1].name.
python tools/data/create_low_data_subsets.py --dataset fineair \
    --source "$fineair/train" --destination "$subset/fineair/rgbir/train" \
    --fraction 0.25 --seed 0 >>"$log" 2>&1
python tools/data/create_low_data_subsets.py --dataset rareplanes \
    --source "$rareplanes/train" --destination "$subset/rareplanes/rgb/train" \
    --fraction 0.25 --seed 0 >>"$log" 2>&1
python tools/data/create_low_data_subsets.py --dataset dota_aircraft \
    --source "$dota/train" --destination "$subset/dota_aircraft/rgb/train" \
    --fraction 0.25 --seed 0 >>"$log" 2>&1

while [[ ! -f "$controls" ]]; do sleep 30; done

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
mkdir -p "$output"
for attempt in 1 2 3; do
    resume=()
    [[ -s "$output/last.pt" ]] && resume=(--resume "$output/last.pt")
    printf '%s corrected shared 25%% attempt=%s\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$attempt" | tee -a "$log"
    if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        tools/train/yolo_mtl.py \
        --config configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_strict_detached.yaml \
        --train-folders "$subset/fineair/rgbir/train" "$subset/rareplanes/rgb/train" "$subset/dota_aircraft/rgb/train" \
        --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" \
        --out-folder "$output" --seed 0 \
        --wandb-project yolo11-multidataset-baselines \
        --wandb-group low-data-25pct-seed0-v1 \
        --wandb-name lowdata25-shared-strict-detached-corrected-s0 \
        --wandb-id lowdata25-shared-strict-det-corrected-s0 \
        --wandb-notes "Corrected low-data study: strict-detached YOLO11s trained simultaneously on deterministic stratified 25% subsets. Dataset-preserving path layout restores aircraft-family geometry routing; equal dataset sampling; full unchanged validation/test; seed=0. The earlier lowdata25-shared-strict-detached-s0 run is invalid due to domain-path misrouting." \
        "${resume[@]}" >>"$log" 2>&1; then
        touch "$output/completed.ok"
        break
    fi
    if [[ -s "$output/best.pt" && -s "$output/last.pt" ]] \
        && tail -n 2500 "$log" | grep -q 'early stopping at epoch='; then
        touch "$output/completed.ok"
        break
    fi
    sleep 30
done
