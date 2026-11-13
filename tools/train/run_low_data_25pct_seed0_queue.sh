#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

root=/output/yolo11_baselines/low_data_25pct/seed0
subset=/output/yolo_mtl_patches/low_data_25pct_seed0
fineair_full=/output/yolo_mtl_patches/fineair/rgbir
rareplanes_full=/output/yolo_mtl_patches/rareplanes/rgb
dota_full=/output/yolo_mtl_patches/dota_aircraft/rgb
log="$root/queue.log"
mkdir -p "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

make_subset() {
    local name="$1" source="$2"
    python tools/data/create_low_data_subsets.py --dataset "$name" \
        --source "$source/train" --destination "$subset/$name/train" \
        --fraction 0.25 --seed 0 >>"$log" 2>&1
}

make_subset fineair "$fineair_full"
make_subset rareplanes "$rareplanes_full"
make_subset dota_aircraft "$dota_full"

run_training() {
    local name="$1" config="$2" output="$3" run_id="$4" notes="$5"
    shift 5
    mkdir -p "$output"
    if [[ ! -f "$output/completed.ok" ]]; then
        for attempt in 1 2 3; do
            local resume=()
            [[ -s "$output/last.pt" ]] && resume=(--resume "$output/last.pt")
            record "$name attempt=$attempt"
            if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
                tools/train/yolo_mtl.py --config "$config" "$@" --out-folder "$output" \
                --seed 0 --wandb-project yolo11-multidataset-baselines \
                --wandb-group low-data-25pct-seed0-v1 --wandb-name "$name" --wandb-id "$run_id" \
                --wandb-notes "$notes" "${resume[@]}" >>"$log" 2>&1; then
                touch "$output/completed.ok"
                break
            fi
            if [[ -s "$output/best.pt" && -s "$output/last.pt" ]] \
                && tail -n 2500 "$log" | grep -q 'early stopping at epoch='; then
                touch "$output/completed.ok"
                record "$name accepted after NCCL teardown"
                break
            fi
            record "$name failed attempt=$attempt"
            sleep 30
        done
    fi
}

# Main hypothesis first: simultaneous balanced training on 25% of every dataset.
run_training lowdata25-shared-strict-detached-s0 \
    configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_strict_detached.yaml \
    "$root/shared_strict_detached" lowdata25-shared-strict-det-s0 \
    "Low-data study: shared strict-detached YOLO11s trained simultaneously on deterministic stratified 25% subsets of FineAir, RarePlanes, and DOTA-aircraft. Equal dataset sampling; full unchanged validation/test; seed=0." \
    --train-folders "$subset/fineair/train" "$subset/rareplanes/train" "$subset/dota_aircraft/train" \
    --val-folders "$fineair_full/val" "$rareplanes_full/val" "$dota_full/val"

run_training lowdata25-fineair-independent-s0 configs/yolo_mtl/yolo11s_fineair_detection_only.yaml \
    "$root/independent_fineair" lowdata25-fineair-independent-s0 \
    "Low-data matched baseline: independent FineAir YOLO11s trained on the identical deterministic stratified 25% subset; full unchanged validation/test; seed=0." \
    --train-folders "$subset/fineair/train" --val-folders "$fineair_full/val"

run_training lowdata25-rareplanes-independent-s0 configs/yolo_mtl/yolo11s_rareplanes_detection_only.yaml \
    "$root/independent_rareplanes" lowdata25-rareplanes-independent-s0 \
    "Low-data matched baseline: independent RarePlanes YOLO11s trained on the identical deterministic stratified 25% subset; full unchanged validation/test; seed=0." \
    --train-folders "$subset/rareplanes/train" --val-folders "$rareplanes_full/val"

run_training lowdata25-dota-independent-s0 configs/yolo_mtl/yolo11s_dota_aircraft_detection_only.yaml \
    "$root/independent_dota" lowdata25-dota-independent-s0 \
    "Low-data matched baseline: independent DOTA-aircraft YOLO11s trained on the identical deterministic aircraft-count-stratified 25% subset; full unchanged validation/test; seed=0." \
    --train-folders "$subset/dota_aircraft/train" --val-folders "$dota_full/val"

touch "$root/queue.completed.ok"
record "low-data 25% seed-0 training queue completed"
