#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

while [[ ! -f /output/yolo11_baselines/low_data_50pct/seed0/queue.completed.ok ]]; do sleep 30; done

root=/output/yolo11_baselines/low_data_75pct/seed0
subset=/output/yolo_mtl_patches/low_data_75pct_seed0
fa=/output/yolo_mtl_patches/fineair/rgbir
rp=/output/yolo_mtl_patches/rareplanes/rgb
da=/output/yolo_mtl_patches/dota_aircraft/rgb
log="$root/queue.log"
mkdir -p "$root"
read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

python tools/data/create_low_data_subsets.py --dataset fineair --source "$fa/train" \
    --destination "$subset/fineair/rgbir/train" --fraction .75 --seed 0 >>"$log" 2>&1
python tools/data/create_low_data_subsets.py --dataset rareplanes --source "$rp/train" \
    --destination "$subset/rareplanes/rgb/train" --fraction .75 --seed 0 >>"$log" 2>&1
python tools/data/create_low_data_subsets.py --dataset dota_aircraft --source "$da/train" \
    --destination "$subset/dota_aircraft/rgb/train" --fraction .75 --seed 0 >>"$log" 2>&1

run_one() {
    local name="$1" config="$2" output="$3" run_id="$4" notes="$5"; shift 5
    mkdir -p "$output"; [[ -f "$output/completed.ok" ]] && return 0
    for attempt in 1 2 3; do
        resume=(); [[ -s "$output/last.pt" ]] && resume=(--resume "$output/last.pt")
        record "$name attempt=$attempt"
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py --config "$config" "$@" --out-folder "$output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines --wandb-group low-data-75pct-seed0-v1 \
            --wandb-name "$name" --wandb-id "$run_id" --wandb-notes "$notes" \
            "${resume[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"; break
        fi
        if [[ -s "$output/best.pt" && -s "$output/last.pt" ]] \
            && tail -n 2500 "$log" | grep -q 'early stopping at epoch='; then
            touch "$output/completed.ok"; record "$name accepted after NCCL teardown"; break
        fi
        sleep 30
    done
}

run_one lowdata75-shared-strict-detached-s0 \
    configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_strict_detached.yaml \
    "$root/shared_strict_detached" lowdata75-shared-strict-det-s0 \
    "Low-data curve: shared strict-detached YOLO11s on deterministic stratified 75% subsets; verified domain routing; equal dataset sampling; full validation/test; seed=0." \
    --train-folders "$subset/fineair/rgbir/train" "$subset/rareplanes/rgb/train" "$subset/dota_aircraft/rgb/train" \
    --val-folders "$fa/val" "$rp/val" "$da/val"
run_one lowdata75-fineair-independent-s0 configs/yolo_mtl/yolo11s_fineair_detection_only.yaml \
    "$root/independent_fineair" lowdata75-fineair-independent-s0 \
    "Matched 75% FineAir independent baseline; full validation/test; seed=0." \
    --train-folders "$subset/fineair/rgbir/train" --val-folders "$fa/val"
run_one lowdata75-rareplanes-independent-s0 configs/yolo_mtl/yolo11s_rareplanes_detection_only.yaml \
    "$root/independent_rareplanes" lowdata75-rareplanes-independent-s0 \
    "Matched 75% RarePlanes independent baseline; full validation/test; seed=0." \
    --train-folders "$subset/rareplanes/rgb/train" --val-folders "$rp/val"
run_one lowdata75-dota-independent-s0 configs/yolo_mtl/yolo11s_dota_aircraft_detection_only.yaml \
    "$root/independent_dota" lowdata75-dota-independent-s0 \
    "Matched 75% DOTA-aircraft independent baseline; full validation/test; seed=0." \
    --train-folders "$subset/dota_aircraft/rgb/train" --val-folders "$da/val"

touch "$root/queue.completed.ok"
record "low-data 75% seed-0 queue completed"
