#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

data=/output/aircraft_semantic_chips/rgb_256_chip_split_seed42
root=/output/aircraft_chip_mtl
log="$root/queue.log"
mkdir -p "$root"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.

record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$log"; }

while [[ ! -s "$data/manifest.json" ]]; do
    record "waiting for completed chip dataset manifest"
    sleep 30
done

run_job() {
    local gpu="$1" name="$2" datasets="$3" mask_route="$4" notes="$5"
    local output="$root/$name"
    mkdir -p "$output"
    if [[ -s "$output/test_metrics.json" && -s "$output/best.pt" ]]; then
        record "$name already complete"
        return 0
    fi
    for attempt in 1 2 3; do
        record "$name gpu=$gpu attempt=$attempt started"
        if CUDA_VISIBLE_DEVICES="$gpu" python tools/train/train_aircraft_chip_mtl.py \
            --data "$data" --output "$output" --datasets $datasets \
            --mask-route "$mask_route" --epochs 50 --batch-size 64 \
            --workers 8 --patience 8 --seed 0 \
            --wandb-project aircraft-chip-mtl \
            --wandb-group chip-mtl-seed0-v1 --run-name "$name" \
            --wandb-notes "$notes" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "$name completed"
            return 0
        fi
        record "$name attempt=$attempt failed; retrying"
        sleep 30
    done
    record "$name exhausted retries; worker continues"
    return 0
}

worker_0() {
    run_job 0 fineair_independent "fineair" none \
        "Independent FineAir RGB chip classifier; 256px target-oriented chips; oracle oriented-box prompt; chip-specific scene split; seed 0."
    run_job 0 shared_conditional_semantics "fineair rareplanes" none \
        "Shared ResNet18 encoder with masked dataset/task-specific FineAir and RarePlanes semantic heads; balanced dataset sampling; no DOTA mask task; seed 0."
}

worker_1() {
    run_job 1 rareplanes_independent "rareplanes" none \
        "Independent RarePlanes RGB chip attribute classifier; 256px target-oriented chips; oracle oriented-box prompt; chip-specific scene split; seed 0."
    run_job 1 shared_full_mask "fineair rareplanes dota" full \
        "Shared ResNet18 encoder with conditional FineAir/RarePlanes heads and DOTA silhouette decoder from final features; balanced dataset sampling; seed 0."
}

worker_2() {
    run_job 2 dota_mask_independent "dota" full \
        "Independent DOTA aircraft silhouette baseline using real aligned masks; held-out scene-level mask test split; seed 0."
    run_job 2 shared_early_mask "fineair rareplanes dota" early \
        "Shared ResNet18 semantic encoder with DOTA mask decoder attached at early/mid features, limiting mask gradients to lower layers; balanced dataset sampling; seed 0."
}

record "aircraft chip MTL queue started"
worker_0 & pid0=$!
worker_1 & pid1=$!
worker_2 & pid2=$!
wait "$pid0" "$pid1" "$pid2"
touch "$root/queue.completed.ok"
record "aircraft chip MTL queue completed"
