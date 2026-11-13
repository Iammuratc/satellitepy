#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

worker="${1:?worker index required}"
gpu="${2:?GPU index required}"
data=/output/aircraft_semantic_chips/rgb_256_chip_split_seed42
root=/output/aircraft_chip_mtl
name="dota_mask_independent_early_seed${worker}"
output="$root/$name"
log="$root/matched_dota_early_worker_${worker}.log"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$log"; }
refresh_table() {
    flock /tmp/aircraft_chip_results.lock \
        python tools/evaluate/update_aircraft_chip_results.py >>"$log" 2>&1 || true
}

mkdir -p "$output"
if [[ -s "$output/test_metrics.json" && -s "$output/best.pt" ]]; then
    touch "$output/completed.ok"
    record "$name already complete"
    refresh_table
    exit 0
fi

for attempt in 1 2 3; do
    record "$name gpu=$gpu attempt=$attempt started"
    if CUDA_VISIBLE_DEVICES="$gpu" python tools/train/train_aircraft_chip_mtl.py \
        --data "$data" --output "$output" --datasets dota \
        --mask-route early --mask-weight 1.0 \
        --epochs 50 --batch-size 64 --workers 4 --patience 8 --seed "$worker" \
        --wandb-project aircraft-chip-mtl \
        --wandb-group chip-mtl-matched-dota-control-v1 \
        --run-name "$name" \
        --wandb-notes "Matched independent DOTA control using the same early mask decoder as the best shared model; seed=$worker." \
        >>"$log" 2>&1; then
        touch "$output/completed.ok"
        record "$name completed"
        refresh_table
        exit 0
    fi
    record "$name attempt=$attempt failed; retrying from last.pt"
    sleep 20
done

record "$name exhausted retries"
exit 1
