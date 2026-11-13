#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

job="${1:?job name required}"
gpu="${2:?GPU index required}"
data=/output/aircraft_semantic_chips/rgb_256_chip_split_seed42
root=/output/aircraft_chip_mtl
log="$root/recovery_${job}.log"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$log"; }

case "$job" in
    dota_mask_independent)
        datasets=(dota)
        mask_route=full
        notes="Recovery/resume of independent DOTA aircraft silhouette baseline; real masks and held-out scene-level mask test split; seed 0."
        ;;
    shared_conditional_semantics)
        datasets=(fineair rareplanes)
        mask_route=none
        notes="Recovery/resume of shared conditional FineAir and RarePlanes semantic chip model; balanced dataset sampling; seed 0."
        ;;
    shared_early_mask)
        datasets=(fineair rareplanes dota)
        mask_route=early
        notes="Shared semantic chip model with DOTA mask decoder at early/mid encoder features; balanced dataset sampling; seed 0."
        ;;
    *)
        record "unknown job: $job"
        exit 2
        ;;
esac

output="$root/$job"
mkdir -p "$output"
if [[ -s "$output/test_metrics.json" && -s "$output/best.pt" ]]; then
    touch "$output/completed.ok"
    record "$job already complete"
    exit 0
fi

for attempt in 1 2 3; do
    record "$job gpu=$gpu recovery attempt=$attempt started"
    if CUDA_VISIBLE_DEVICES="$gpu" python tools/train/train_aircraft_chip_mtl.py \
        --data "$data" --output "$output" --datasets "${datasets[@]}" \
        --mask-route "$mask_route" --epochs 50 --batch-size 64 \
        --workers 0 --patience 8 --seed 0 \
        --wandb-project aircraft-chip-mtl \
        --wandb-group chip-mtl-seed0-recovery-v1 \
        --run-name "${job}_recovery" --wandb-notes "$notes" >>"$log" 2>&1; then
        touch "$output/completed.ok"
        record "$job recovery completed"
        exit 0
    fi
    record "$job recovery attempt=$attempt failed"
    sleep 20
done

record "$job exhausted recovery attempts"
exit 1
