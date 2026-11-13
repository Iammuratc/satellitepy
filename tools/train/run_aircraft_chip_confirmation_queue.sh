#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

worker="${1:?worker index required}"
gpu="${2:?GPU index required}"
data=/output/aircraft_semantic_chips/rgb_256_chip_split_seed42
root=/output/aircraft_chip_mtl
log="$root/confirmation_queue_worker_${worker}.log"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$log"; }
refresh_table() {
    flock /tmp/aircraft_chip_results.lock \
        python tools/evaluate/update_aircraft_chip_results.py >>"$log" 2>&1 || true
}

run_job() {
    local name="$1" seed="$2" mask_route="$3"
    shift 3
    local datasets=("$@")
    local output="$root/$name"
    mkdir -p "$output"
    if [[ -s "$output/test_metrics.json" && -s "$output/best.pt" ]]; then
        touch "$output/completed.ok"
        record "$name already complete"
        refresh_table
        return 0
    fi
    for attempt in 1 2 3; do
        record "$name gpu=$gpu seed=$seed route=$mask_route attempt=$attempt started"
        if CUDA_VISIBLE_DEVICES="$gpu" python tools/train/train_aircraft_chip_mtl.py \
            --data "$data" --output "$output" --datasets "${datasets[@]}" \
            --mask-route "$mask_route" --mask-weight 1.0 \
            --epochs 50 --batch-size 64 --workers 4 --patience 8 --seed "$seed" \
            --wandb-project aircraft-chip-mtl \
            --wandb-group chip-mtl-routing-confirmation-v1 \
            --run-name "$name" \
            --wandb-notes "Three-seed aircraft-chip routing confirmation; datasets=${datasets[*]}; route=$mask_route; seed=$seed." \
            >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "$name completed"
            refresh_table
            return 0
        fi
        record "$name attempt=$attempt failed; retrying from last.pt"
        sleep 20
    done
    record "$name exhausted retries; worker continues"
    return 0
}

record "confirmation worker=$worker gpu=$gpu started"
case "$worker" in
    0)
        run_job shared_layer3_mask_seed1 1 layer3 fineair rareplanes dota
        run_job shared_layer3_mask_seed2 2 layer3 fineair rareplanes dota
        ;;
    1)
        run_job shared_conditional_semantics_seed1 1 none fineair rareplanes
        run_job shared_conditional_semantics_seed2 2 none fineair rareplanes
        run_job shared_early_mask_detached_seed1 1 early_detached fineair rareplanes dota
        ;;
    2)
        run_job shared_full_mask_seed1 1 full fineair rareplanes dota
        run_job shared_full_mask_seed2 2 full fineair rareplanes dota
        run_job shared_early_mask_detached_seed2 2 early_detached fineair rareplanes dota
        ;;
    3)
        run_job shared_layer1_mask_seed1 1 layer1 fineair rareplanes dota
        run_job shared_layer1_mask_seed2 2 layer1 fineair rareplanes dota
        ;;
    *)
        record "unknown worker index: $worker"
        exit 2
        ;;
esac

touch "$root/confirmation_queue_worker_${worker}.completed.ok"
record "confirmation worker=$worker completed"
if [[ -f "$root/confirmation_queue_worker_0.completed.ok" &&
      -f "$root/confirmation_queue_worker_1.completed.ok" &&
      -f "$root/confirmation_queue_worker_2.completed.ok" &&
      -f "$root/confirmation_queue_worker_3.completed.ok" ]]; then
    touch "$root/confirmation_queue.completed.ok"
    refresh_table
fi
