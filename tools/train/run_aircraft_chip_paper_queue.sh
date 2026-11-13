#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

worker="${1:?worker index required}"
gpu="${2:?GPU index required}"
data=/output/aircraft_semantic_chips/rgb_256_chip_split_seed42
root=/output/aircraft_chip_mtl
log="$root/paper_queue_worker_${worker}.log"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$log"; }

run_job() {
    local name="$1" seed="$2" mask_route="$3" mask_weight="$4"
    shift 4
    local datasets=("$@")
    local output="$root/$name"
    mkdir -p "$output"
    if [[ -s "$output/test_metrics.json" && -s "$output/best.pt" ]]; then
        touch "$output/completed.ok"
        record "$name already complete"
        return 0
    fi
    for attempt in 1 2 3; do
        record "$name gpu=$gpu seed=$seed route=$mask_route weight=$mask_weight attempt=$attempt started"
        if CUDA_VISIBLE_DEVICES="$gpu" python tools/train/train_aircraft_chip_mtl.py \
            --data "$data" --output "$output" --datasets "${datasets[@]}" \
            --mask-route "$mask_route" --mask-weight "$mask_weight" \
            --epochs 50 --batch-size 64 --workers 4 --patience 8 --seed "$seed" \
            --wandb-project aircraft-chip-mtl \
            --wandb-group chip-mtl-paper-controls-v1 \
            --run-name "$name" \
            --wandb-notes "Aircraft chip paper control; datasets=${datasets[*]}; mask route=$mask_route; mask weight=$mask_weight; seed=$seed; scene-disjoint chip split." \
            >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "$name completed"
            return 0
        fi
        record "$name attempt=$attempt failed; retrying from last.pt"
        sleep 20
    done
    record "$name exhausted retries; worker continues"
    return 0
}

record "paper queue worker=$worker gpu=$gpu started"
case "$worker" in
    0)
        run_job shared_early_mask_seed1 1 early 1.0 fineair rareplanes dota
        run_job shared_early_mask_seed2 2 early 1.0 fineair rareplanes dota
        ;;
    1)
        run_job fineair_independent_seed1 1 none 1.0 fineair
        run_job fineair_independent_seed2 2 none 1.0 fineair
        run_job shared_layer1_mask_seed0 0 layer1 1.0 fineair rareplanes dota
        run_job shared_early_mask_weight05_seed0 0 early 0.5 fineair rareplanes dota
        ;;
    2)
        run_job rareplanes_independent_seed1 1 none 1.0 rareplanes
        run_job rareplanes_independent_seed2 2 none 1.0 rareplanes
        run_job shared_layer3_mask_seed0 0 layer3 1.0 fineair rareplanes dota
        ;;
    3)
        run_job dota_mask_independent_seed1 1 full 1.0 dota
        run_job dota_mask_independent_seed2 2 full 1.0 dota
        run_job shared_early_mask_detached_seed0 0 early_detached 1.0 fineair rareplanes dota
        ;;
    *)
        record "unknown worker index: $worker"
        exit 2
        ;;
esac

touch "$root/paper_queue_worker_${worker}.completed.ok"
record "paper queue worker=$worker completed"
if [[ -f "$root/paper_queue_worker_0.completed.ok" &&
      -f "$root/paper_queue_worker_1.completed.ok" &&
      -f "$root/paper_queue_worker_2.completed.ok" &&
      -f "$root/paper_queue_worker_3.completed.ok" ]]; then
    touch "$root/paper_queue.completed.ok"
fi
