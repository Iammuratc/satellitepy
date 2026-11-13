#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

worker="${1:?worker index required}"
gpu="${2:?GPU index required}"
fair1m=/output/vessel_semantic_chips/fair1m_official_rgb_256_split_seed42
shipimagenet=/output/vessel_semantic_chips/shipimagenet_rgb_256_split_seed42
dota=/output/vessel_semantic_chips/dota_ship_rgb_256_split_seed42
root=/output/vessel_chip_mtl
log="$root/queue_worker_${worker}.log"

mkdir -p "$root"
read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$log"; }

run_job() {
    local name="$1" seed="$2" route="$3"
    shift 3
    local datasets=("$@") output="$root/$name"
    mkdir -p "$output"
    if [[ -s "$output/test_metrics.json" && -s "$output/best.pt" ]]; then
        touch "$output/completed.ok"
        record "$name already complete"
        return 0
    fi
    for attempt in 1 2 3 4 5; do
        record "$name gpu=$gpu seed=$seed route=$route datasets=${datasets[*]} attempt=$attempt started"
        if CUDA_VISIBLE_DEVICES="$gpu" python tools/train/train_aircraft_chip_mtl.py \
            --data "$fair1m" --fair1m-data "$fair1m" \
            --shipimagenet-data "$shipimagenet" --dota-data "$dota" \
            --output "$output" --datasets "${datasets[@]}" \
            --mask-route "$route" --mask-weight 1.0 \
            --epochs 50 --batch-size 64 --workers 4 --patience 8 --seed "$seed" \
            --wandb-project vessel-chip-mtl \
            --wandb-group vessel-chip-mtl-paper-matrix-v1 \
            --run-name "$name" \
            --wandb-notes "Target-prompted vessel-chip MTL; FAIR1M and ShipImageNet semantic heads; native iSAID/DOTA ship masks; datasets=${datasets[*]}; mask route=$route; seed=$seed; scene-disjoint splits; balanced dataset sampling." \
            >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "$name completed"
            return 0
        fi
        record "$name attempt=$attempt failed; retrying from last.pt"
        sleep 10
    done
    record "$name exhausted retries; worker continues"
    return 0
}

record "vessel queue worker=$worker gpu=$gpu started"
case "$worker" in
    0)
        run_job shared_early_mask_seed0 0 early fair1m shipimagenet dota
        run_job shared_conditional_semantics_seed0 0 none fair1m shipimagenet
        run_job shared_early_mask_seed1 1 early fair1m shipimagenet dota
        run_job shared_early_mask_seed2 2 early fair1m shipimagenet dota
        run_job shared_conditional_semantics_seed1 1 none fair1m shipimagenet
        run_job shared_conditional_semantics_seed2 2 none fair1m shipimagenet
        ;;
    1)
        run_job fair1m_independent_seed0 0 none fair1m
        run_job shared_layer1_mask_seed0 0 layer1 fair1m shipimagenet dota
        run_job fair1m_independent_seed1 1 none fair1m
        run_job fair1m_independent_seed2 2 none fair1m
        run_job shared_layer1_mask_seed1 1 layer1 fair1m shipimagenet dota
        run_job shared_layer1_mask_seed2 2 layer1 fair1m shipimagenet dota
        run_job shared_early_mask_detached_seed0 0 early_detached fair1m shipimagenet dota
        run_job shared_early_mask_detached_seed1 1 early_detached fair1m shipimagenet dota
        run_job shared_early_mask_detached_seed2 2 early_detached fair1m shipimagenet dota
        ;;
    2)
        run_job shipimagenet_independent_seed0 0 none shipimagenet
        run_job shared_layer3_mask_seed0 0 layer3 fair1m shipimagenet dota
        run_job shipimagenet_independent_seed1 1 none shipimagenet
        run_job shipimagenet_independent_seed2 2 none shipimagenet
        run_job shared_layer3_mask_seed1 1 layer3 fair1m shipimagenet dota
        run_job shared_layer3_mask_seed2 2 layer3 fair1m shipimagenet dota
        run_job shared_full_mask_seed0 0 full fair1m shipimagenet dota
        run_job shared_full_mask_seed1 1 full fair1m shipimagenet dota
        run_job shared_full_mask_seed2 2 full fair1m shipimagenet dota
        ;;
    3)
        run_job dota_mask_independent_full_seed0 0 full dota
        run_job dota_mask_independent_early_seed0 0 early dota
        run_job dota_mask_independent_full_seed1 1 full dota
        run_job dota_mask_independent_full_seed2 2 full dota
        run_job dota_mask_independent_early_seed1 1 early dota
        run_job dota_mask_independent_early_seed2 2 early dota
        ;;
    *)
        record "unknown worker index: $worker"
        exit 2
        ;;
esac

touch "$root/queue_worker_${worker}.completed.ok"
record "vessel queue worker=$worker completed"
