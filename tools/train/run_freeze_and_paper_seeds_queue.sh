#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_aircraft/rgb
root=/output/yolo11_baselines
base="$root/dota_aircraft_controls"
log="$base/freeze_and_paper_seeds_queue.log"
mkdir -p "$base" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

train_run() {
    local variant="$1" config="$2" output="$3" seed="$4" run_id="$5" notes="$6"
    shift 6
    local attempt resume_args
    mkdir -p "$output"
    [[ -f "$output/completed.ok" ]] && { record "$variant seed=$seed already completed"; return 0; }
    for attempt in 1 2 3; do
        resume_args=()
        [[ -s "$output/last.pt" ]] && resume_args=(--resume "$output/last.pt")
        record "$variant seed=$seed attempt $attempt"
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py --config "$config" "$@" \
            --out-folder "$output" --seed "$seed" \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group freeze-and-paper-seeds-v1 \
            --wandb-name "$variant-seed$seed" --wandb-id "$run_id" \
            --wandb-notes "$notes" "${resume_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "$variant seed=$seed training completed"
            return 0
        fi
        record "$variant seed=$seed attempt $attempt failed"
        sleep 30
    done
    record "$variant seed=$seed exhausted three attempts"
    return 1
}

evaluate_one() {
    local config="$1" checkpoint="$2" split="$3" result="$4"
    [[ -s "$result" ]] && return 0
    CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$checkpoint" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1
}

update_results() {
    PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
}

evaluate_shared() {
    local config="$1" output="$2" prefix="$3"
    evaluate_one "$config" "$output/best.pt" "$fineair/test" \
        "$root/evaluations/${prefix}_fineair_test.json" || return 1
    evaluate_one "$config" "$output/best.pt" "$rareplanes/test" \
        "$root/evaluations/${prefix}_rareplanes_test.json" || return 1
    evaluate_one "$config" "$output/best.pt" "$dota/test" \
        "$root/evaluations/${prefix}_dota_aircraft_test.json" || return 1
}

# Requested freezing control: modules 1-3 and their BN state are frozen, while
# module 0 remains trainable for four-channel FineAir input adaptation.
freeze_config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_dota_adapter_freeze_early.yaml
freeze_output="$base/dota_adapter_freeze_early/seed0"
train_run dota-adapter-freeze-early "$freeze_config" "$freeze_output" 0 \
    aircraft-dota-aircraft-y11s-dota-adapter-freeze-early-s0 \
    "DOTA-only adapter freezing ablation. YOLO backbone modules 1-3, including BatchNorm running statistics, are frozen; the RGB+IR input stem remains trainable; shared later backbone/FPN/aircraft head; detached semantics; equal sampling; stop on macro mAP50-95; seed=0." \
    --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
    --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" || exit 1
evaluate_shared "$freeze_config" "$freeze_output" \
    aircraft_dota_aircraft_dota_adapter_freeze_early || exit 1
update_results

independent_config=configs/yolo_mtl/yolo11s_dota_aircraft_detection_only.yaml
strict_config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_strict_detached.yaml
adapter_config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_dota_adapter.yaml

for seed in 1 2; do
    independent_output="$base/independent/seed$seed"
    train_run dota-aircraft-independent "$independent_config" "$independent_output" "$seed" \
        "dota-aircraft-y11s-detonly-s$seed" \
        "Independent DOTA-v2 aircraft YOLO11s OBB baseline; RGB; other DOTA objects ignored; early stopping on DOTA mAP50-95; seed=$seed." \
        --train-folders "$dota/train" --val-folders "$dota/val" || exit 1
    evaluate_one "$independent_config" "$independent_output/best.pt" "$dota/test" \
        "$root/evaluations/dota_aircraft_detection_only_seed${seed}_test.json" || exit 1
    update_results

    strict_output="$base/strict_detached/seed$seed"
    train_run aircraft-dota-strict-detached "$strict_config" "$strict_output" "$seed" \
        "aircraft-dota-aircraft-y11s-strict-det-s$seed" \
        "FineAir+RarePlanes+DOTA-aircraft shared backbone/FPN/head control; semantics strictly detached with isolated optimizers; equal sampling; stop on macro mAP50-95; seed=$seed." \
        --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
        --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" || exit 1
    evaluate_shared "$strict_config" "$strict_output" \
        "aircraft_dota_aircraft_strict_detached_seed$seed" || exit 1
    update_results

    adapter_output="$base/dota_adapter/seed$seed"
    train_run aircraft-dota-dota-adapter "$adapter_config" "$adapter_output" "$seed" \
        "aircraft-dota-aircraft-y11s-dota-adapter-s$seed" \
        "FineAir+RarePlanes+DOTA-aircraft DOTA-only adapter; shared backbone/FPN/head and DOTA gated P3/P4/P5 residual route; detached semantics; equal sampling; stop on macro mAP50-95; seed=$seed." \
        --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
        --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" || exit 1
    evaluate_shared "$adapter_config" "$adapter_output" \
        "aircraft_dota_aircraft_dota_adapter_seed$seed" || exit 1
    update_results
done

record "freeze control and paper seed queue completed"
