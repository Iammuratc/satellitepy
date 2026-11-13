#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches/dota_aircraft/rgb
root=/output/yolo11_baselines
base="$root/dota_aircraft_controls"
log="$base/seed0_queue.log"
mkdir -p "$base" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

train_run() {
    local variant="$1" config="$2" output="$3" run_id="$4" run_name="$5" notes="$6"
    shift 6
    local attempt resume_args
    mkdir -p "$output"
    [[ -f "$output/completed.ok" ]] && { record "$variant already completed"; return 0; }
    for attempt in 1 2 3; do
        resume_args=()
        [[ -s "$output/last.pt" ]] && resume_args=(--resume "$output/last.pt")
        record "$variant attempt $attempt"
        if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
            tools/train/yolo_mtl.py --config "$config" "$@" \
            --out-folder "$output" --seed 0 \
            --wandb-project yolo11-multidataset-baselines \
            --wandb-group dota-aircraft-domain-controls-seed0-v1 \
            --wandb-name "$run_name" --wandb-id "$run_id" --wandb-notes "$notes" \
            "${resume_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "$variant training completed"
            return 0
        fi
        record "$variant attempt $attempt failed"
        sleep 30
    done
    record "$variant exhausted three attempts"
    return 1
}

evaluate_one() {
    local config="$1" checkpoint="$2" split="$3" result="$4"
    [[ -s "$result" ]] && return 0
    CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
        --config "$config" --checkpoint "$checkpoint" --split "$split" \
        --output "$result" --device cuda:0 >>"$log" 2>&1
}

independent_config=configs/yolo_mtl/yolo11s_dota_aircraft_detection_only.yaml
independent_output="$base/independent/seed0"
train_run independent "$independent_config" "$independent_output" \
    dota-aircraft-y11s-detonly-s0 dota-aircraft-detection-only-yolo11s-seed0 \
    "Independent DOTA-v2 airplane-only YOLO11s OBB baseline. RGB; other annotated DOTA objects retained as ignored boxes; early stopping on DOTA-aircraft mAP50-95; seed=0." \
    --train-folders "$dota/train" --val-folders "$dota/val" || exit 1
evaluate_one "$independent_config" "$independent_output/best.pt" "$dota/test" \
    "$root/evaluations/dota_aircraft_detection_only_seed0_test.json" || exit 1

for variant in strict_detached detection_gated; do
    if [[ "$variant" == strict_detached ]]; then
        config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_strict_detached.yaml
        run_id=aircraft-dota-aircraft-y11s-strict-det-s0
        notes="FineAir+RarePlanes+DOTA-aircraft same-object domain control. One aircraft OBB head; shared backbone/FPN; detached semantics with isolated optimizers; equal three-dataset sampling; stop on macro mAP50-95; seed=0."
    else
        config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_gated.yaml
        run_id=aircraft-dota-aircraft-y11s-detection-gated-s0
        notes="FineAir+RarePlanes+DOTA-aircraft detection-gated domain control. One aircraft OBB head with detection-trained P3/P4/P5 residual gates; semantics strictly detached; equal three-dataset sampling; stop on macro mAP50-95; seed=0."
    fi
    output="$base/$variant/seed0"
    train_run "$variant" "$config" "$output" "$run_id" \
        "aircraft-dota-aircraft-$variant-seed0" "$notes" \
        --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
        --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" || exit 1
    evaluate_one "$config" "$output/best.pt" "$fineair/test" \
        "$root/evaluations/aircraft_dota_aircraft_${variant}_fineair_test.json" || exit 1
    evaluate_one "$config" "$output/best.pt" "$rareplanes/test" \
        "$root/evaluations/aircraft_dota_aircraft_${variant}_rareplanes_test.json" || exit 1
    evaluate_one "$config" "$output/best.pt" "$dota/test" \
        "$root/evaluations/aircraft_dota_aircraft_${variant}_dota_aircraft_test.json" || exit 1
done

PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
record "DOTA-aircraft seed-0 control queue finished"
