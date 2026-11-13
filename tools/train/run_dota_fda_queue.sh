#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
dota=/output/yolo_mtl_patches_fda/dota_aircraft/rgb
root=/output/yolo11_baselines
base="$root/dota_fda_controls"
prior="$root/dota_gsd030_controls"
log="$base/queue.log"
mkdir -p "$base" "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

record "waiting for FDA patches and GSD queue"
while [[ ! -s "$dota/manifest.json" \
      || ! -s "$dota/train/manifest.json" \
      || ! -s "$dota/val/manifest.json" \
      || ! -s "$dota/test/manifest.json" \
      || ! -f "$prior/queue.completed.ok" ]]; do
    sleep 60
done

train_run() {
    local variant="$1" config="$2" output="$3" run_id="$4" notes="$5"
    shift 5
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
            --wandb-group dota-fda-controls-seed0-v1 \
            --wandb-name "$variant-seed0" --wandb-id "$run_id" \
            --wandb-notes "$notes" "${resume_args[@]}" >>"$log" 2>&1; then
            touch "$output/completed.ok"
            record "$variant training completed"
            return 0
        fi
        record "$variant attempt $attempt failed"
        sleep 30
    done
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
train_run dota-fda-independent "$independent_config" "$independent_output" \
    dota-aircraft-fda-y11s-detonly-s0 \
    "Independent DOTA-aircraft FDA control. Training-only Fourier low-frequency amplitude replacement from alternating FineAir/RarePlanes RGB references; beta=0.01, amplitude mix=1.0, assignment seed=42; native DOTA validation/test; seed=0." \
    --train-folders "$dota/train" --val-folders "$dota/val" || exit 1
evaluate_one "$independent_config" "$independent_output/best.pt" "$dota/test" \
    "$root/evaluations/dota_aircraft_fda_detection_only_seed0_test.json" || exit 1
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true

adapter_config=configs/yolo_mtl/yolo11s_aircraft_dota_aircraft_dota_adapter.yaml
adapter_output="$base/dota_adapter/seed0"
train_run aircraft-dota-fda-adapter "$adapter_config" "$adapter_output" \
    aircraft-dota-aircraft-fda-y11s-dota-adapter-s0 \
    "FineAir+RarePlanes+DOTA-aircraft FDA control. DOTA training patches use Fourier low-frequency amplitude replacement from alternating FineAir/RarePlanes RGB references; beta=0.01, mix=1.0, seed=42; native DOTA validation/test; DOTA-only gated adapter; detached semantics; equal sampling; stop on macro mAP50-95; seed=0." \
    --train-folders "$fineair/train" "$rareplanes/train" "$dota/train" \
    --val-folders "$fineair/val" "$rareplanes/val" "$dota/val" || exit 1
for spec in "fineair:$fineair/test" "rareplanes:$rareplanes/test" "dota_aircraft:$dota/test"; do
    name=${spec%%:*}; split=${spec#*:}
    evaluate_one "$adapter_config" "$adapter_output/best.pt" "$split" \
        "$root/evaluations/aircraft_dota_aircraft_fda_dota_adapter_${name}_test.json" || exit 1
done
PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
touch "$base/queue.completed.ok"
record "DOTA FDA queue completed"
