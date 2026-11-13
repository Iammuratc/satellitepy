#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

root=/output/yolo11_baselines
fineair=/output/yolo_mtl_patches/fineair/rgbir
rareplanes=/output/yolo_mtl_patches/rareplanes/rgb
log="$root/missing_independent_paper_seeds_queue.log"
mkdir -p "$root/evaluations"

read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=satellitepy PYTHONPATH=.
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

run_seed() {
    local dataset="$1" seed="$2" data="$3" config="$4" output="$5"
    local run_id="$6" result="$7"
    mkdir -p "$output"
    # A completed early-stopped run can occasionally return a non-zero torchrun
    # status during NCCL teardown. Preserve its valid checkpoint instead of
    # restarting a finished optimization run.
    if [[ ! -f "$output/completed.ok" && -s "$output/best.pt" && -s "$output/last.pt" ]] \
        && grep -q "early stopping at epoch=" "$log"; then
        touch "$output/completed.ok"
        record "$dataset seed=$seed accepted completed checkpoint after distributed teardown"
    fi
    if [[ ! -f "$output/completed.ok" ]]; then
        for attempt in 1 2 3; do
            local resume_args=()
            [[ -s "$output/last.pt" ]] && resume_args=(--resume "$output/last.pt")
            record "$dataset independent detection-only seed=$seed attempt=$attempt"
            if CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
                tools/train/yolo_mtl.py --config "$config" \
                --train-folders "$data/train" --val-folders "$data/val" \
                --out-folder "$output" --seed "$seed" \
                --wandb-project yolo11-multidataset-baselines \
                --wandb-group independent-detection-only-paper-seeds-v1 \
                --wandb-name "${dataset,,}-independent-detection-only-seed${seed}" \
                --wandb-id "$run_id" \
                --wandb-notes "$dataset independent YOLO11s OBB detection-only paper baseline; native dataset input channels; early stopping on dataset validation mAP50-95; seed=$seed." \
                "${resume_args[@]}" >>"$log" 2>&1; then
                touch "$output/completed.ok"
                record "$dataset seed=$seed training completed"
                break
            fi
            record "$dataset seed=$seed attempt=$attempt failed"
            sleep 30
        done
    fi
    if [[ ! -f "$output/completed.ok" ]]; then
        record "$dataset seed=$seed exhausted retries; continuing queue"
        return 0
    fi
    if [[ ! -s "$result" ]]; then
        CUDA_VISIBLE_DEVICES=0 python tools/evaluate/evaluate_sparse_yolo.py \
            --config "$config" --checkpoint "$output/best.pt" --split "$data/test" \
            --output "$result" --device cuda:0 >>"$log" 2>&1 \
            || record "$dataset seed=$seed test evaluation failed"
    fi
    PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
}

record "missing independent paper seed queue started"
run_seed FineAir 1 "$fineair" configs/yolo_mtl/yolo11s_fineair_detection_only.yaml \
    "$root/detection_only/fineair_seed1" fineair-y11s-detonly-s1 \
    "$root/evaluations/fineair_detection_only_seed1_test.json"
run_seed FineAir 2 "$fineair" configs/yolo_mtl/yolo11s_fineair_detection_only.yaml \
    "$root/detection_only/fineair_seed2" fineair-y11s-detonly-s2 \
    "$root/evaluations/fineair_detection_only_seed2_test.json"
run_seed RarePlanes 1 "$rareplanes" configs/yolo_mtl/yolo11s_rareplanes_detection_only.yaml \
    "$root/rareplanes/detection_only/seed1" rareplanes-y11s-detonly-s1 \
    "$root/evaluations/rareplanes_detection_only_seed1_test.json"
run_seed RarePlanes 2 "$rareplanes" configs/yolo_mtl/yolo11s_rareplanes_detection_only.yaml \
    "$root/rareplanes/detection_only/seed2" rareplanes-y11s-detonly-s2 \
    "$root/evaluations/rareplanes_detection_only_seed2_test.json"
touch "$root/missing_independent_paper_seeds_queue.completed.ok"
record "missing independent paper seed queue completed"
