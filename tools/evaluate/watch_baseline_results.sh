#!/usr/bin/env bash
set -uo pipefail

cd /workspace/satellitepy || exit 1
read -r WANDB_API_KEY < docs/wandb_api.txt
export WANDB_API_KEY WANDB_BASE_URL=https://api.wandb.ai
log=/output/yolo11_baselines/baseline_results_updater.log

while true; do
    PYTHONPATH=. python tools/evaluate/update_baseline_results.py >>"$log" 2>&1
    queue_running=false
    for cmdline in /proc/[0-9]*/cmdline; do
        command=$(tr '\0' ' ' < "$cmdline" 2>/dev/null) || continue
        if [[ "$command" == *resume_fineair_rareplanes_controls.sh* ]]; then
            queue_running=true
            break
        fi
    done
    if [[ "$queue_running" == false ]]; then
        break
    fi
    sleep 300
done
