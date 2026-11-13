#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

marker=/output/yolo11_baselines/missing_independent_paper_seeds_queue.completed.ok
log=/output/yolo11_baselines/semantic_paper_baselines.log

while [[ ! -f "$marker" ]]; do
    sleep 30
done

printf '%s semantic checkpoint summary started\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"$log"
python tools/evaluate/summarize_semantic_paper_baselines.py >>"$log" 2>&1
python tools/evaluate/update_baseline_results.py >>"$log" 2>&1 || true
printf '%s semantic checkpoint summary completed\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"$log"
