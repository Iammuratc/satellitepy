#!/usr/bin/env bash
set -uo pipefail
cd /workspace/satellitepy || exit 1

log=/output/yolo11_baselines/dota_late_backbone_then_masks_queue.log
record() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log"; }

record "waiting for the no-mask late-backbone/FPN runner"
while pgrep -f 'bash tools/train/[r]un_dota_late_backbone_fpn_seed0.sh' >/dev/null; do
    sleep 60
done
record "no-mask runner ended; starting the DOTA mask-auxiliary ablation"
exec bash tools/train/run_dota_late_backbone_fpn_masks_seed0.sh
