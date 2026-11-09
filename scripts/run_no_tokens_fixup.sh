#!/bin/bash
# Fixup: mtl_no_tokens ablation failed in run_ablations.sh because
# --no-task-tokens is type=int and the script passed it as a bare flag.
# Re-run it with the correct value after the EA follow-up chain finishes.
set -u
cd /Users/daxu/software/bci-fm

EA_CHAIN_PID=${1:-3220}

echo "[$(date)] waiting for EA follow-up chain (PID $EA_CHAIN_PID)..."
while kill -0 "$EA_CHAIN_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] EA chain finished. launching mtl_no_tokens fixup."

# No task tokens (shared conditioning) — isolates task-conditioning gain
python src/mtl.py --folds 4 --epochs 30 --test-subjects 27 \
    --no-task-tokens 1 --tag mtl_no_tokens > /tmp/mtl_no_tokens.log 2>&1
echo "[$(date)] mtl_no_tokens done"
