#!/bin/bash
# Follow-up runs after the ablation matrix: EA combinations that are missing cells.
set -u
cd /Users/daxu/software/bci-fm

CHAIN_PID=${1:-83829}
echo "[$(date)] waiting for ablation chain (PID $CHAIN_PID)..."
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] ablation chain finished. launching EA follow-ups."

# 5) TC-MT MI-only + EA  (expected best: best arch x best alignment)
python src/mtl.py --folds 4 --epochs 30 --test-subjects 27 \
    --use-exec 0 --use-rest 0 --align 1 --tag mtl_mi_only_ea > /tmp/mtl_mi_only_ea.log 2>&1
echo "[$(date)] mtl_mi_only_ea done"

# 6) TC-MT MI+rest + EA  (tests whether rest task synergizes under EA)
python src/mtl.py --folds 4 --epochs 30 --test-subjects 27 \
    --use-exec 0 --align 1 --tag mtl_mi_rest_ea > /tmp/mtl_mi_rest_ea.log 2>&1
echo "[$(date)] mtl_mi_rest_ea done"

echo "[$(date)] EA follow-ups complete"
