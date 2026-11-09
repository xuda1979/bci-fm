#!/bin/bash
# Protocol consistency fix-ups: run AFTER the EA chain (PID arg).
# 1) EEGNet baseline with --strict-folds (exact mtl.py fold protocol)
# 2) MI+exec under strict protocol (v2's mi_exec used merge-remainder folds)
set -u
cd /Users/daxu/software/bci-fm
CHAIN_PID=${1:?need PID of EA chain bash process}
echo "[$(date)] waiting for EA chain (PID $CHAIN_PID)..."
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] EA chain done. running protocol fix-ups."

python src/baseline_eegnet.py --folds 4 --epochs 60 --batch-size 64 --lr 0.01 \
  --test-subjects 27 --strict-folds --tag eegnet_strict > /tmp/eegnet_strict.log 2>&1
echo "[$(date)] eegnet_strict done: $(python -c "import json;print(json.load(open('results/eegnet_strict.json'))['mean_test_acc'])" 2>/dev/null)"

python src/mtl.py --folds 4 --epochs 30 --test-subjects 27 \
  --use-exec 1 --use-rest 0 --align 0 --tag mtl_mi_exec > /tmp/mtl_mi_exec.log 2>&1
echo "[$(date)] mtl_mi_exec done: $(python -c "import json;print(json.load(open('results/finetune_mtl_mi_exec.json'))['mean_test_acc'])" 2>/dev/null)"

echo "[$(date)] protocol fix-ups complete"
