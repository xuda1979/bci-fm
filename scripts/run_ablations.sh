#!/bin/bash
# Ablation matrix runner: waits for mtl_full to finish, then runs all ablations
# sequentially (GPU/MPS is single-tenant). Each run uses the identical 4-fold
# x 30ep protocol; only the ablated flag differs.
set -u
cd /Users/daxu/software/bci-fm

MAIN_PID=${1:-75672}

echo "[$(date)] waiting for main run (PID $MAIN_PID)..."
while kill -0 "$MAIN_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] main run finished. launching ablation matrix."

# 1) MI-only baseline under identical MTL code path (isolates multi-task gain)
python src/mtl.py --folds 4 --epochs 30 --test-subjects 27 \
    --use-exec 0 --use-rest 0 --tag mtl_mi_only > /tmp/mtl_mi_only.log 2>&1
echo "[$(date)] mtl_mi_only done"

# 2) No task tokens (shared conditioning) — isolates task-conditioning gain
python src/mtl.py --folds 4 --epochs 30 --test-subjects 27 \
    --no-task-tokens --tag mtl_no_tokens > /tmp/mtl_no_tokens.log 2>&1
echo "[$(date)] mtl_no_tokens done"

# 3) MI + rest (no exec)
python src/mtl.py --folds 4 --epochs 30 --test-subjects 27 \
    --use-exec 0 --tag mtl_mi_rest > /tmp/mtl_mi_rest.log 2>&1
echo "[$(date)] mtl_mi_rest done"

# 4) EA + full multi-task (combined with domain alignment)
python src/mtl.py --folds 4 --epochs 30 --test-subjects 27 \
    --align --tag mtl_full_ea > /tmp/mtl_full_ea.log 2>&1
echo "[$(date)] mtl_full_ea done"

echo "[$(date)] all ablations complete"
