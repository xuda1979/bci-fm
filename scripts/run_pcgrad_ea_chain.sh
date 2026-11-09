#!/bin/bash
# Chain after pcgrad_full: EA follow-up cells + PCGrad x EA cell.
# Usage: run_pcgrad_ea_chain.sh <PCGRAD_PID>
set -u
cd /Users/daxu/software/bci-fm

PCGRAD_PID=${1:?need PID of running pcgrad_full process}

echo "[$(date)] waiting for pcgrad_full (PID $PCGRAD_PID)..."
while kill -0 "$PCGRAD_PID" 2>/dev/null; do sleep 120; done
echo "[$(date)] pcgrad_full finished. launching EA follow-up chain."

# 1) TC-MT MI-only + EA  (best single-task arch x best alignment)
python src/mtl.py --folds 4 --epochs 30 --test-subjects 27 \
    --use-exec 0 --use-rest 0 --align 1 --tag mtl_mi_only_ea > /tmp/mtl_mi_only_ea.log 2>&1
echo "[$(date)] mtl_mi_only_ea done: $(python -c "import json;print(json.load(open('results/finetune_mtl_mi_only_ea.json'))['mean_test_acc'])" 2>/dev/null)"

# 2) TC-MT MI+rest + EA
python src/mtl.py --folds 4 --epochs 30 --test-subjects 27 \
    --use-exec 0 --align 1 --tag mtl_mi_rest_ea > /tmp/mtl_mi_rest_ea.log 2>&1
echo "[$(date)] mtl_mi_rest_ea done: $(python -c "import json;print(json.load(open('results/finetune_mtl_mi_rest_ea.json'))['mean_test_acc'])" 2>/dev/null)"

# 3) PCGrad + EA  (gradient surgery x alignment interaction)
python src/mtl_pcgrad.py --folds 4 --epochs 30 --align 1 --tag pcgrad_full_ea > /tmp/pcgrad_full_ea.log 2>&1
echo "[$(date)] pcgrad_full_ea done: $(python -c "import json;print(json.load(open('results/finetune_pcgrad_full_ea.json'))['mean_test_acc'])" 2>/dev/null)"

echo "[$(date)] chain complete"
