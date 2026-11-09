"""Finetune CBraMod on eegmmidb motor-imagery 4-class task (Apple MPS).

Load:  data/processed/S*_19ch.npz  (19ch, 200Hz, 4s epochs)
Model: braindecode CBraMod + pretrained weights (safetensors)
Eval:  leave-N-subjects-out cross-validation (cross-subject generalization)

Usage:
  python3 src/finetune.py --folds 5 --epochs 30 --batch-size 32
  python3 src/finetune.py --quick   # 1 fold smoke test
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from braindecode.models import CBraMod
from safetensors.torch import load_file

DATA_DIR = "data/processed"
RESULTS_DIR = "results"
WEIGHTS = "models/model.safetensors"
LABELS = ["left_hand", "right_hand", "both_fists", "both_feet"]


def load_all_subjects():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith("_19ch.npz"))
    subjects = []
    for fn in files:
        d = np.load(os.path.join(DATA_DIR, fn))
        subjects.append({
            "name": fn.split("_")[0],
            "X": d["X"],           # [90, 19, 800]
            "y": d["y"],           # [90]
        })
    print(f"loaded {len(subjects)} subjects: {[s['name'] for s in subjects]}")
    return subjects


def zscore_per_epoch(X):
    """Per-epoch, per-channel z-scoring (standard for EEG decoders)."""
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-6
    return (X - mu) / sd


def trim_to_patch(X, patch=200):
    """Trim trailing time samples so n_times is divisible by CBraMod patch (200)."""
    n_times = X.shape[-1]
    usable = (n_times // patch) * patch
    return X[..., :usable]


def build_model(device, n_outputs=4):
    model = CBraMod(n_chans=19, n_outputs=n_outputs, sfreq=200,
                    input_window_seconds=4.0)
    sd = load_file(WEIGHTS)
    model_sd = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    print(f"loaded {len(matched)}/{len(model_sd)} pretrained tensors")
    return model.to(device)


def make_loaders(train_X, train_y, val_X, val_y, batch_size):
    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y).long())
    val_ds = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y).long())
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    return train_ld, val_ld


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_class = np.zeros(4)
    per_class_total = np.zeros(4)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
            for c in range(4):
                mask = yb == c
                per_class[c] += (pred[mask] == c).sum().item()
                per_class_total[c] += mask.sum().item()
    acc = correct / max(total, 1)
    per_class_acc = per_class / np.maximum(per_class_total, 1)
    return acc, per_class_acc


def train_one_fold(model, train_ld, val_ld, device, epochs, lr, log_prefix=""):
    # differential LR: lower for pretrained backbone, higher for new head
    head_params = [p for n, p in model.named_parameters() if "final_layer" in n]
    body_params = [p for n, p in model.named_parameters() if "final_layer" not in n]
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ], weight_decay=0.01)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_state, patience, bad = 0.0, None, 10, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = n = 0
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item() * yb.numel()
            n += yb.numel()
        sched.step()

        val_acc, _ = evaluate(model, val_ld, device)
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            marker = " *"
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1 or marker:
            print(f"  {log_prefix}ep{ep:03d} loss={tot_loss/n:.4f} val_acc={val_acc:.4f}{marker} [{time.time()-t0:.0f}s]", flush=True)
        if bad >= patience:
            print(f"  {log_prefix}early stop at ep{ep}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-subjects", type=int, default=4, help="held-out subjects per fold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.folds, args.epochs, args.test_subjects = 1, 3, 4

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    subjects = load_all_subjects()
    n_sub = len(subjects)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # leave-N-subjects-out CV
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_sub)
    fold_size = args.test_subjects
    folds = []
    for i in range(0, n_sub, fold_size):
        test_idx = perm[i:i + fold_size]
        if len(test_idx) < fold_size and folds:
            folds[-1] = np.concatenate([folds[-1], test_idx])
        else:
            folds.append(test_idx)
    folds = folds[:args.folds]

    results = []
    for k, test_idx in enumerate(folds):
        print(f"\n=== FOLD {k+1}/{len(folds)} | test subjects: {[subjects[i]['name'] for i in test_idx]} ===", flush=True)
        test_subs = [subjects[i] for i in test_idx]
        train_subs = [subjects[i] for i in range(n_sub) if i not in test_idx]
        if not train_subs:
            print("no training subjects in this split, skipping", flush=True)
            continue

        # split train subjects into train/val for early stopping
        n_val = max(1, len(train_subs) // 5)
        rng2 = np.random.RandomState(args.seed + k)
        train_perm = rng2.permutation(len(train_subs))
        val_subs = [train_subs[i] for i in train_perm[:n_val]]
        tr_subs = [train_subs[i] for i in train_perm[n_val:]]

        train_X = trim_to_patch(zscore_per_epoch(np.concatenate([s["X"] for s in tr_subs])))
        train_y = np.concatenate([s["y"] for s in tr_subs])
        val_X = trim_to_patch(zscore_per_epoch(np.concatenate([s["X"] for s in val_subs])))
        val_y = np.concatenate([s["y"] for s in val_subs])
        test_X = trim_to_patch(zscore_per_epoch(np.concatenate([s["X"] for s in test_subs])))
        test_y = np.concatenate([s["y"] for s in test_subs])

        model = build_model(device)
        train_ld, val_ld = make_loaders(train_X, train_y, val_X, val_y, args.batch_size)
        _, test_ld = make_loaders(train_X, train_y, test_X, test_y, args.batch_size)

        best_val = train_one_fold(model, train_ld, val_ld, device, args.epochs, args.lr)

        test_acc, per_class = evaluate(model, test_ld, device)
        print(f"FOLD {k+1}: val={best_val:.4f} test={test_acc:.4f} per_class={np.round(per_class,3).tolist()}", flush=True)
        results.append({"fold": k + 1, "test_subjects": [subjects[i]["name"] for i in test_idx],
                        "val_acc": best_val, "test_acc": test_acc,
                        "per_class": per_class.tolist()})

        if args.save_model and k == 0:
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "cbramod_finetuned_fold1.pt"))
            print("saved finetuned model (fold 1)")

    accs = [r["test_acc"] for r in results]
    summary = {
        "task": "eegmmidb motor-imagery 4-class, cross-subject leave-out CV",
        "model": "CBraMod finetuned",
        "n_subjects": n_sub,
        "folds": len(results),
        "mean_test_acc": float(np.mean(accs)),
        "std_test_acc": float(np.std(accs)),
        "chance": 0.25,
        "results": results,
        "args": vars(args),
    }
    out = os.path.join(RESULTS_DIR, "finetune_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}\nMEAN TEST ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f} (chance 0.25)\n{'='*60}")
    print(f"results saved to {out}")


if __name__ == "__main__":
    main()
