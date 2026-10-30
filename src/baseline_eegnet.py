"""EEGNetv4 from-scratch baseline on eegmmidb MI 4-class (identical CV protocol).

Purpose: ablation control for CBraMod finetuning. Same folds, same preprocessing,
same eval — only the model differs (no pretrained weights).

Usage:
  python3 src/baseline_eegnet.py --folds 5 --epochs 60
  python3 src/baseline_eegnet.py --quick
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
from torch.utils.data import DataLoader, TensorDataset

from braindecode.models import EEGNet

DATA_DIR = "data/processed"
RESULTS_DIR = "results"


def zscore_per_epoch(X):
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-6
    return (X - mu) / sd


def trim_to_patch(X, patch=200):
    usable = (X.shape[-1] // patch) * patch
    return X[..., :usable]


def make_loaders(train_X, train_y, val_X, val_y, batch_size):
    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y).long())
    val_ds = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y).long())
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False))


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--test-subjects", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="eegnet_20sub")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.folds, args.epochs = 1, 3

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device} | EEGNet from scratch, lr={args.lr}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith("_19ch.npz"))
    subjects = []
    for fn in files:
        d = np.load(os.path.join(DATA_DIR, fn))
        subjects.append({"name": fn.split("_")[0], "X": d["X"], "y": d["y"]})
    n_sub = len(subjects)
    print(f"loaded {n_sub} subjects")

    # identical fold split as finetune.py (same seed & logic)
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
        print(f"\n=== FOLD {k+1}/{len(folds)} | test: {[subjects[i]['name'] for i in test_idx]} ===", flush=True)
        test_subs = [subjects[i] for i in test_idx]
        train_subs = [subjects[i] for i in range(n_sub) if i not in test_idx]
        if not train_subs:
            continue

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

        # EEGNet (v4-style): 19ch, 800 samples @ 200Hz
        model = EEGNet(n_chans=19, n_outputs=4, n_times=800, sfreq=200,
                       input_window_seconds=4.0, drop_prob=0.4,
                       final_conv_length="auto").to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if k == 0:
            print(f"EEGNet params: {n_params:,}")

        train_ld, val_ld = make_loaders(train_X, train_y, val_X, val_y, args.batch_size)
        test_ld = DataLoader(TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y).long()),
                             batch_size=args.batch_size * 2)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_acc, best_state, bad, patience = 0.0, None, 0, 12
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            for xb, yb in train_ld:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            va = evaluate(model, val_ld, device)
            if va > best_acc:
                best_acc, bad, marker = va, 0, " *"
                best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            else:
                bad += 1; marker = ""
            if ep % 10 == 0 or ep == 1:
                print(f"  ep{ep:03d} val={va:.4f}{marker} [{time.time()-t0:.0f}s]", flush=True)
            if bad >= patience:
                print(f"  early stop ep{ep}", flush=True)
                break

        if best_state: model.load_state_dict(best_state)
        test_acc = evaluate(model, test_ld, device)
        print(f"FOLD {k+1}: val={best_acc:.4f} test={test_acc:.4f}", flush=True)
        results.append({"fold": k+1, "test_subjects": [subjects[i]["name"] for i in test_idx],
                        "val_acc": best_acc, "test_acc": test_acc})

    accs = [r["test_acc"] for r in results]
    summary = {"model": "EEGNet from scratch", "n_subjects": n_sub, "folds": len(results),
               "mean_test_acc": float(np.mean(accs)), "std_test_acc": float(np.std(accs)),
               "lr": args.lr, "results": results, "args": vars(args)}
    out = os.path.join(RESULTS_DIR, f"{args.tag}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nEEGNet MEAN TEST ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
