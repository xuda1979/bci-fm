"""Finetune CBraMod v2: multi-source training + Euclidean Alignment ablations.

Extends finetune.py with three orthogonal capabilities:

1. Multi-source data: combine motor-imagery (tag _19ch) and motor-execution
   (tag _exec19ch) epochs as training data. Execution runs share the 4-class
   label space (except tongue, which has no execution analog).
   Test is ALWAYS motor imagery only (the task we care about).

2. Euclidean Alignment (EA): per-subject whitening X <- R^{-1/2} X where
   R = mean covariance of that subject's epochs (over the whole recording
   session used). Applied per subject independently, including test subjects
   (unsupervised — no label leakage). Standard trick from Riemannian BCI
   literature (He & Wu, IEEE TBME 2019) for cross-subject transfer.

3. Tagged result files so multiple experiments can coexist in results/.

Usage examples:
  # MI only + EA (ablation)
  python3 src/finetune_v2.py --use-exec 0 --align 1 --tag mi_ea

  # MI + execution joint training, no EA
  python3 src/finetune_v2.py --use-exec 1 --align 0 --tag mi_exec

  # MI + execution joint training + EA
  python3 src/finetune_v2.py --use-exec 1 --align 1 --tag mi_exec_ea
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from braindecode.models import CBraMod
from safetensors.torch import load_file

DATA_DIR = "data/processed"
RESULTS_DIR = "results"
WEIGHTS = "models/model.safetensors"
LABELS = ["left_hand", "right_hand", "feet", "tongue"]


def load_mi_subjects():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith("_19ch.npz"))
    subjects = []
    for fn in files:
        d = np.load(os.path.join(DATA_DIR, fn), allow_pickle=True)
        subjects.append({
            "name": fn.split("_")[0],
            "X": d["X"],
            "y": d["y"],
        })
    print(f"loaded {len(subjects)} MI subjects")
    return subjects


def load_exec_subjects():
    """Execution epochs per subject, same label space (no tongue)."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith("_exec19ch.npz"))
    out = {}
    for fn in files:
        d = np.load(os.path.join(DATA_DIR, fn), allow_pickle=True)
        out[fn.split("_")[0]] = (d["X"], d["y"])
    print(f"loaded execution data for {len(out)} subjects")
    return out


def zscore_per_epoch(X):
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-6
    return (X - mu) / sd


def trim_to_patch(X, patch=200):
    n_times = X.shape[-1]
    usable = (n_times // patch) * patch
    return X[..., :usable]


def euclidean_align(X):
    """Per-subject Euclidean Alignment. X: [n_epochs, ch, t] -> R^{-1/2} @ X.

    R = mean over epochs of (x x^T / n_times). Aligns each subject's
    covariance to identity, drastically reducing inter-subject feature shift.
    Unsupervised: usable on test subjects without label leakage.
    """
    n, ch, t = X.shape
    R = np.einsum("nct,nst->cs", X, X) / (n * t)
    R += 1e-8 * np.eye(ch)
    # symmetrize and whiten via eigendecomposition
    w, V = np.linalg.eigh(R)
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    return np.einsum("cs,nst->nct", W, X)


def build_model(device, n_outputs=4):
    model = CBraMod(n_chans=19, n_outputs=n_outputs, sfreq=200,
                    input_window_seconds=4.0)
    sd = load_file(WEIGHTS)
    model_sd = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in model_sd and model_sd[k].shape == v.shape}
    model_sd.update(matched)
    model.load_state_dict(model_sd)
    model.to(device)
    print(f"loaded {len(matched)}/{len(model_sd)} pretrained tensors")
    return model


def make_loaders(train_X, train_y, val_X, val_y, batch_size):
    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y).long())
    val_ds = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y).long())
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False))


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
    head_params = [p for n, p in model.named_parameters() if "final_layer" in n]
    body_params = [p for n, p in model.named_parameters() if "final_layer" not in n]
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

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
        if ep % 5 == 0 or ep == 1:
            print(f"  {log_prefix}ep{ep:03d} loss={tot_loss/n:.4f} val_acc={val_acc:.4f}{marker} [{time.time()-t0:.0f}s]", flush=True)
        if bad >= patience:
            print(f"  {log_prefix}early stop at ep{ep}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


def prep_subject_X(sub, use_align):
    """Apply (optional) EA then per-epoch z-score then patch trim to one subject."""
    X = sub["X"]
    if use_align:
        X = euclidean_align(X.astype(np.float64)).astype(np.float32)
    return trim_to_patch(zscore_per_epoch(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-subjects", type=int, default=27)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-exec", type=int, default=1,
                    help="1: add motor-execution epochs to TRAIN (test stays MI only)")
    ap.add_argument("--align", type=int, default=0, help="1: Euclidean Alignment on")
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    print(f"config: use_exec={args.use_exec} align={args.align} tag={args.tag}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mi_subjects = load_mi_subjects()
    exec_data = load_exec_subjects() if args.use_exec else {}
    n_sub = len(mi_subjects)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # leave-N-subjects-out CV over MI subjects (identical split to finetune.py)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_sub)
    fold_size = n_sub // args.folds
    folds = []
    for k in range(args.folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < args.folds - 1 else n_sub
        folds.append(perm[start:end])

    results = []
    for k, test_idx in enumerate(folds):
        print(f"\n=== FOLD {k+1}/{len(folds)} | test subjects: {[mi_subjects[i]['name'] for i in test_idx][:6]}... ===", flush=True)
        test_names = {mi_subjects[i]["name"] for i in test_idx}
        train_subs = [mi_subjects[i] for i in range(n_sub) if i not in test_idx]

        # train/val split of subjects for early stopping (val = MI only, mirrors test domain)
        n_val = max(1, len(train_subs) // 5)
        rng2 = np.random.RandomState(args.seed + k)
        train_perm = rng2.permutation(len(train_subs))
        val_subs = [train_subs[i] for i in train_perm[:n_val]]
        tr_subs = [train_subs[i] for i in train_perm[n_val:]]

        # ---- features ----
        use_align = bool(args.align)

        # TRAIN: MI + (optional) execution from training subjects only
        tr_X_mi = [prep_subject_X(s, use_align) for s in tr_subs]
        tr_y_mi = [s["y"] for s in tr_subs]
        n_mi = sum(len(y) for y in tr_y_mi)

        if exec_data:
            tr_X_ex, tr_y_ex = [], []
            for s in tr_subs:
                if s["name"] in exec_data:
                    Xe, ye = exec_data[s["name"]]
                    sub = {"X": Xe, "y": ye}
                    tr_X_ex.append(prep_subject_X(sub, use_align))
                    tr_y_ex.append(ye)
            n_ex = sum(len(y) for y in tr_y_ex) if tr_y_ex else 0
            train_X = np.concatenate(tr_X_mi + tr_X_ex)
            train_y = np.concatenate(tr_y_mi + tr_y_ex)
            print(f"  train: {n_mi} MI + {n_ex} exec = {len(train_y)} epochs")
        else:
            train_X = np.concatenate(tr_X_mi)
            train_y = np.concatenate(tr_y_mi)
            print(f"  train: {len(train_y)} MI epochs")

        val_X = trim_to_patch(zscore_per_epoch(np.concatenate([s["X"] for s in val_subs])))
        val_y = np.concatenate([s["y"] for s in val_subs])
        test_X = trim_to_patch(zscore_per_epoch(np.concatenate([mi_subjects[i]["X"] for i in test_idx])))
        test_y = np.concatenate([mi_subjects[i]["y"] for i in test_idx])
        if use_align:
            val_X = trim_to_patch(zscore_per_epoch(
                euclidean_align(np.concatenate([s["X"] for s in val_subs]).astype(np.float64)).astype(np.float32)))
            test_X = trim_to_patch(zscore_per_epoch(
                euclidean_align(np.concatenate([mi_subjects[i]["X"] for i in test_idx]).astype(np.float64)).astype(np.float32)))
        print(f"  val: {len(val_y)} | test: {len(test_y)} epochs (MI only)")

        model = build_model(device)
        train_ld, val_ld = make_loaders(train_X, train_y, val_X, val_y, args.batch_size)
        _, test_ld = make_loaders(train_X, train_y, test_X, test_y, args.batch_size)

        best_val = train_one_fold(model, train_ld, val_ld, device, args.epochs, args.lr)

        test_acc, per_class = evaluate(model, test_ld, device)
        print(f"FOLD {k+1}: val={best_val:.4f} test={test_acc:.4f} per_class={np.round(per_class,3).tolist()}", flush=True)
        results.append({"fold": k + 1,
                        "test_subjects": [mi_subjects[i]["name"] for i in test_idx],
                        "val_acc": best_val, "test_acc": test_acc,
                        "per_class": per_class.tolist()})

    accs = [r["test_acc"] for r in results]
    summary = {
        "task": "eegmmidb MI 4-class cross-subject CV (v2: multi-source/EA)",
        "model": "CBraMod finetuned",
        "use_exec": bool(args.use_exec),
        "euclidean_alignment": bool(args.align),
        "n_subjects": n_sub,
        "folds": len(results),
        "mean_test_acc": float(np.mean(accs)),
        "std_test_acc": float(np.std(accs)),
        "chance": 0.25,
        "results": results,
        "args": vars(args),
    }
    out = os.path.join(RESULTS_DIR, f"finetune_v2_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}\n[{args.tag}] MEAN TEST ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f} (chance 0.25)\n{'='*60}")
    print(f"results saved to {out}")


if __name__ == "__main__":
    main()
