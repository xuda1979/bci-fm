"""Test-time adaptation (TTA) for TC-MT models using unlabeled resting-state
EEG of the test subject.

Motivation
----------
In cross-subject BCI the model never sees the test subject during training.
However, a real deployment *always* has access to a few minutes of
unlabeled resting-state EEG from the new user (calibration-free protocols
start from eyes-open / eyes-closed rest anyway). The TC-MT model was
trained with a rest head (EO/EC classification) and a masked-reconstruction
objective on rest data, so both objectives can serve as *self-supervised
adaptation signals* on the test subject's own rest recordings:

  1. rest classification (EO vs EC)        -- a surrogate task with known labels
  2. masked patch reconstruction (MAE-style) -- purely self-supervised

We adapt *all* parameters with a low learning rate for a few epochs, then
evaluate MI accuracy on the (labeled) test set. Because adaptation uses
only rest data + labels that come for free (EO/EC), this is a realistic
calibration-free / label-free TTA protocol.

Usage
-----
  python src/tta_rest.py --ckpt models/mtl/mtl_full_fold1.pt --fold 1 \
      --methods none rest_ce rest_ce_rec rec_only --tag tta_fold1

Reproduces the exact fold split of mtl.py (same seed & permutation), so the
test subjects match the training run.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import mtl
from mtl import (T_MI, T_REST, zscore_per_epoch, trim_to_patch,
                 euclidean_align, evaluate_mi, TaskConditionedCBraMod,
                 DATA_DIR, CKPT_DIR, WEIGHTS)

RESULTS_DIR = "results"


# ----------------------------------------------------------------------------
# rest-only test-subject dataset
# ----------------------------------------------------------------------------
def load_subject_rest(names, align=False):
    """Return {name: (X_rest, y_rest)} with EO=0 / EC=1 labels."""
    out = {}
    for name in names:
        Xs, ys = [], []
        for cond, lab in (("eo", 0), ("ec", 1)):
            p = os.path.join(DATA_DIR, f"{name}_rest_{cond}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            # match mtl.py prep(): EA (float64) -> zscore -> trim
            if align:
                X = euclidean_align(d["X"].astype(np.float64)).astype(np.float32)
            else:
                X = d["X"].astype(np.float32)
            X = trim_to_patch(zscore_per_epoch(X))
            Xs.append(X)
            ys.append(np.full(len(X), lab, dtype=np.int64))
        if Xs:
            out[name] = (np.concatenate(Xs), np.concatenate(ys))
    return out


def load_subject_mi(names, align=False):
    out = {}
    for name in names:
        p = os.path.join(DATA_DIR, f"{name}_19ch.npz")
        d = np.load(p)
        # match mtl.py prep(): EA (float64) -> zscore -> trim
        if align:
            X = euclidean_align(d["X"].astype(np.float64)).astype(np.float32)
        else:
            X = d["X"].astype(np.float32)
        X = trim_to_patch(zscore_per_epoch(X))
        out[name] = (X, d["y"].astype(np.int64))
    return out


# ----------------------------------------------------------------------------
# TTA methods
# ----------------------------------------------------------------------------
@torch.no_grad()
def base_eval(model, mi_loader, device):
    acc, per_class = evaluate_mi(model, mi_loader, device)
    return acc


def adapt_rest(model, rest_loader, device, epochs, lr, method,
               mask_ratio=0.5, recon_weight=1.0, backbone_only=False):
    """Adapt model on test-subject rest data.

    method:
      "rest_ce"     : EO/EC cross-entropy only
      "rec_only"    : masked reconstruction only
      "rest_ce_rec" : both
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        model.train()
        for xb, yb in rest_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = xb.new_zeros(())
            if "rest_ce" in method:
                logits, _, _ = model.forward_rest(xb, mask_ratio)
                loss = loss + F.cross_entropy(logits, yb)
            if "rec" in method:
                logits, rec, r = model.forward_rest(xb, mask_ratio)
                B, C, P, S = rec.shape
                mask = (torch.rand(B, 1, P, 1, device=device) < mask_ratio).float()
                rec_loss = (F.mse_loss(rec * mask, r * mask, reduction="sum")
                            / (mask.sum() * S).clamp(min=1))
                loss = loss + recon_weight * rec_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--test-subjects", type=int, default=27)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--ckpt-tag", type=str, default="mtl_full",
                    help="checkpoint tag from mtl.py training run")
    ap.add_argument("--methods", type=str,
                    default="none,rest_ce,rec_only,rest_ce_rec")
    ap.add_argument("--tta-epochs", type=int, default=10)
    drift = ap.add_mutually_exclusive_group()
    drift.add_argument("--tta-lr", type=float, default=None)
    drift.add_argument("--tta-lrs", type=str, default="3e-5,1e-5,3e-6",
                       help="comma list; best on val rest acc is picked per fold")
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tag", type=str, default="tta_rest")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    # ---- reproduce the exact fold split of mtl.py ----
    sub_names = sorted(n.split("_")[0] for n in os.listdir(DATA_DIR)
                       if n.endswith("_19ch.npz"))
    sub_names = sorted(set(sub_names))
    n_sub = len(sub_names)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_sub)
    fold_size = args.test_subjects
    folds = [perm[k * fold_size:(k + 1) * fold_size] for k in range(args.folds)]

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    lrs = ([args.tta_lr] if args.tta_lr is not None
           else [float(x) for x in args.tta_lrs.split(",")])

    all_results = []
    for k, test_idx in enumerate(folds):
        names = [sub_names[i] for i in test_idx]
        print(f"\n=== FOLD {k+1}/{len(folds)} | test: {names[:5]}... ===", flush=True)

        ck_path = os.path.join(CKPT_DIR, f"{args.ckpt_tag}_fold{k+1}.pt")
        if not os.path.exists(ck_path):
            print(f"  !! checkpoint missing: {ck_path}, skipping fold")
            continue

        # test-subject MI data
        mi = load_subject_mi(names, align=args.align)
        mi_X = np.concatenate([mi[n][0] for n in names])
        mi_y = np.concatenate([mi[n][1] for n in names])
        mi_loader = DataLoader(TensorDataset(torch.from_numpy(mi_X),
                                             torch.from_numpy(mi_y)),
                               batch_size=64)

        # test-subject rest data (adaptation + internal val for lr picking)
        rest = load_subject_rest(names, align=args.align)
        rest_avail = [n for n in names if n in rest]
        print(f"  rest available: {len(rest_avail)}/{len(names)} subjects")

        fold_res = {"fold": k + 1, "methods": {}}

        for method in methods:
            t0 = time.time()
            # fresh copy of trained weights per method
            model = TaskConditionedCBraMod()
            sd = torch.load(ck_path, map_location="cpu")["state_dict"]
            model.load_state_dict(sd)
            model.to(device)

            if method == "none":
                acc, per_class = evaluate_mi(model, mi_loader, device)
            else:
                # split rest: 80% adapt, 20% held-out for lr selection
                Xs, ys = [], []
                for n in rest_avail:
                    X, y = rest[n]
                    Xs.append(X); ys.append(y)
                rest_X = np.concatenate(Xs); rest_y = np.concatenate(ys)
                n_val = max(4, len(rest_y) // 5)
                r2 = np.random.RandomState(args.seed + 100 + k)
                vp = r2.permutation(len(rest_y))
                val_X, val_y = rest_X[vp[:n_val]], rest_y[vp[:n_val]]
                ad_X, ad_y = rest_X[vp[n_val:]], rest_y[vp[n_val:]]
                ad_loader = DataLoader(TensorDataset(torch.from_numpy(ad_X),
                                                     torch.from_numpy(ad_y)),
                                       batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(TensorDataset(torch.from_numpy(val_X),
                                                      torch.from_numpy(val_y)),
                                        batch_size=64)

                best = None
                for lr in lrs:
                    model_t = TaskConditionedCBraMod()
                    model_t.load_state_dict(
                        torch.load(ck_path, map_location="cpu")["state_dict"])
                    model_t.to(device)
                    adapt_rest(model_t, ad_loader, device, args.tta_epochs, lr,
                               method, mask_ratio=args.mask_ratio)
                    # select lr by rest val acc (label-free wrt MI: EO/EC labels
                    # are metadata, not MI labels)
                    racc, _ = rest_val_acc(model_t, val_loader, device)
                    print(f"    [{method}] lr={lr:g} rest_val={racc:.4f}", flush=True)
                    if best is None or racc > best[0]:
                        best = (racc, lr)
                    del model_t
                _, lr = best
                model_t = TaskConditionedCBraMod()
                model_t.load_state_dict(
                    torch.load(ck_path, map_location="cpu")["state_dict"])
                model_t.to(device)
                adapt_rest(model_t, ad_loader, device, args.tta_epochs, lr,
                           method, mask_ratio=args.mask_ratio)
                acc, per_class = evaluate_mi(model_t, mi_loader, device)
                model = model_t

            fold_res["methods"][method] = {
                "test_acc": acc, "per_class": per_class.tolist(),
                "time_s": round(time.time() - t0, 1)}
            print(f"  [{method}] MI test acc = {acc:.4f} [{time.time()-t0:.0f}s]",
                  flush=True)
            del model

        all_results.append(fold_res)

    # ---- summary ----
    summary = {"task": "TC-MT + rest-based test-time adaptation (MI 4-class)",
               "ckpt_tag": args.ckpt_tag, "tta_epochs": args.tta_epochs,
               "lrs_tried": lrs, "align": args.align,
               "folds": args.folds, "chance": 0.25, "results": all_results}
    for method in methods:
        accs = [r["methods"][method]["test_acc"] for r in all_results
                if method in r["methods"]]
        if accs:
            summary[f"mean_{method}"] = float(np.mean(accs))
            summary[f"std_{method}"] = float(np.std(accs))
            print(f"[{method}] MEAN: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{args.tag}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"results saved to {out}")


def rest_val_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, T_REST).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1), None


if __name__ == "__main__":
    main()
