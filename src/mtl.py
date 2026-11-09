"""Task-Conditioned Multi-Task (TC-MT) finetuning of CBraMod.

One backbone, three data types, three task objectives:

  task 0  motor imagery (MI)   4-class CE           (data/processed/S###_19ch.npz)
  task 1  motor execution (ME) 4-class CE           (data/processed/S###_exec19ch.npz)
  task 2  resting state        EO/EC binary CE + masked patch reconstruction
                                                      (data/processed/S###_rest_eo/ec.npz)

Core mechanism: a learnable task token is added to every patch embedding
AFTER patch embedding and BEFORE the transformer encoder:

    pe = patch_embedding(x)            # [B, ch, n_patch, d]
    pe = pe + task_emb[t]              # zero-init: initial forward == pretrained

Zero initialization guarantees the conditioned model starts exactly at the
pretrained CBraMod solution for every task (no-regret conditioning), then
task-specific routing is learned through the shared encoder.

Heads are task-specific (mi_head, me_head, rest_head); the reconstruction
head (patch -> raw 200-sample time series, CBraMod pretraining objective)
is shared across rest batches.

Evaluation: leave-N-subjects-out CV on MI (identical split protocol to
finetune.py / finetune_v2.py). Checkpoints per fold saved for TTA.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from braindecode.models import CBraMod
from safetensors.torch import load_file

DATA_DIR = "data/processed"
RESULTS_DIR = "results"
CKPT_DIR = "models/mtl"
WEIGHTS = "models/model.safetensors"

LABELS = ["left_hand", "right_hand", "both_fists", "both_feet"]

# task ids
T_MI, T_ME, T_REST = 0, 1, 2
TASK_NAMES = ["mi", "me", "rest"]

N_CH = 19
PATCH = 200
SFREQ = 200
FLAT_DIM = 19 * 4 * 200  # ch * n_patch * emb


# ----------------------------------------------------------------------------
# data helpers (identical normalization protocol to finetune_v2.py)
# ----------------------------------------------------------------------------
def zscore_per_epoch(X):
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-6
    return (X - mu) / sd


def trim_to_patch(X, patch=PATCH):
    n_times = X.shape[-1]
    usable = (n_times // patch) * patch
    return X[..., :usable]


def euclidean_align(X):
    n, ch, t = X.shape
    R = np.einsum("nct,nst->cs", X, X) / (n * t)
    R += 1e-8 * np.eye(ch)
    w, V = np.linalg.eigh(R)
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    return np.einsum("cs,nst->nct", W, X)


def load_task_data(align, rest_per_cond, seed):
    """Returns dict name -> {task: (X, y)}; rest labels: 0=EO, 1=EC."""
    rng = np.random.RandomState(seed)
    subs = {}

    def prep(X):
        if align:
            X = euclidean_align(X.astype(np.float64)).astype(np.float32)
        return trim_to_patch(zscore_per_epoch(X))

    for fn in sorted(f for f in os.listdir(DATA_DIR) if f.endswith("_19ch.npz")):
        name = fn.split("_")[0]
        d = np.load(os.path.join(DATA_DIR, fn))
        subs.setdefault(name, {})[T_MI] = (prep(d["X"]).astype(np.float32), d["y"].astype(np.int64))

    for fn in sorted(f for f in os.listdir(DATA_DIR) if f.endswith("_exec19ch.npz")):
        name = fn.split("_")[0]
        d = np.load(os.path.join(DATA_DIR, fn))
        subs.setdefault(name, {})[T_ME] = (prep(d["X"]).astype(np.float32), d["y"].astype(np.int64))

    for name in [n for n in subs]:
        rest = []
        for cond, lab in (("eo", 0), ("ec", 1)):
            p = os.path.join(DATA_DIR, f"{name}_rest_{cond}.npz")
            if os.path.exists(p):
                d = np.load(p)
                X = prep(d["X"]).astype(np.float32)
                idx = rng.permutation(len(X))[:rest_per_cond]
                rest.append((X[idx], np.full(len(idx), lab, dtype=np.int64)))
        if len(rest) == 2 and all(len(x) for x, _ in rest):
            X = np.concatenate([x for x, _ in rest])
            y = np.concatenate([y for _, y in rest])
            subs[name][T_REST] = (X, y)
    return subs


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class TaskConditionedCBraMod(nn.Module):
    """CBraMod backbone + zero-init task tokens + task-specific heads."""

    def __init__(self, n_mi_classes=4, n_me_classes=4, n_rest_classes=2,
                 emb_dim=200, n_patch=4, n_ch=N_CH):
        super().__init__()
        self.backbone = CBraMod(n_chans=n_ch, n_outputs=n_mi_classes,
                                sfreq=SFREQ, input_window_seconds=4.0)
        d = emb_dim

        # zero-init task tokens: [n_tasks, d] -> initial forward == pretrained
        self.task_emb = nn.Parameter(torch.zeros(3, d))
        nn.init.normal_(self.task_emb, std=1e-4)

        flat = n_ch * n_patch * d
        self.mi_head = nn.Linear(flat, n_mi_classes)
        self.me_head = nn.Linear(flat, n_me_classes)
        self.rest_head = nn.Linear(flat, n_rest_classes)
        # reconstruction head: token (d) -> raw patch (200 samples), per token
        self.recon_head = nn.Linear(d, PATCH)
        self.n_patch = n_patch
        self.n_ch = n_ch

    def load_pretrained(self, path=WEIGHTS):
        # materialize LazyLinear in final_layer with a dummy forward so its
        # params exist (state_dict otherwise contains uninitialized params)
        with torch.no_grad():
            self.backbone(torch.zeros(1, N_CH, 4 * PATCH))
        sd = load_file(path)
        bsd = self.backbone.state_dict()
        matched = {k: v for k, v in sd.items()
                   if k in bsd and bsd[k].shape == v.shape}
        bsd.update(matched)
        self.backbone.load_state_dict(bsd)
        # final_layer of the backbone is unused; drop its params from training
        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False
        return len(matched), len(bsd)

    def encode(self, x, task):
        """x: [B, 19, 800] -> backbone features [B, ch, n_patch, d]."""
        r = self.backbone.rearrange(x)                    # [B, ch, n_patch, patch]
        pe = self.backbone.patch_embedding(r)             # [B, ch, n_patch, d]
        pe = pe + self.task_emb[task].view(1, 1, 1, -1)   # task conditioning
        feats = self.backbone.encoder(pe)
        return self.backbone.proj_out(feats), r           # r kept as recon target

    def forward(self, x, task):
        feats, _ = self.encode(x, task)
        flat = feats.flatten(1)
        if task == T_MI:
            return self.mi_head(flat)
        if task == T_ME:
            return self.me_head(flat)
        return self.rest_head(flat)

    def forward_rest(self, x, mask_ratio=0.5):
        """Rest forward: EO/EC logits + masked reconstruction."""
        feats, r = self.encode(x, T_REST)
        flat = feats.flatten(1)
        logits = self.rest_head(flat)
        rec = self.recon_head(feats)                      # [B, ch, n_patch, 200]
        return logits, rec, r


def build_model(device):
    model = TaskConditionedCBraMod()
    n_matched, n_total = model.load_pretrained()
    model.to(device)
    print(f"loaded {n_matched}/{n_total} pretrained tensors into backbone")
    return model


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def make_mtl_loaders(train_X, train_t, train_y, val_X, val_y, batch_size):
    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_t), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
            DataLoader(val_ds, batch_size=batch_size * 2))


def evaluate_mi(model, loader, device):
    model.eval()
    correct = total = 0
    per_class = np.zeros(4)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, T_MI).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
            for c in range(4):
                per_class[c] += ((pred == yb) & (yb == c)).sum().item()
    cls_counts = np.zeros(4)
    for _, yb in loader:
        for c in range(4):
            cls_counts[c] += (yb == c).sum().item()
    per_class = per_class / np.maximum(cls_counts, 1)
    return correct / max(total, 1), per_class


def train_one_fold(model, train_ld, val_ld, device, epochs, lr, args):
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc, best_state, patience, bad = 0.0, None, 8, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = {"loss": 0.0, "ce": 0.0, "rec": 0.0, "mi": 0.0, "me": 0.0, "rest": 0.0}
        n = 0
        for xb, tb, yb in train_ld:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            opt.zero_grad()

            # split batch by task (batch may mix tasks)
            loss = xb.new_zeros(())
            ce_terms, rec_terms = [], []

            for t in (T_MI, T_ME):
                m = tb == t
                if m.any():
                    logits = model(xb[m], t)
                    ce = F.cross_entropy(logits, yb[m])
                    loss = loss + ce
                    ce_terms.append(ce)
                    tot["mi" if t == T_MI else "me"] += ce.item() * int(m.sum())

            m = tb == T_REST
            if m.any():
                logits, rec, r = model.forward_rest(xb[m], args.mask_ratio)
                ce = F.cross_entropy(logits, yb[m])
                # masked reconstruction loss (CBraMod pretraining objective)
                B, C, P, S = rec.shape
                mask = (torch.rand(B, 1, P, 1, device=device) < args.mask_ratio).float()
                rec_loss = (F.mse_loss(rec * mask, r * mask, reduction="sum")
                            / mask.sum().clamp(min=1) / S)
                loss = loss + ce + args.recon_weight * rec_loss
                ce_terms.append(ce)
                rec_terms.append(rec_loss)
                tot["rest"] += ce.item() * int(m.sum())
                tot["rec"] += rec_loss.item() * int(m.sum())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = xb.shape[0]
            tot["loss"] += loss.item() * bs
            if ce_terms:
                tot["ce"] += sum(t.item() for t in ce_terms) * bs / max(len(ce_terms), 1)
            n += bs
        sched.step()

        val_acc, _ = evaluate_mi(model, val_ld, device)
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            marker = " *"
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:03d} loss={tot['loss']/n:.4f} (mi={tot['mi']/n:.3f} me={tot['me']/n:.3f} "
                  f"rest={tot['rest']/n:.3f} rec={tot['rec']/n:.3f}) val_mi={val_acc:.4f}{marker} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        if bad >= patience:
            print(f"  early stop at ep{ep}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-subjects", type=int, default=27)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--align", type=int, default=0)
    ap.add_argument("--use-exec", type=int, default=1)
    ap.add_argument("--use-rest", type=int, default=1)
    ap.add_argument("--rest-per-cond", type=int, default=20,
                    help="rest epochs per condition (EO/EC) per subject in train")
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--recon-weight", type=float, default=0.5)
    ap.add_argument("--no-task-tokens", type=int, default=0,
                    help="ablation: disable task tokens (shared conditioning)")
    ap.add_argument("--tag", default="mtl")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    print(f"config: {vars(args)}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    subs = load_task_data(bool(args.align), args.rest_per_cond, args.seed)
    names = sorted(subs.keys())
    n_sub = len(names)
    n_rest_sub = sum(1 for n in names if T_REST in subs[n])
    print(f"loaded {n_sub} subjects; rest available for {n_rest_sub}")

    # leave-N-subjects-out CV (identical split protocol to finetune_v2)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_sub)
    fold_size = args.test_subjects
    folds = [perm[k * fold_size:(k + 1) * fold_size] for k in range(args.folds)]

    results = []
    for k, test_idx in enumerate(folds):
        print(f"\n=== FOLD {k+1}/{len(folds)} | test: {[names[i] for i in test_idx][:6]}... ===", flush=True)
        train_idx = [i for i in range(n_sub) if i not in set(test_idx.tolist())]

        n_val = max(1, len(train_idx) // 5)
        rng2 = np.random.RandomState(args.seed + k)
        tperm = rng2.permutation(train_idx)
        val_idx, tr_idx = tperm[:n_val], tperm[n_val:]

        # ---- build multi-task train set ----
        Xs, ts, ys = [], [], []
        for i in tr_idx:
            s = subs[names[i]]
            for t in (T_MI, T_ME, T_REST):
                if t == T_ME and not args.use_exec:
                    continue
                if t == T_REST and not args.use_rest:
                    continue
                if t in s:
                    X, y = s[t]
                    Xs.append(X)
                    ts.append(np.full(len(y), t, dtype=np.int64))
                    ys.append(y)
        train_X = np.concatenate(Xs); train_t = np.concatenate(ts); train_y = np.concatenate(ys)
        print(f"  train: {len(train_y)} epochs "
              f"(MI {int((train_t==T_MI).sum())} / ME {int((train_t==T_ME).sum())} / rest {int((train_t==T_REST).sum())})")

        # val = MI only (mirrors test domain)
        val_X = np.concatenate([subs[names[i]][T_MI][0] for i in val_idx])
        val_y = np.concatenate([subs[names[i]][T_MI][1] for i in val_idx])

        test_X = np.concatenate([subs[names[i]][T_MI][0] for i in test_idx])
        test_y = np.concatenate([subs[names[i]][T_MI][1] for i in test_idx])
        print(f"  val: {len(val_y)} | test: {len(test_y)} MI epochs")

        model = build_model(device)
        if args.no_task_tokens:
            # ablation: zero + freeze task tokens => single shared conditioning
            with torch.no_grad():
                model.task_emb.zero_()
            model.task_emb.requires_grad_(False)

        train_ld, val_ld = make_mtl_loaders(train_X, train_t, train_y, val_X, val_y, args.batch_size)
        _, test_ld = make_mtl_loaders(train_X, train_t, train_y, test_X, test_y, args.batch_size)

        best_val = train_one_fold(model, train_ld, val_ld, device, args.epochs, args.lr, args)

        test_acc, per_class = evaluate_mi(model, test_ld, device)
        print(f"FOLD {k+1}: val={best_val:.4f} test={test_acc:.4f} per_class={np.round(per_class,3).tolist()}", flush=True)

        # save fold checkpoint for TTA (rest adaptation needs the trained state)
        ck = os.path.join(CKPT_DIR, f"{args.tag}_fold{k+1}.pt")
        torch.save({"state_dict": model.state_dict(), "config": vars(args)}, ck)

        results.append({"fold": k + 1,
                        "test_subjects": [names[i] for i in test_idx],
                        "val_acc": best_val, "test_acc": test_acc,
                        "per_class": per_class.tolist()})

    accs = [r["test_acc"] for r in results]
    summary = {
        "task": "eegmmidb MI 4-class cross-subject CV (task-conditioned multi-task)",
        "model": "CBraMod + zero-init task tokens + task heads (mi/me/rest+recon)",
        "config": vars(args),
        "n_subjects": n_sub,
        "folds": len(results),
        "mean_test_acc": float(np.mean(accs)),
        "std_test_acc": float(np.std(accs)),
        "chance": 0.25,
        "fold_results": results,
    }
    out = os.path.join(RESULTS_DIR, f"finetune_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}\n[{args.tag}] MEAN TEST ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f} (chance 0.25)\n{'='*60}")
    print(f"results saved to {out}")


if __name__ == "__main__":
    main()
