"""PCGrad variant of the Task-Conditioned Multi-Task (TC-MT) finetuning.

Identical data, model, split protocol, and evaluation to src/mtl.py; only the
gradient combination rule changes:

  naive (mtl.py):     L = L_mi + L_me + L_rest          -> one backward
  PCGrad (this file): per-task gradients g_i computed separately; for every
      conflicting pair (g_i . g_j < 0) g_i is projected onto the normal
      plane of g_j, with a random projection order per step; the projected
      gradients are then summed (Yu et al., NeurIPS 2020, Alg. 1).

Key property: with zero conflicts the PCGrad step is numerically identical
to the naive summed-loss step, so any accuracy difference is attributable
to conflict resolution alone.  Pairwise conflict rates are logged per fold
(the fraction of steps where the raw dot product between two task
gradients is negative).

Extra options:
  --pcgrad 0          fall back to the naive summed loss (in-file A/B path)
  --rest-as-two 1     treat rest EO/EC CE and masked reconstruction as two
                      separate task gradients (4 PCGrad tasks total)

Checkpoints per fold are saved in the same format as mtl.py, so
src/tta_rest.py works unchanged on them (--ckpt-tag <tag>).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import mtl
from mtl import (T_MI, T_ME, T_REST, TASK_NAMES,
                 load_task_data, build_model, make_mtl_loaders, evaluate_mi,
                 RESULTS_DIR, CKPT_DIR)


# ----------------------------------------------------------------------------
# per-task losses
# ----------------------------------------------------------------------------
def compute_task_losses(model, xb, tb, yb, args):
    """List of (task_name, scalar_loss) for the tasks present in this batch."""
    losses = []
    for t in (T_MI, T_ME):
        m = tb == t
        if m.any():
            logits = model(xb[m], t)
            losses.append((TASK_NAMES[t], F.cross_entropy(logits, yb[m])))

    m = tb == T_REST
    if m.any():
        logits, rec, r = model.forward_rest(xb[m], args.mask_ratio)
        ce = F.cross_entropy(logits, yb[m])
        B, C, P, S = rec.shape
        mask = (torch.rand(B, 1, P, 1, device=xb.device) < args.mask_ratio).float()
        rec_loss = (F.mse_loss(rec * mask, r * mask, reduction="sum")
                    / mask.sum().clamp(min=1) / S)
        if args.rest_as_two:
            losses.append(("rest_ce", ce))
            losses.append(("rec", args.recon_weight * rec_loss))
        else:
            losses.append(("rest", ce + args.recon_weight * rec_loss))
    return losses


# ----------------------------------------------------------------------------
# PCGrad
# ----------------------------------------------------------------------------
def flat_dot(gs, hs):
    return sum(torch.sum(a * b) for a, b in zip(gs, hs))


def pcgrad_project(grads, names, stats):
    """Project conflicting task gradients onto each other's normal planes.

    Random projection order per step (Yu et al., 2020).  Records raw
    pairwise conflict counts into `stats` (key -> {steps, conflicts}).
    """
    n = len(grads)
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{names[i]}-{names[j]}"
            d = stats.setdefault(key, {"steps": 0, "conflicts": 0})
            d["steps"] += 1
            if flat_dot(grads[i], grads[j]).item() < 0:
                d["conflicts"] += 1

    order = torch.randperm(n).tolist()
    proj = [list(g) for g in grads]
    for i in order:
        for j in order:
            if i == j:
                continue
            gi, gj = proj[i], proj[j]
            dot = flat_dot(gi, gj)
            if dot.item() < 0:
                nrm = flat_dot(gj, gj).clamp(min=1e-12)
                coef = dot / nrm
                proj[i] = [a - coef * b for a, b in zip(gi, gj)]
    return proj


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def train_one_fold(model, train_ld, val_ld, device, epochs, lr, args):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc, best_state, patience, bad = 0.0, None, 8, 0
    t0 = time.time()
    conflict_stats = {}

    for ep in range(1, epochs + 1):
        model.train()
        tot = {"loss": 0.0, "mi": 0.0, "me": 0.0, "rest": 0.0, "rec": 0.0}
        n = 0
        for xb, tb, yb in train_ld:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            losses = compute_task_losses(model, xb, tb, yb, args)
            if not losses:
                continue

            if args.pcgrad:
                gs, names = [], []
                for name, loss in losses:
                    g = torch.autograd.grad(loss, params, allow_unused=True)
                    gs.append([gi if gi is not None else torch.zeros_like(p)
                               for gi, p in zip(g, params)])
                    names.append(name)
                proj = pcgrad_project(gs, names, conflict_stats)
                total = [sum(pg[k] for pg in proj) for k in range(len(params))]
                for p, g in zip(params, total):
                    p.grad = g
                loss_val = float(sum(l.item() for _, l in losses))
            else:
                opt.zero_grad()
                loss = sum(l for _, l in losses)
                loss.backward()
                loss_val = loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = xb.shape[0]
            tot["loss"] += loss_val * bs
            for name, l in losses:
                key = {"mi": "mi", "me": "me", "rest": "rest",
                       "rest_ce": "rest", "rec": "rec"}[name]
                tot[key] += l.item() * bs
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
            print(f"  early stop at ep{ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc, conflict_stats


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
    ap.add_argument("--rest-per-cond", type=int, default=20)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--recon-weight", type=float, default=0.5)
    ap.add_argument("--no-task-tokens", type=int, default=0)
    ap.add_argument("--pcgrad", type=int, default=1,
                    help="1: PCGrad gradient surgery; 0: naive summed loss")
    ap.add_argument("--rest-as-two", type=int, default=0,
                    help="split rest CE and reconstruction into 2 task grads")
    ap.add_argument("--tag", default="mtl_pcgrad")
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

    # leave-N-subjects-out CV (identical split protocol to mtl.py / finetune_v2)
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
            with torch.no_grad():
                model.task_emb.zero_()
            model.task_emb.requires_grad_(False)

        train_ld, val_ld = make_mtl_loaders(train_X, train_t, train_y, val_X, val_y, args.batch_size)
        _, test_ld = make_mtl_loaders(train_X, train_t, train_y, test_X, test_y, args.batch_size)

        best_val, conflict_stats = train_one_fold(model, train_ld, val_ld, device, args.epochs, args.lr, args)

        test_acc, per_class = evaluate_mi(model, test_ld, device)
        print(f"FOLD {k+1}: val={best_val:.4f} test={test_acc:.4f} per_class={np.round(per_class,3).tolist()}", flush=True)
        for key, d in conflict_stats.items():
            print(f"  conflict {key}: {d['conflicts']}/{d['steps']} = {d['conflicts']/max(d['steps'],1):.1%}")

        # save fold checkpoint for TTA (same format as mtl.py)
        ck = os.path.join(CKPT_DIR, f"{args.tag}_fold{k+1}.pt")
        torch.save({"state_dict": model.state_dict(), "config": vars(args)}, ck)

        results.append({"fold": k + 1,
                        "test_subjects": [names[i] for i in test_idx],
                        "val_acc": best_val, "test_acc": test_acc,
                        "per_class": per_class.tolist(),
                        "conflict_rate": {key: d["conflicts"] / max(d["steps"], 1)
                                          for key, d in conflict_stats.items()}})

    accs = [r["test_acc"] for r in results]
    summary = {
        "task": "eegmmidb MI 4-class cross-subject CV (task-conditioned multi-task)",
        "model": ("CBraMod + zero-init task tokens + task heads (mi/me/rest+recon), "
                  f"PCGrad={'on' if args.pcgrad else 'off'}"
                  + (", rest split into CE+rec task grads" if args.rest_as_two else "")),
        "config": vars(args),
        "n_subjects": n_sub,
        "folds": len(results),
        "mean_test_acc": float(np.mean(accs)),
        "std_test_acc": float(np.std(accs)),
        "chance": 0.25,
        "mean_conflict_rate": {
            key: float(np.mean([r["conflict_rate"][key] for r in results if key in r["conflict_rate"]]))
            for key in (results[0]["conflict_rate"] if results else {})},
        "fold_results": results,
    }
    out = os.path.join(RESULTS_DIR, f"finetune_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}\n[{args.tag}] MEAN TEST ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f} (chance 0.25)\n{'='*60}")
    print(f"results saved to {out}")


if __name__ == "__main__":
    main()
