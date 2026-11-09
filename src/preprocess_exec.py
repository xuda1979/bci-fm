"""Preprocess eegmmidb motor-EXECUTION runs (3,5,7,9,11,13) into extra training data.

PhysioNet eegmmidb alternates execution/imagery runs:
  R03/07/11 = motor execution: open/close LEFT or RIGHT fist  (T1=left, T2=right)
  R05/09/13 = motor execution: open/close BOTH FISTS or BOTH FEET (T1=fists, T2=feet)
  R04/08/12 = motor imagery:   left/right hand     (T1=left, T2=right)
  R06/10/14 = motor imagery:   both fists/feet     (T1=fists, T2=feet)

Label space is shared with the MI preprocessing (left_hand / right_hand /
both_fists / both_feet), so execution epochs can be mixed into MI training
as same-label auxiliary data. NOTE: earlier versions of this file dropped
the T1 (both-fists) class of R05/09/13 and mislabeled T2 as "feet" in a
4-class space that also (wrongly) contained "tongue"; both are fixed here.

Mapping to the 4-class label space of our MI task:
  execution R03/07/11: T1 -> left_hand, T2 -> right_hand        (direct match)
  execution R05/09/13: T1 -> *dropped* (both fists has no MI analog),
                       T2 -> feet                               (both feet ≈ feet)
  NOTE: "tongue" has no execution analog — these runs simply contribute
  0-2 left/right/feet epochs each, no tongue epochs.

Output schema identical to preprocess.py: per-subject npz with
  X [n, 19, 802], y [n], runs [n], chs, sfreq=200, labels
so downstream finetune.py loads it unchanged (tag: exec).

Usage:
  python3 src/preprocess_exec.py                # all subjects on disk
  python3 src/preprocess_exec.py --quick 4      # first 4 subjects only
"""
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Standard 1020 channel names in eegmmidb (64ch BrainVision system) -- 19ch subset
ALL_CHS = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.",
    "Fp1.", "Fpz.", "Fp2.", "Af7.", "Af3.", "Afz.", "Af4.", "Af8.",
    "F7..", "F3..", "Fz..", "F4..", "F8..",
    "T7..", "T8..", "P7..", "P3..", "Pz..", "P4..", "P8..",
    "O1..", "O2..",
]
SUBSET_19 = ["Fp1.", "Fp2.", "F7..", "F3..", "Fz..", "F4..", "F8..",
             "T7..", "C3..", "Cz..", "C4..", "T8..",
             "P7..", "P3..", "Pz..", "P4..", "P8..",
             "O1..", "O2.."]

# execution run -> (T1 label, T2 label)
EXEC_RUNS = {
    3:  ("left_hand", "right_hand"),
    7:  ("left_hand", "right_hand"),
    11: ("left_hand", "right_hand"),
    5:  ("both_fists", "both_feet"),
    9:  ("both_fists", "both_feet"),
    13: ("both_fists", "both_feet"),
}

LABELS = ["left_hand", "right_hand", "both_fists", "both_feet"]
LABEL_IDX = {l: i for i, l in enumerate(LABELS)}

TARGET_SFREQ = 200
EPOCH_TMIN, EPOCH_TMAX = 0.0, 4.0


def process_subject(subject_dir, out_dir, chs, tag):
    import mne

    subject = os.path.basename(subject_dir)
    all_X, all_y, all_runs = [], [], []

    for run_num, (t1_label, t2_label) in EXEC_RUNS.items():
        edf = os.path.join(subject_dir, f"{subject}R{run_num:02d}.edf")
        if not os.path.exists(edf):
            print(f"  missing {edf}, skipping", flush=True)
            continue
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        raw = raw.rename_channels({c: c.rstrip(".") for c in raw.ch_names})
        picks = [c.rstrip(".") for c in chs if c.rstrip(".") in raw.ch_names]
        raw = raw.pick(picks)
        raw = raw.resample(TARGET_SFREQ)

        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        # map annotation codes to our 4 labels (drop classes with None)
        name_to_label = {}
        if t1_label:
            name_to_label["T1"] = t1_label
        if t2_label:
            name_to_label["T2"] = t2_label
        remap = {}
        for name, eid in event_id.items():
            base = name.replace("ST", "").strip()
            if base in name_to_label:
                remap[eid] = LABEL_IDX[name_to_label[base]]
        if not remap:
            continue

        epochs = mne.Epochs(raw, events, event_id=list(remap.keys()),
                            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=None,
                            preload=True, reject_by_annotation=False, verbose="ERROR")
        if len(epochs) == 0:
            continue

        y = np.array([remap[e] for e in epochs.events[:, 2]])
        all_X.append(epochs.get_data().astype(np.float32))
        all_y.append(y)
        all_runs += [run_num] * len(y)

    if not all_X:
        print(f"  {subject}: no epochs extracted!", flush=True)
        return False

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    runs = np.array(all_runs)

    out = os.path.join(out_dir, f"{subject}_{tag}.npz")
    np.savez_compressed(out, X=X, y=y, runs=runs,
                        chs=np.array(chs), sfreq=TARGET_SFREQ, labels=np.array(LABELS))
    counts = {LABELS[i]: int((y == i).sum()) for i in range(4)}
    print(f"  {subject}: {len(y)} epochs, shape {X.shape}, dist {counts}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/eegmmidb")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--tag", default="exec19ch")
    ap.add_argument("--quick", type=int, default=0, help="only first N subjects (0=all)")
    args = ap.parse_args()

    chs = SUBSET_19
    os.makedirs(args.out_dir, exist_ok=True)

    sdirs = sorted(d for d in (os.path.join(args.in_dir, s)
                               for s in os.listdir(args.in_dir))
                   if os.path.isdir(d) and "S" in os.path.basename(d))
    if args.quick:
        sdirs = sdirs[:args.quick]

    done = 0
    for sdir in sdirs:
        print(f"processing {os.path.basename(sdir)}...", flush=True)
        if process_subject(sdir, args.out_dir, chs, args.tag):
            done += 1

    print(f"\nDONE: {done} subjects processed into {args.out_dir} (tag: {args.tag})")


if __name__ == "__main__":
    main()
