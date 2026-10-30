"""Preprocess PhysioNet eegmmidb motor-imagery data for CBraMod finetuning.

Extracts motor-imagery runs (4/8/12 = hands, 6/10/14 = feet) with 4-class labels:
  left hand / right hand / both feet / tongue (annotations T1/T2 per run)

Output: per-subject .npz files with epochs [n_epochs, 64, n_times] + labels.
Also writes 19-channel subset (CBraMod-compatible) and resamples to 200Hz.
"""
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Standard 1020 channel names in eegmmidb (64ch BrainVision system)
ALL_CHS = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.",
    "Fp1.", "Fpz.", "Fp2.", "Af7.", "Af3.", "Afz.", "Af4.", "Af8.",
    "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..",
    "T7..", "T8..", "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..",
    "P6..", "P8..", "Po7.", "Po3.", "Poz.", "Po4.", "Po8.", "Iz..",
    "Oz..", "O1..", "O2..", "Ex1..", "Ex2..", "Ex3..", "Ex4..",
]

# 19 channels CBraMod was pretrained on (standard 10-20)
CBRAMOD_19 = ["Fp1.", "Fp2.", "F7..", "F3..", "Fz..", "F4..", "F8..",
              "T7..", "C3..", "Cz..", "C4..", "T8..",
              "P7..", "P3..", "Pz..", "P4..", "P8..",
              "O1..", "O2.."]

# run -> (T1 label, T2 label) for motor imagery runs
MI_RUNS = {
    4: ("left_hand", "right_hand"),
    8: ("left_hand", "right_hand"),
    12: ("left_hand", "right_hand"),
    6: ("feet", "tongue"),
    10: ("feet", "tongue"),
    14: ("feet", "tongue"),
}

LABELS = ["left_hand", "right_hand", "feet", "tongue"]
LABEL_IDX = {l: i for i, l in enumerate(LABELS)}

TARGET_SFREQ = 200
EPOCH_TMIN, EPOCH_TMAX = 0.0, 4.0  # 4s imagery window per trial


def process_subject(subject_dir, out_dir, chs, tag):
    import mne

    subject = os.path.basename(subject_dir)
    all_X, all_y, all_runs = [], [], []

    for run_num, (t1_label, t2_label) in MI_RUNS.items():
        edf = os.path.join(subject_dir, f"{subject}R{run_num:02d}.edf")
        if not os.path.exists(edf):
            print(f"  missing {edf}, skipping", flush=True)
            continue
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        # MNE lowercases names with trailing dots: 'Fc5.', 'C3..' -> strip dots
        raw = raw.rename_channels({c: c.rstrip(".") for c in raw.ch_names})
        picks = [c.rstrip(".") for c in chs if c.rstrip(".") in raw.ch_names]
        if len(picks) < len(chs):
            print(f"  {subject} R{run_num}: only {len(picks)}/{len(chs)} chs found", flush=True)
        raw.pick(picks)

        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        # map annotation ids to our 4 labels: T1 -> first, T2 -> second
        name_to_label = {"T1": t1_label, "T2": t2_label}
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

    # resample to target sfreq
    n_times_orig = X.shape[-1]
    sfreq_orig = int(round(epochs.info["sfreq"]))
    if sfreq_orig != TARGET_SFREQ:
        X = resample_epochs(X, sfreq_orig, TARGET_SFREQ)

    out = os.path.join(out_dir, f"{subject}_{tag}.npz")
    np.savez_compressed(out, X=X, y=y, runs=runs,
                        chs=np.array(chs), sfreq=TARGET_SFREQ, labels=np.array(LABELS))
    counts = {LABELS[i]: int((y == i).sum()) for i in range(4)}
    print(f"  {subject}: {len(y)} epochs, shape {X.shape}, dist {counts}", flush=True)
    return True


def resample_epochs(X, sfreq_old, sfreq_new):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sfreq_new, sfreq_old)
    up, down = sfreq_new // g, sfreq_old // g
    # X: [n, ch, t] -> resample along time axis
    return resample_poly(X, up, down, axis=-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/eegmmidb")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--all-channels", action="store_true",
                    help="keep all 64 channels instead of 19-ch CBraMod subset")
    args = ap.parse_args()

    chs = ALL_CHS if args.all_channels else CBRAMOD_19
    tag = "64ch" if args.all_channels else "19ch"
    os.makedirs(args.out_dir, exist_ok=True)

    done = 0
    for n in range(1, args.subjects + 1):
        sdir = os.path.join(args.data_dir, f"S{n:03d}")
        if not os.path.isdir(sdir):
            print(f"{sdir} not found, skip", flush=True)
            continue
        print(f"processing S{n:03d}...", flush=True)
        if process_subject(sdir, args.out_dir, chs, tag):
            done += 1

    print(f"\nDONE: {done} subjects processed into {args.out_dir}")


if __name__ == "__main__":
    main()
