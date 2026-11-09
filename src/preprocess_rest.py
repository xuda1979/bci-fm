"""Preprocess eegmmidb resting-state runs R01 (eyes open) / R02 (eyes closed).

Each 1-min baseline run is sliced into 4-s trials (15 per run), band-pass
filtered 1-40 Hz and resampled to 200 Hz, with the same 19-channel montage
used for MI/exec (only the first 19 EEG channels of R01/R02 are recorded).
Produces:

  data/processed/S###_rest_eo.npz   X [n, 19, 800] (eyes open)
  data/processed/S###_rest_ec.npz   X [n, 19, 800] (eyes closed)

No labels — these are used for unsupervised test-time adaptation (TTA).
"""
import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

TARGET_SFREQ = 200
KEEP_CH = 19
EPOCH_SEC = 4.0

DATA_DIR = "data/processed"


def load_edf(path):
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    X = raw.get_data()[:KEEP_CH].astype(np.float64) * 1e6  # to µV
    sfreq = raw.info["sfreq"]
    return X, sfreq, raw.ch_names[:KEEP_CH]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--file-list", default="data/processed/rest_files.txt")
    args = ap.parse_args()

    files = [l.strip() for l in open(args.file_list) if l.strip()]
    print(f"{len(files)} resting files to process")

    n_ok, n_fail = 0, 0
    for i, rel in enumerate(files, 1):
        sub = os.path.basename(rel).split("_")[0]  # S001R01.edf -> S001
        run = os.path.basename(rel).split("R")[1].split(".")[0]  # 01 or 02
        tag = "eo" if run == "01" else "ec"
        out = os.path.join(DATA_DIR, f"{sub}_rest_{tag}.npz")
        if os.path.exists(out):
            n_ok += 1
            continue
        try:
            X, sfreq, chs = load_edf(os.path.join(args.data_root, rel))

            # band-pass 1-40 Hz (same as MI/exec preprocessing)
            from mne.filter import filter_data
            X = filter_data(X, sfreq, l_freq=1.0, h_freq=40.0, verbose="ERROR")

            # resample to 200 Hz
            if abs(sfreq - TARGET_SFREQ) > 0.1:
                from mne.filter import resample
                X = resample(X, up=TARGET_SFREQ, down=sfreq, axis=-1)

            # slice into 4-s epochs (no overlap)
            L = int(EPOCH_SEC * TARGET_SFREQ)
            n_ep = X.shape[1] // L
            X = X[:, : n_ep * L].reshape(KEEP_CH, n_ep, L).transpose(1, 0, 2)
            X = X.astype(np.float32)

            np.savez_compressed(out, X=X, chs=np.array(chs), sfreq=np.array(TARGET_SFREQ))
            n_ok += 1
            if i % 10 == 0 or i == len(files):
                print(f"{i}/{len(files)}: {sub} {tag}: {n_ep} rest epochs", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"FAIL {rel}: {e}", flush=True)

    print(f"DONE: {n_ok} ok, {n_fail} failed")


if __name__ == "__main__":
    main()
