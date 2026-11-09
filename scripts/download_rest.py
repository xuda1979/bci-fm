"""Download eegmmidb resting-state runs R01 (eyes open) / R02 (eyes closed).

Neither run was fetched by the earlier download scripts, so all 109 subjects
are missing these files. R01/R02 are 5-minute baseline recordings with no
task annotations — the basis for (a) unsupervised test-time adaptation and
(b) the resting-state pretraining task of the multi-task model.
"""

import concurrent.futures
import os
import sys
import time
import urllib.request

BASE = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0"
RUNS = [1, 2]  # R01 eyes-open rest, R02 eyes-closed rest
SUBJECTS = [f"S{n:03d}" for n in range(1, 110)]


def fetch(url_path, dest, retries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(f"{BASE}/{url_path}", dest)
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"FAIL {url_path}: {e}", flush=True)
                return False
            time.sleep(2 * (attempt + 1))
    return False


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else "data/eegmmidb"
    jobs = []
    for s in SUBJECTS:
        sdir = os.path.join(out_root, s)
        os.makedirs(sdir, exist_ok=True)
        for r in RUNS:
            fn = f"{s}R{r:02d}.edf"
            jobs.append((f"{s}/{fn}", os.path.join(sdir, fn)))

    todo = [(u, d) for u, d in jobs if not os.path.exists(d) or os.path.getsize(d) == 0]
    print(f"{len(jobs)} files total, {len(todo)} to download (~{len(todo)*1.3:.0f}MB)", flush=True)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch, u, d): u for u, d in todo}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 50 == 0:
                el = time.time() - t0
                print(f"{done}/{len(todo)}, {el:.0f}s, {done/el*60:.0f}/min", flush=True)

    fails = [u for u, d in jobs if not os.path.exists(d) or os.path.getsize(d) == 0]
    print(f"COMPLETE: {len(jobs)-len(fails)}/{len(jobs)} ok, {len(fails)} failed")
    for f in fails:
        print("FAIL:", f)


if __name__ == "__main__":
    main()
