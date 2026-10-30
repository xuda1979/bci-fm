"""Download missing eegmmidb motor-EXECUTION runs (R03/05/07/09/11/13) for all 109 subjects.

Only the first 20 subjects were fully downloaded by download_s3.py; the other 89 have
just the MI runs (04/06/08/10/12/14). This fetches the execution runs from the same
public S3 bucket so we can use execution data as auxiliary training signal.

Usage:
  python3 scripts/download_exec.py [--workers 8]
"""
import argparse
import concurrent.futures
import os
import time
import urllib.request

BASE = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0"
RUNS = [3, 5, 7, 9, 11, 13]
ROOT = "data/eegmmidb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = []
    for s in sorted(os.listdir(ROOT)):
        sdir = os.path.join(ROOT, s)
        if not os.path.isdir(sdir) or not s.startswith("S"):
            continue
        have = set(int(f.split("R")[-1].split(".")[0])
                   for f in os.listdir(sdir) if f.endswith(".edf"))
        for r in RUNS:
            if r in have:
                continue
            fn = f"{s}R{r:02d}.edf"
            url = f"{BASE}/{s}/{fn}"
            dst = os.path.join(sdir, fn)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                continue
            jobs.append((url, dst))

    print(f"{len(jobs)} files to fetch", flush=True)
    if not jobs:
        return

    def fetch(url, dst):
        try:
            urllib.request.urlretrieve(url, dst)
            return None
        except Exception as e:
            return url

    t0 = time.time()
    done, fails = 0, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, u, d): u for u, d in jobs}
        for fut in concurrent.futures.as_completed(futs):
            err = fut.result()
            done += 1
            if err:
                fails.append(err)
            if done % 40 == 0:
                el = time.time() - t0
                print(f"{done}/{len(jobs)}, {el:.0f}s, {done/el*60:.0f} files/min", flush=True)

    print(f"\nDONE: {len(jobs)-len(fails)}/{len(jobs)} fetched in {time.time()-t0:.0f}s")
    for f in fails:
        print("FAIL:", f)


if __name__ == "__main__":
    main()
