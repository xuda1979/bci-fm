import concurrent.futures, urllib.request, os, sys, time

BASE = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0"
RUNS = [f"R{r:02d}" for r in [3,4,5,6,7,8,9,10,11,12,13,14]]  # task runs 3-14
SUBJECTS = [f"S{n:03d}" for n in range(1, 21)]  # first 20 subjects

def fetch(url_path, dest, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{BASE}/{url_path}", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk: break
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"FAIL {url_path}: {e}", flush=True)
                return False
            time.sleep(2 * (attempt + 1))

total, done = 0, 0
jobs = []
for s in SUBJECTS:
    os.makedirs(s, exist_ok=True)
    for r in RUNS:
        fn = f"{s}{r}.edf"
        jobs.append((f"{s}/{fn}", os.path.join(s, fn)))

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch, u, d): u for u, d in jobs}
    for fut in concurrent.futures.as_completed(futs):
        done += 1
        if done % 20 == 0:
            el = time.time() - t0
            print(f"{done}/{len(jobs)} files, {el:.0f}s elapsed, {done/el*60:.0f} files/min", flush=True)

fails = [u for u, d in jobs if not os.path.exists(d) or os.path.getsize(d) == 0]
print(f"COMPLETE: {len(jobs)-len(fails)}/{len(jobs)} succeeded, {len(fails)} failed")
if fails: print("Failed:", fails[:10])
