"""Download selected tar members (by byte range) into an output folder, in parallel.
usage: fetch_members.py index.json wanted.txt outdir nthreads"""
import http.client
import json
import os
import queue
import sys
import threading
import time

HOST = "fb-ctrl-oss.s3.amazonaws.com"
PATH = "/emg2pose/emg2pose_dataset.tar"
idx_path, wanted_path, outdir, nthreads = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
idx = {os.path.basename(e["name"]): e for e in json.load(open(idx_path))}
wanted = [l.strip() for l in open(wanted_path) if l.strip()]
q = queue.Queue()
for w in wanted:
    q.put(w)
done_bytes = [0]
lock = threading.Lock()
t0 = time.time()


def worker():
    conn = None
    while True:
        try:
            name = q.get_nowait()
        except queue.Empty:
            return
        e = idx[name]
        dst = os.path.join(outdir, name)
        if os.path.exists(dst) and os.path.getsize(dst) == e["size"]:
            continue
        start = e["offset"] + 512
        end = start + e["size"] - 1
        for attempt in range(6):
            try:
                if conn is None:
                    conn = http.client.HTTPSConnection(HOST, timeout=120)
                conn.request("GET", PATH, headers={"Range": f"bytes={start}-{end}"})
                r = conn.getresponse()
                if r.status != 206:
                    raise IOError(r.status)
                tmp = dst + ".part"
                with open(tmp, "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                if os.path.getsize(tmp) != e["size"]:
                    raise IOError("short read")
                os.replace(tmp, dst)
                with lock:
                    done_bytes[0] += e["size"]
                    gb = done_bytes[0] / 1e9
                    print(f"{name} ok  total {gb:.2f} GB  {gb * 1e3 / (time.time() - t0):.1f} MB/s", flush=True)
                break
            except Exception as ex:  # noqa
                conn = None
                print(f"retry {name}: {ex}", flush=True)
                time.sleep(2 + 2 * attempt)


ths = [threading.Thread(target=worker) for _ in range(nthreads)]
for t in ths:
    t.start()
for t in ths:
    t.join()
print(f"done {done_bytes[0] / 1e9:.2f} GB in {time.time() - t0:.0f}s")
