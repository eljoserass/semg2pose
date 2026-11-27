"""Build an index (name, header offset, size) of the remote emg2pose tar by walking headers
with HTTP range requests from several sync points in parallel."""
import http.client
import json
import sys
import threading
import time

HOST = "fb-ctrl-oss.s3.amazonaws.com"
PATH = "/emg2pose/emg2pose_dataset.tar"
TOTAL = 462824048640
OUT = sys.argv[1]
NSEG = int(sys.argv[2]) if len(sys.argv) > 2 else 48


class Conn:
    def __init__(self):
        self.c = None

    def get(self, start, end):
        for attempt in range(6):
            try:
                if self.c is None:
                    self.c = http.client.HTTPSConnection(HOST, timeout=60)
                self.c.request("GET", PATH, headers={"Range": f"bytes={start}-{end}"})
                r = self.c.getresponse()
                data = r.read()
                if r.status not in (200, 206):
                    raise IOError(f"status {r.status}")
                return data
            except Exception as e:  # noqa
                self.c = None
                time.sleep(1 + attempt)
        raise IOError("failed")


def parse_header(b):
    if len(b) < 512 or b[257:262] != b"ustar":
        return None
    chk = b[148:156].split(b"\0")[0].strip()
    try:
        chk = int(chk, 8)
    except ValueError:
        return None
    s = sum(b[:148]) + 8 * 32 + sum(b[156:512])
    if s != chk:
        return None
    name = b[0:100].split(b"\0")[0].decode()
    prefix = b[345:500].split(b"\0")[0].decode(errors="ignore")
    if prefix and b[257:263] == b"ustar\0":
        name = prefix + "/" + name
    sz = b[124:136]
    if sz[0] & 0x80:  # base-256
        size = int.from_bytes(sz[1:], "big")
    else:
        size = int(sz.split(b"\0")[0].strip() or b"0", 8)
    return name, size, chr(b[156]) if b[156] else "0"


def find_sync(conn, off):
    off = (off // 512) * 512
    chunk = 4 << 20
    while off < TOTAL:
        data = conn.get(off, min(off + chunk, TOTAL) - 1)
        for i in range(0, len(data) - 511, 512):
            h = parse_header(data[i:i + 512])
            if h is not None:
                return off + i
        off += len(data)
    return TOTAL


results = {}


def walk(seg, start, stop):
    conn = Conn()
    pos = find_sync(conn, start) if start > 0 else 0
    entries = []
    longname = None
    while pos < stop and pos < TOTAL:
        b = conn.get(pos, pos + 511)
        h = parse_header(b)
        if h is None:
            if b == b"\0" * 512:
                break
            print(f"seg {seg}: bad header at {pos}", flush=True)
            break
        name, size, typ = h
        if typ == "L":
            longname = conn.get(pos + 512, pos + 512 + size - 1).split(b"\0")[0].decode()
        else:
            if longname:
                name, longname = longname, None
            entries.append((name, pos, size, typ))
        pos += 512 + ((size + 511) // 512) * 512
    results[seg] = (entries, pos)
    print(f"seg {seg} done: {len(entries)} entries", flush=True)


bounds = [int(TOTAL * i / NSEG) for i in range(NSEG + 1)]
threads = []
t0 = time.time()
# each segment walks from its own sync point until it passes the start of the next segment
for s in range(NSEG):
    th = threading.Thread(target=walk, args=(s, bounds[s], bounds[s + 1]))
    th.start()
    threads.append(th)
for th in threads:
    th.join()
allent = {}
for s in range(NSEG):
    for name, pos, size, typ in results[s][0]:
        allent[pos] = (name, size, typ)
idx = [dict(name=v[0], offset=k, size=v[1], type=v[2]) for k, v in sorted(allent.items())]
# continuity check: each header must be followed by the next
gaps = 0
for a, b in zip(idx, idx[1:]):
    if a["offset"] + 512 + ((a["size"] + 511) // 512) * 512 != b["offset"]:
        gaps += 1
print(f"{len(idx)} entries, {gaps} gaps, {time.time() - t0:.0f}s")
json.dump(idx, open(OUT, "w"))
