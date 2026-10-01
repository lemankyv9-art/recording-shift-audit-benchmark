# -*- coding: utf-8 -*-
"""带"本地断点续传"的下载器: 服务器忽略 Range 时, 重连后跳过已存前缀继续追加。"""
import os, sys, time, urllib.request

URL = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-1979-latest-published-year.csv"
RAW = "full_download.bin"
TOTAL = 1534937928
CHUNK = 1 << 20

def attempt():
    import time as _t
    have = os.path.getsize(RAW) if os.path.exists(RAW) else 0
    req = urllib.request.Request(URL)
    with urllib.request.urlopen(req, timeout=90) as r, open(RAW, "ab") as f:
        cl = int(r.headers.get("Content-Length") or 0)
        if cl != TOTAL:
            print("WARN content-length", cl, flush=True)
        skip = have
        done = 0
        win_start, win_bytes = _t.time(), 0
        stall_start = None
        while True:
            buf = r.read(CHUNK)
            if not buf:
                break
            if skip > 0:
                skip -= min(skip, len(buf))
                continue  # 跳过阶段不设看门狗(尽快抵达前沿)
            f.write(buf)
            done += len(buf)
            win_bytes += len(buf)
            now = _t.time()
            if now - win_start >= 120:
                rate = win_bytes / (now - win_start) / 1e3
                print(f"rate {rate:.0f}KB/s size={have+done}", flush=True)
                if win_bytes < 6 << 20:  # 追加阶段 <50KB/s -> 掐线换连接
                    raise RuntimeError("throttled connection, reconnect")
                win_start, win_bytes = now, 0
            if done % (100 << 20) < CHUNK:
                print(f"progress {(have+done)/1e6:.0f}MB / {TOTAL/1e6:.0f}MB", flush=True)
        return have + done

for i in range(200):
    try:
        sz = attempt()
        print(f"attempt {i}: size={sz}", flush=True)
        if sz >= TOTAL:
            print("COMPLETE", flush=True)
            break
        print("truncated, will reconnect and continue", flush=True)
    except Exception as e:
        print(f"attempt {i} error: {e!r}", flush=True)
        time.sleep(2)
else:
    sys.exit("too many attempts")
