"""Pick one idle H200 following the user's rule (NOTES D77): prefer GPU 2; if busy, poll every 10 min;
after 30 min any idle H200 may be used. Never returns a busy GPU. Prints the physical index to stdout.

Idle = utilisation <= 5 % and memory used by all processes <= 1500 MiB (another user's idle 528 MiB
context is present on every GPU of this node).
"""
import argparse
import subprocess
import sys
import time


def snapshot():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [{"index": int(i), "util": int(u), "mem": int(m)} for i, u, m in (l.split(",") for l in out.strip().splitlines())]


def idle(g, max_util=5, max_mem=1500):
    return g["util"] <= max_util and g["mem"] <= max_mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer", type=int, default=2)
    ap.add_argument("--poll", type=int, default=600)
    ap.add_argument("--switch-after", type=int, default=1800)
    ap.add_argument("--max-wait", type=int, default=8 * 3600)
    a = ap.parse_args()
    t0 = time.time()
    while time.time() - t0 < a.max_wait:
        # two readings 5 s apart: utilisation counters need a moment to settle after our own previous run
        snap = snapshot()
        pref = next(g for g in snap if g["index"] == a.prefer)
        if not idle(pref):
            time.sleep(5)
            snap = snapshot()
            pref = next(g for g in snap if g["index"] == a.prefer)
        if idle(pref):
            print(a.prefer); return
        waited = time.time() - t0
        if waited >= a.switch_after:
            others = [g for g in snap if g["index"] != a.prefer and idle(g)]
            if others:
                print(f"GPU {a.prefer} busy for {waited/60:.0f} min; switching to idle GPU {others[0]['index']}", file=sys.stderr)
                print(others[0]["index"]); return
        print(f"{time.strftime('%H:%M:%S')} GPU {a.prefer} busy ({pref}); waiting {a.poll}s", file=sys.stderr)
        time.sleep(a.poll)
    print("no idle GPU within max-wait", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
