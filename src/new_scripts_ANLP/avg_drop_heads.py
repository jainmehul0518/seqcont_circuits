#!/usr/bin/env python3
"""
avg_drop_heads.py
Find attention heads whose removal hurts performance by >= THRESHOLD %.

usage:
    python avg_drop_heads.py numerals_alternate_node_prune.txt 50
"""

import re, argparse, json, collections, statistics

def collect(path: str) -> dict[tuple[int, int], list[float]]:
    """Return { (layer,head): [percent, …] } from the log file."""
    head_pat  = re.compile(r'^(\d+)\s+(\d+)$')
    perf_pat  = re.compile(r'\(cand circuit / full\)\s*%:\s*([0-9.]+)')
    curr, store = None, collections.defaultdict(list)

    with open(path) as fh:
        for line in map(str.strip, fh):
            if not line:                 # blank line
                continue
            if m := head_pat.match(line):
                curr = (int(m[1]), int(m[2]))
            elif m := perf_pat.match(line):
                if curr is not None:
                    store[curr].append(float(m[1]))
                    curr = None          # reset for next head
    return store

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("threshold", type=float,
                    help="required drop in %% (e.g. 50)")
    args = ap.parse_args()

    cutoff = 100.0 - args.threshold      # performance ≤ this ⇒ drop ≥ threshold
    perf   = {h: statistics.mean(vs) for h, vs in collect(args.logfile).items()}
    bad    = sorted(h for h, avg in perf.items() if avg <= cutoff)

    print(json.dumps(bad, indent=2))

if __name__ == "__main__":
    main()
