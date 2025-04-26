#!/usr/bin/env python3
"""
avg_drop_nodes.py
Select ATTENTION HEADS *and* MLPs whose removal hurts performance
by at least THRESHOLD %.

usage examples
    python avg_drop_nodes.py numerals_alternate_node_prune.txt 50
    python avg_drop_nodes.py numerals_alternate_node_prune.txt 60
"""
from __future__ import annotations
import re, argparse, collections, statistics, json, sys, pathlib

# ──────────────────────────────────────────────────────────────────────────
def parse_log(path: str):
    """
    Return two dicts:
        heads : {(layer, head): [%, …]}
        mlps  : {layer:           [%, …]}
    """
    perf_pat   = re.compile(r'\(cand circuit / full\)\s*%:\s*([0-9.]+)')
    head_rm_pat= re.compile(r'Removed:\s*\((\d+),\s*(\d+)\)')
    mlp_rm_pat = re.compile(r'Removed:\s*MLP\s+(\d+)')

    heads = collections.defaultdict(list)
    mlps  = collections.defaultdict(list)
    last_pct = None                                    # save the latest % line

    with open(path) as fh:
        for line in map(str.strip, fh):
            if not line:
                continue
            if m := perf_pat.match(line):
                last_pct = float(m[1])
                continue
            if m := head_rm_pat.match(line):
                if last_pct is not None:
                    heads[(int(m[1]), int(m[2]))].append(last_pct)
                    last_pct = None
                continue
            if m := mlp_rm_pat.match(line):
                if last_pct is not None:
                    mlps[int(m[1])].append(last_pct)
                    last_pct = None
                continue
    return heads, mlps

# ──────────────────────────────────────────────────────────────────────────
def select(avg_map: dict, thresh: float):
    """Return {node: avg_%} whose avg_% ≤ 100 − thresh."""
    cutoff = 100.0 - thresh
    return {k: v for k, v in avg_map.items() if v <= cutoff}

# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("threshold", type=float, help="drop (%) threshold")
    args = ap.parse_args()

    heads_raw, mlps_raw = parse_log(args.logfile)

    heads_avg = {h: statistics.mean(v) for h, v in heads_raw.items()}
    mlps_avg  = {l: statistics.mean(v) for l, v in mlps_raw.items()}

    picked_heads = select(heads_avg, args.threshold)
    picked_mlps  = select(mlps_avg,  args.threshold)

    out = {
        "threshold": args.threshold,
        "heads": [{"layer": h[0], "head": h[1], "avg_pct": round(p, 4)}
                  for h, p in sorted(picked_heads.items())],
        "mlps":  [{"layer": l, "avg_pct": round(p, 4)}
                  for l, p in sorted(picked_mlps.items())]
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
