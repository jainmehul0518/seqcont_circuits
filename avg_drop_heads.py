#!/usr/bin/env python3
"""
avg_drop_heads.py  –  list attention heads *and* MLPs whose removal
                       hurts performance by >= THRESHOLD %.

usage:
    python avg_drop_heads.py numerals_alternate_node_prune.txt 50
"""

from __future__ import annotations
import re, argparse, collections, statistics, json, sys

# ──────────────────────────────────────────────────────────────────────
def parse_file(path: str):
    """
    Returns two dicts:
        heads { (layer, head): [ %, … ] }
        mlps  { layer        : [ %, … ] }
    """
    two_int_pat = re.compile(r'^(\d+)\s+(\d+)$')   # head line: “23 7”
    one_int_pat = re.compile(r'^(\d+)$')           # mlp  line: “23”
    pct_pat     = re.compile(r'\(cand circuit / full\)\s*%:\s*([0-9.]+)')

    heads, mlps = collections.defaultdict(list), collections.defaultdict(list)
    curr_node, curr_type = None, None              # remember what was last seen

    with open(path) as fh:
        for line in map(str.strip, fh):
            if not line:
                continue
            if m := two_int_pat.match(line):       # attention head
                curr_node, curr_type = (int(m[1]), int(m[2])), "head"
                continue
            if m := one_int_pat.match(line):       # MLP layer
                curr_node, curr_type = int(m[1]), "mlp"
                continue
            if m := pct_pat.match(line):           # % line – record & reset
                pct = float(m[1])
                if curr_type == "head":
                    heads[curr_node].append(pct)
                elif curr_type == "mlp":
                    mlps[curr_node].append(pct)
                curr_node = curr_type = None
    return heads, mlps

# ──────────────────────────────────────────────────────────────────────
def select(avg_map: dict, threshold: float):
    cutoff = 100.0 - threshold                    # ≤ cutoff ⇒ drop ≥ threshold
    return {k: v for k, v in avg_map.items() if v <= cutoff}

# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("threshold", type=float)
    args = ap.parse_args()

    heads_raw, mlps_raw = parse_file(args.logfile)
    heads_avg = {h: statistics.mean(v) for h, v in heads_raw.items()}
    mlps_avg  = {l: statistics.mean(v) for l, v in mlps_raw.items()}

    result = {
        "threshold": args.threshold,
        "heads": [
            {"layer": h[0], "head": h[1], "avg_pct": round(p, 4)}
            for h, p in sorted(select(heads_avg, args.threshold).items())
        ],
        "mlps": [
            {"layer": l, "avg_pct": round(p, 4)}
            for l, p in sorted(select(mlps_avg, args.threshold).items())
        ],
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
