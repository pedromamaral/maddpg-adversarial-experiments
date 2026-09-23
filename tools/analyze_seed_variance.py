#!/usr/bin/env python3
"""Seed replication: is the adversarial gap a property of the architecture or of one run?

Every headline gap in the FGSM study comes from a single trained victim per
architecture, so a reviewer can ask whether the ordering between variants (say
CC-Simple at +0.4 pp against LC-Simple at +3.3 pp) reflects the design or just the
particular training run. This joins the canonical victim with the two extra seeds
(s1042, s2042) and reports the gap as mean +/- sd over three independently trained
victims of the same architecture.

The gap itself is a PAIRED within-victim measure -- gradient against a budget-matched
random control on the same traffic -- so seed-to-seed variation in baseline delivery
largely cancels. Whether the gap is stable is exactly what this measures.

Stdlib only: runs on the host without torch/numpy.
"""

import json
import math
import os

ROOT = "host_data/results"
CANONICAL = os.path.join(ROOT, "fgsm_tighten")
SEEDS = [("canonical", None), ("s1042", "seed_probe/s1042"), ("s2042", "seed_probe/s2042")]
VARIANTS = ["CC-Simple", "CC-Duelling", "LC-Simple", "LC-Duelling"]
CELL = "load2_fail0"


def gap_for(path):
    """Paired gradient-minus-random gap (pp) and its 95% CI, at the nominal cell."""
    fp = os.path.join(path, "fgsm_probe_results.json")
    if not os.path.exists(fp):
        return None
    d = json.load(open(fp))
    g = d.get(f"{CELL}__packet_loss_eps0.3_steps1")
    r = d.get(f"{CELL}__random_eps0.3_steps1")
    if not g or not r:
        return None
    gs, rs = g.get("attacked_pdr_series"), r.get("attacked_pdr_series")
    if not gs or not rs or len(gs) != len(rs) or len(gs) < 2:
        return None
    # Series are attacked PDR; random minus gradient is positive when the gradient
    # left delivery lower, i.e. when it beat its control.
    diffs = [a - b for a, b in zip(rs, gs)]
    n = len(diffs)
    m = sum(diffs) / n
    sd = (sum((x - m) ** 2 for x in diffs) / (n - 1)) ** 0.5
    return {"gap": m, "ci": 1.96 * sd / math.sqrt(n),
            "clean": g["clean_pdr"], "flips": (g.get("action_flip_rate") or 0) * 100}


def stdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


print(f"Adversarial gap at {CELL}, eps=0.3 — three independently trained victims\n")
print(f"{'variant':<14} {'canonical':>18} {'s1042':>18} {'s2042':>18}   "
      f"{'mean':>7} {'sd':>6}")
print("-" * 92)

summary = {}
for v in VARIANTS:
    cells, gaps = [], []
    for label, sub in SEEDS:
        path = os.path.join(CANONICAL, v) if sub is None else os.path.join(ROOT, sub, v)
        r = gap_for(path)
        if r is None:
            cells.append(f"{'n/a':>18}")
            continue
        gaps.append(r["gap"])
        cells.append(f"{r['gap']:+.2f} +/- {r['ci']:.2f}".rjust(18))
    mean = sum(gaps) / len(gaps) if gaps else float("nan")
    sd = stdev(gaps)
    summary[v] = {"gaps": gaps, "mean": mean, "sd": sd}
    print(f"{v:<14} {' '.join(cells)}   {mean:>+7.2f} {sd:>6.2f}")

print("\nclean delivery per seed (sanity anchor):")
for v in VARIANTS:
    row = []
    for label, sub in SEEDS:
        path = os.path.join(CANONICAL, v) if sub is None else os.path.join(ROOT, sub, v)
        r = gap_for(path)
        row.append(f"{label}={r['clean']:.1f}%" if r else f"{label}=n/a")
    print(f"  {v:<14} " + "  ".join(row))

print("\nordering check — does the between-variant ordering survive the seed spread?")
ordered = sorted(summary.items(), key=lambda kv: -kv[1]["mean"])
for v, s in ordered:
    lo, hi = s["mean"] - s["sd"], s["mean"] + s["sd"]
    print(f"  {v:<14} mean {s['mean']:+.2f}  +/-1sd [{lo:+.2f}, {hi:+.2f}]  "
          f"seeds {['%+.2f' % g for g in s['gaps']]}")
