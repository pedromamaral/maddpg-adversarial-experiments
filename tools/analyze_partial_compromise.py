#!/usr/bin/env python3
"""Partial compromise: how much does the adversary gain per agent it controls?

Each cell is the paired gradient-minus-random gap at a given compromise fraction.
The compromised subset is re-drawn every episode, so one cell already averages over
n_episodes different subsets; repeating the cell under independent subset seeds
(traffic held fixed) separates two things that otherwise share one interval:

  * within-seed CI  -- episode-to-episode variation, for one sequence of subsets
  * across-seed sd  -- how much the answer depends on WHICH agents were compromised

If the across-seed spread is comparable to the effect, then "compromising k agents
costs the victim x pp" is not a statement about k at all.

Stdlib only.
"""

import json
import math
import os

ROOT = "host_data/results/partial_compromise_multi"
VARIANTS = ["CC-Simple", "CC-Duelling", "LC-Simple"]
N_AGENTS = 14
CELL = "load2_fail0"


def paired_gap(grad, rand):
    """Paired (random - gradient) attacked-PDR gap in pp, with 95% CI."""
    gs, rs = grad.get("attacked_pdr_series"), rand.get("attacked_pdr_series")
    if not gs or not rs or len(gs) != len(rs) or len(gs) < 2:
        return None
    d = [a - b for a, b in zip(rs, gs)]
    n = len(d)
    m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5
    return m, 1.96 * sd / math.sqrt(n)


def stdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


for variant in VARIANTS:
    fp = os.path.join(ROOT, variant, "fgsm_probe_results.json")
    if not os.path.exists(fp):
        print(f"=== {variant}: no results ===\n")
        continue
    d = json.load(open(fp))

    # index: fraction -> subset seed -> {attack_type: cell}
    idx = {}
    for v in d.values():
        if v.get("condition") != CELL:
            continue
        idx.setdefault(v.get("attack_fraction", 1.0), {}) \
           .setdefault(v.get("compromise_seed"), {})[v["attack_type"]] = v

    print(f"=== {variant} ===")
    print(f"  {'frac':>5} {'agents':>6}  {'per-seed gaps (pp)':<34} "
          f"{'mean':>7} {'sd':>6} {'per-agent':>10}")
    for frac in sorted(idx):
        gaps, shown = [], []
        for sseed in sorted(idx[frac], key=lambda s: (s is None, s)):
            arms = idx[frac][sseed]
            g, r = arms.get("packet_loss"), arms.get("random")
            if not g or not r:
                continue
            res = paired_gap(g, r)
            if res is None:
                continue
            gaps.append(res[0])
            shown.append(f"{res[0]:+.2f}")
        if not gaps:
            continue
        mean = sum(gaps) / len(gaps)
        n_ag = max(1, round(frac * N_AGENTS))
        print(f"  {frac:>5.2f} {n_ag:>6d}  {' '.join(shown):<34} "
              f"{mean:>+7.2f} {stdev(gaps):>6.2f} {mean / n_ag:>+10.2f}")
    print()

print("per-agent = mean gap / number of compromised agents; falling values mean")
print("diminishing returns, i.e. the first compromised agent is the valuable one.")
print("sd is across independent draws of the compromised set, traffic held fixed.")
