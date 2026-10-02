#!/usr/bin/env python3
"""Does a trained policy route? Compare it with the rules and with v1.

For each RUN:VARIANT, reads host_data/results/RUN/eval/ (written by
tools/run_eval.sh) and prints, per number of failed links, the policy's delivery
beside the four rules on the same 20 episodes, then its decision attribution.
The v1 victim of the same variant is shown for reference: rules and policy from
failsev_fixed (CC-Simple only) or ceil2x (no failures), attribution from
policy_attribution.json.

    python tools/compare_eval.py v2_pilot_lc:LC-Simple v2_pilot_cc:CC-Simple

Stdlib only.
"""
import json
import os
import sys

ROOT = "host_data/results"
RULES = ["greedy", "random", "sp", "worst"]


def load(*p):
    fp = os.path.join(ROOT, *p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def pdr(d, row):
    return d[row]["mean_end_to_end_pdr"] if d and row in d else None


def v1_ceiling(variant, k):
    if variant == "CC-Simple":
        return load("failsev_fixed", f"n_{k}", "damage_ceiling.json")
    return load("ceil2x", variant, "damage_ceiling.json") if k == 0 else None


def fmt(x):
    return f"{x:6.1f}" if x is not None else "     -"


def attribution_line(tag, a):
    imp = a.get("importance_live", {})
    return (f"  {tag:<4} {a['_n']}: static {100 * a['static_share_live']:5.1f}%  "
            f"least-loaded {100 * a['greedy_agreement']:5.1f}% (chance {100 * a['greedy_chance']:4.1f}%)  "
            f"saturated {100 * a['saturation_live']:4.0f}%  "
            f"decisions moved by shuffling util {100 * imp.get('path_util', float('nan')):4.1f}% / all {100 * imp.get('all', float('nan')):4.1f}%")


def main(specs):
    v1_attr = load("policy_attribution.json") or {}
    for spec in specs:
        run, variant = spec.split(":")
        print(f"\n=== {run} / {variant}   (PDR %, 2x hotspot, 20 paired episodes; v1 = old design)")
        print(f"  {'fail':>4} | {'policy':>6} {'v1':>6} | " + " ".join(f"{r:>6}" for r in RULES) + " | beats")
        for k in (0, 2, 4, 6, 8):
            d = load(run, "eval", f"n_{k}", "damage_ceiling.json")
            if not d:
                continue
            v1 = v1_ceiling(variant, k)
            p = pdr(d, "policy")
            # rules are policy-independent: they must match v1's rule rows on the same episodes
            same = v1 is None or all(pdr(d, r) == pdr(v1, r) for r in RULES)
            beats = [r for r in RULES if p > pdr(d, r)]
            print(f"  {k:>4} | {fmt(p)} {fmt(pdr(v1, 'policy'))} | " + " ".join(fmt(pdr(d, r)) for r in RULES)
                  + f" | {','.join(beats) or '-'}" + ("" if same else "   !! rule rows differ from v1 (pairing broken)"))
        a = load(run, "eval", f"{variant}_attribution.json")
        for tag, src in (("v2", (a or {}).get(variant, {})), ("v1", v1_attr.get(variant, {}))):
            for key in ("fail0", "fail2", "fail4"):
                if key in src:
                    print(attribution_line(tag, dict(src[key], _n=key)))


if __name__ == "__main__":
    main(sys.argv[1:] or ["v2_pilot_lc:LC-Simple", "v2_pilot_cc:CC-Simple"])
