#!/usr/bin/env python3
"""How far do the victims' decisions follow their utilisation telemetry?

Reads host_data/results/telemetry_reliance (four fixed rewrites of the
path-utilisation slots, see FGSMAttackFramework._rewrite_telemetry) and the
per-victim rule rollouts of ceil2x. All arms share the clean episodes of
threat_util, so differences are paired; intervals are 95 % t (df = 14).

  flips   share of (agent, destination) decisions the rewrite changes
  drop    clean PDR minus PDR under the rewrite, pp (positive = harm)
  %ceil   lure drop / (policy PDR - worst-path PDR): the share of the damage a
          fully trusting policy would suffer that this victim actually suffers

Context columns: what the worst-path and greedy (least-utilised path) rules
deliver on the ceiling's episodes, i.e. where trusting the telemetry leads.

Stdlib only.
"""
import json
import math
import os

ROOT = "host_data/results"
VARIANTS = ["CC-Simple", "CC-Duelling", "LC-Simple", "LC-Duelling",
            "CC-Simple-GNN", "CC-Duelling-GNN", "LC-Duelling-GNN"]
REWRITES = ["ablate_mean", "ablate_shuffle", "lie_repel", "lie_lure"]
T975 = 2.145  # t_{0.975, 14}


def load(*p):
    fp = os.path.join(ROOT, *p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n, m = len(d), sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return m, T975 * sd / math.sqrt(n)


def main():
    out = {}
    print(f"{'victim':<16} {'clean':>6} {'ceil':>5} {'greedy':>6} | " +
          " | ".join(f"{r:>14} flip  drop" for r in REWRITES) + " | lure %ceil")
    for v in VARIANTS:
        t, ceil = load("telemetry_reliance", v, "fgsm_probe_results.json"), load("ceil2x", v, "damage_ceiling.json")
        if not t:
            print(f"{v:<16} no results"); continue
        C = ceil["_meta"]["policy_minus_worst_pp"] if ceil else float("nan")
        greedy = ceil["greedy"]["mean_end_to_end_pdr"] if ceil else float("nan")
        rec, cells = {"ceiling_pp": C, "greedy_pdr": greedy}, []
        clean = None
        for r in REWRITES:
            c = t[f"load2_fail0__{r}_eps1.0_steps1"]
            clean = c["clean_pdr"]
            m, h = paired(c["clean_pdr_series"], c["attacked_pdr_series"])
            rec[r] = {"flip": c["action_flip_rate"], "drop_pp": m, "ci": h}
            cells.append(f"{100 * c['action_flip_rate']:>18.1f}% {m:+5.2f}{'*' if abs(m) > h else ' '}")
        rec["clean_pdr"] = clean
        rec["lure_frac_ceiling"] = rec["lie_lure"]["drop_pp"] / C
        out[v] = rec
        print(f"{v:<16} {clean:>6.1f} {C:>5.1f} {greedy:>6.1f} | " + " | ".join(cells) +
              f" | {100 * rec['lure_frac_ceiling']:>6.0f}%")
    print("\n* = 95% CI of the paired drop excludes 0")
    json.dump(out, open(os.path.join(ROOT, "telemetry_reliance_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
