#!/usr/bin/env python3
"""Stage B under the utilisation-only threat model: seeds, failures, partial compromise.

  seeds     each variant's canonical victim (threat_util) beside its two extra
            training seeds (seed_util_s1042 / _s2042), at eps 0.3 and 1.0
  failures  the canonical victims at 0 (threat_util), 2, 4 and 6 failed links (failures_fixed)
  partial   1, 4 and 7 of 14 agents compromised, 4 subset draws each (partial_util),
            beside full compromise (threat_util)

gap = drop(attack) - drop(random control, same eps and condition), paired over the
15 episodes; 95 % t interval; * = excludes 0. %ceil uses the victim's own ceiling
where one was measured (ceil2x for canonical victims, seeds/<s>/ceil_<v> for the
non-GNN seeds; none exists for the GNN seeds).

Stdlib only.
"""
import json
import math
import os

ROOT = "host_data/results"
VARIANTS = ["CC-Simple", "CC-Duelling", "LC-Simple", "LC-Duelling",
            "CC-Simple-GNN", "CC-Duelling-GNN", "LC-Duelling-GNN"]
GRAD = ["packet_loss", "logit_margin", "logit_congestion"]
SHORT = {"packet_loss": "FGSM", "logit_margin": "margin", "logit_congestion": "congest"}
T975 = 2.145


def load(*p):
    fp = os.path.join(ROOT, *p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n, m = len(d), sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return m, T975 * sd / math.sqrt(n)


def gap(run, cond, atype, eps, suffix=""):
    a = run.get(f"{cond}__{atype}_eps{eps}_steps1{suffix}")
    r = run.get(f"{cond}__random_eps{eps}_steps1{suffix}")
    if not a or not r:
        return None
    return paired(r["attacked_pdr_series"], a["attacked_pdr_series"])


def fmt(g):
    if g is None:
        return "    --     "
    m, h = g
    return f"{m:+5.2f}{'*' if abs(m) > h else ' '}±{h:4.2f}"


def ceiling(v, seed):
    c = load("ceil2x", v, "damage_ceiling.json") if seed == "canon" else \
        load("seeds", seed, f"ceil_{v}", "damage_ceiling.json")
    return c["_meta"]["policy_minus_worst_pp"] if c else None


def seeds():
    print("== SEEDS: gap over random (pp), and the largest of the three as % of the victim's ceiling")
    print(f"{'variant':<16} {'victim':<6} {'clean':>5} {'ceil':>5} | eps   " +
          " ".join(f"{SHORT[a]:>11}" for a in GRAD) + "   max %ceil  margin-flips")
    for v in VARIANTS:
        for seed, run in (("canon", load("threat_util", v, "fgsm_probe_results.json")),
                          ("s1042", load("seed_util_s1042", v, "fgsm_probe_results.json")),
                          ("s2042", load("seed_util_s2042", v, "fgsm_probe_results.json"))):
            if not run:
                continue
            C = ceiling(v, seed)
            clean = run["load2_fail0__random_eps0.3_steps1"]["clean_pdr"]
            for eps in (0.3, 1.0):
                gs = [gap(run, "load2_fail0", a, eps) for a in GRAD]
                best = max(g[0] for g in gs if g)
                fl = run[f"load2_fail0__logit_margin_eps{eps}_steps1"]["action_flip_rate"]
                pc = f"{100 * best / C:5.0f}%" if C else "   n/a"
                lead = f"{v:<16} {seed:<6} {clean:>5.1f} {C if C else float('nan'):>5.1f}" if eps == 0.3 else " " * 34
                print(f"{lead} | {eps:<4} " + " ".join(fmt(g) for g in gs) + f"   {pc:>7}  {100 * fl:6.1f}%")
        print()


def failures():
    print("== FAILURES (canonical victims, eps 0.3): clean PDR, random drop, gap over random (pp)")
    for v in VARIANTS:
        t0, tf = load("threat_util", v, "fgsm_probe_results.json"), load("failures_fixed", v, "fgsm_probe_results.json")
        for nf, run in ((0, t0), (2, tf), (4, tf), (6, tf)):
            if not run:
                continue
            cond = f"load2_fail{nf}"
            r = run.get(f"{cond}__random_eps0.3_steps1")
            if not r:
                continue
            print(f"  {v:<16} n={nf}: clean {r['clean_pdr']:5.1f}%  random drop {r['drop_pp']:+5.2f}  " +
                  "  ".join(f"{SHORT[a]} {fmt(gap(run, cond, a, 0.3))}" for a in GRAD))
        print()


def partial():
    print("== PARTIAL COMPROMISE (eps 0.3): gap over random, mean ± sd across 4 subset draws")
    for f in sorted(os.listdir(os.path.join(ROOT, "partial_util"))) if os.path.isdir(os.path.join(ROOT, "partial_util")) else []:
        run = load("partial_util", f, "fgsm_probe_results.json")
        if not run:
            continue
        atype = next(k for k in run if "random" not in k).split("__")[1].split("_eps")[0]
        row = []
        for frac, n in ((0.08, 1), (0.29, 4), (0.5, 7)):
            ms = [gap(run, "load2_fail0", atype, 0.3, f"_frac{frac:.2f}_sub{s}")[0] for s in (101, 202, 303, 404)]
            mu = sum(ms) / len(ms)
            sd = math.sqrt(sum((x - mu) ** 2 for x in ms) / (len(ms) - 1))
            row.append(f"{n} agents {mu:+5.2f}±{sd:4.2f}")
        full = gap(load("threat_util", f, "fgsm_probe_results.json"), "load2_fail0", atype, 0.3)
        print(f"  {f:<16} ({SHORT[atype]}): " + "   ".join(row) + f"   14 agents {fmt(full)}")


if __name__ == "__main__":
    seeds()
    failures()
    partial()
