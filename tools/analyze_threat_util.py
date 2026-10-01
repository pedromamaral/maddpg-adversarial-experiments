#!/usr/bin/env python3
"""Stage A under the utilisation-only threat model: budget sweep and iterated attack.

Reads host_data/results/threat_util (single-step sweep) and threat_util_iter
(MI-FGSM on the logit objectives), the per-victim ceilings (ceil2x) and, for
contrast, the legacy all-feature run (logit_attack). All arms of a victim share
the same 15 seeded episodes, so every difference is paired; intervals are 95 % t
(df = 14). The clean arms are checked to be identical across runs.

  gap     = attacked-PDR drop under the attack minus that under the random control
            at the same epsilon (the adversarial-specific effect), in pp
  %ceil   = gap / (clean policy PDR - worst-path PDR) of that victim (ceil2x, 20 eps)

Stdlib only.
"""
import json
import math
import os

ROOT = "host_data/results"
VARIANTS = ["CC-Simple", "CC-Duelling", "LC-Simple", "LC-Duelling",
            "CC-Simple-GNN", "CC-Duelling-GNN", "LC-Duelling-GNN"]
EPS = [0.1, 0.2, 0.3, 0.5, 1.0]
GRAD = ["packet_loss", "logit_margin", "logit_congestion"]
SHORT = {"packet_loss": "FGSM", "logit_margin": "margin", "logit_congestion": "congest"}
T975 = 2.145  # t_{0.975, 14}


def load(*p):
    fp = os.path.join(ROOT, *p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def cell(run, atype, eps, steps=1):
    return (run or {}).get(f"load2_fail0__{atype}_eps{eps}_steps{steps}")


def paired(a, b):
    """mean and t half-width of a - b over episodes."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return m, T975 * sd / math.sqrt(n)


def gap(att, rnd):
    """(drop_att - drop_rnd) = PDR_rnd - PDR_att, per episode."""
    return paired(rnd["attacked_pdr_series"], att["attacked_pdr_series"])


def fmt(m, h):
    return f"{m:+5.2f}{'*' if abs(m) > h else ' '}±{h:4.2f}"


def main():
    out = {}
    print("Utilisation-only threat model, 2x hotspot, no failures, 15 paired episodes.")
    print("gap = drop(attack) - drop(random, same eps), pp; * = 95% CI excludes 0\n")
    for v in VARIANTS:
        sw, it = load("threat_util", v, "fgsm_probe_results.json"), load("threat_util_iter", v, "fgsm_probe_results.json")
        legacy = load("logit_attack", v, "fgsm_probe_results.json")
        ceil = load("ceil2x", v, "damage_ceiling.json")
        if not sw:
            print(f"{v}: no threat_util results"); continue
        c0 = cell(sw, "random", 0.3)
        clean = c0["clean_pdr_series"]
        for other in (it, legacy):
            if other:
                k = next(iter(other))
                assert other[k]["clean_pdr_series"] == clean, f"{v}: clean arms differ"
        C = ceil["_meta"]["policy_minus_worst_pp"] if ceil else float("nan")
        print(f"=== {v}   clean {c0['clean_pdr']:.1f}%   ceiling {C:.1f} pp")
        print(f"  {'eps':>4} {'rand drop':>9} {'rand flip':>9} | " +
              " | ".join(f"{SHORT[a]:>7} flip   gap        %ceil" for a in GRAD))
        rec = {"clean_pdr": c0["clean_pdr"], "ceiling_pp": C, "sweep": {}}
        for e in EPS:
            r = cell(sw, "random", e)
            row = f"  {e:>4} {r['drop_pp']:>+9.2f} {100 * r['action_flip_rate']:>8.1f}% | "
            parts = []
            for a in GRAD:
                g = cell(sw, a, e)
                m, h = gap(g, r)
                parts.append(f"{100 * g['action_flip_rate']:>11.1f}% {fmt(m, h)} {100 * m / C:>5.0f}%")
                rec["sweep"].setdefault(str(e), {})[a] = {
                    "flip": g["action_flip_rate"], "drop_pp": g["drop_pp"], "gap_pp": m, "ci": h,
                    "frac_ceiling": m / C}
            rec["sweep"][str(e)]["random"] = {"flip": r["action_flip_rate"], "drop_pp": r["drop_pp"]}
            print(row + " | ".join(parts))
        if it:
            r = cell(sw, "random", 0.3)
            for a in ("logit_margin", "logit_congestion"):
                s, i = cell(sw, a, 0.3), cell(it, a, 0.3, 20)
                if not i:
                    continue
                m, h = paired(s["attacked_pdr_series"], i["attacked_pdr_series"])
                gm, gh = gap(i, r)
                print(f"  iterated {SHORT[a]:<8} n=20: flips {100 * i['action_flip_rate']:5.1f}% "
                      f"(1-step {100 * s['action_flip_rate']:5.1f}%), gap {fmt(gm, gh)} "
                      f"({100 * gm / C:.0f}% ceil); extra damage over 1-step {fmt(m, h)}")
                rec.setdefault("iterated", {})[a] = {"flip": i["action_flip_rate"], "gap_pp": gm, "ci": gh,
                                                     "extra_over_single_pp": m, "extra_ci": h}
        if legacy:
            lr = cell(legacy, "random", 0.3)
            parts = []
            for a in GRAD:
                lg = cell(legacy, a, 0.3)
                if lg and lr:
                    m, h = gap(lg, lr)
                    parts.append(f"{SHORT[a]} flips {100 * lg['action_flip_rate']:.1f}% gap {fmt(m, h)}")
            print("  legacy all-feature @0.3: " + " | ".join(parts))
        print()
        out[v] = rec
    json.dump(out, open(os.path.join(ROOT, "threat_util_summary.json"), "w"), indent=1)
    print(f"wrote {ROOT}/threat_util_summary.json")


if __name__ == "__main__":
    main()
