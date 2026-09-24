#!/usr/bin/env python3
"""Is the FGSM negative result gradient masking? And how mis-aimed was the GNN attack?

Reads the logit-attack probe (host_data/results/logit_attack, run with
faithful_gnn=true) and the original canonical probe (fgsm_tighten), both at the
nominal cell, eps=0.3, 15 paired episodes, identical traffic seed.

Per variant it reports, for each attack, the delivery drop, the flip rate and the
paired adversarial-specific gap against the budget-matched random control.

Two built-in checks:
  * clean PDR must match fgsm_tighten exactly (same victim, same traffic seed);
  * on non-GNN variants faithful_gnn is a no-op, so packet_loss must reproduce
    fgsm_tighten's packet_loss exactly. On GNN variants the difference between the
    two is the effect of attacking through the victim's real decision path.

Stdlib only.
"""

import json
import math
import os

ROOT = "host_data/results"
NEW, OLD = "logit_attack", "fgsm_tighten"
VARIANTS = ["CC-Simple", "CC-Duelling", "LC-Simple", "LC-Duelling",
            "CC-Simple-GNN", "CC-Duelling-GNN", "LC-Duelling-GNN"]
ATTACKS = ["packet_loss", "logit_congestion", "logit_margin"]
CELL = "load2_fail0"


def load(run, v):
    fp = os.path.join(ROOT, run, v, "fgsm_probe_results.json")
    return json.load(open(fp)) if os.path.exists(fp) else None


def get(d, atype):
    return d.get(f"{CELL}__{atype}_eps0.3_steps1") if d else None


def gap(att, rnd):
    """Paired (random - attack) attacked-PDR gap, pp, with 95% CI."""
    a, r = att.get("attacked_pdr_series"), rnd.get("attacked_pdr_series")
    if not a or not r or len(a) != len(r) or len(a) < 2:
        return None, None
    d = [x - y for x, y in zip(r, a)]
    n, m = len(d), sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5
    return m, 1.96 * sd / math.sqrt(n)


rows = {}
print(f"{'variant':<16}{'attack':<18}{'drop pp':>9}{'flips':>8}   {'gap vs random':>17}")
print("-" * 72)
for v in VARIANTS:
    d, old = load(NEW, v), load(OLD, v)
    if not d:
        print(f"{v:<16}(missing)"); continue
    rnd = get(d, "random")
    rows[v] = {}
    for a in ATTACKS + ["random"]:
        c = get(d, a)
        if not c:
            continue
        g, ci = gap(c, rnd) if a != "random" else (None, None)
        rows[v][a] = (c["drop_pp"], (c.get("action_flip_rate") or 0) * 100, g, ci)
        sig = "*" if (g is not None and abs(g) > ci) else " "
        gs = f"{g:+.2f} +/- {ci:.2f}{sig}" if g is not None else ""
        print(f"{v if a == 'packet_loss' else '':<16}{a:<18}{c['drop_pp']:>+9.2f}"
              f"{rows[v][a][1]:>7.1f}%   {gs:>17}")
    # Sanity anchors against the canonical probe.
    if old:
        oc, nc = get(old, "packet_loss"), get(d, "packet_loss")
        same_clean = abs(oc["clean_pdr"] - nc["clean_pdr"]) < 1e-9
        same_pl = abs(oc["attacked_pdr"] - nc["attacked_pdr"]) < 1e-9
        ofl = (oc.get("action_flip_rate") or 0) * 100
        odrop = oc["drop_pp"]
        pl_msg = ("== canonical" if same_pl else
                  f"differs (canonical {odrop:+.2f}pp, {ofl:.1f}% flips)")
        print(f"{'':<16}{'[check]':<18}clean {'==' if same_clean else '!='} canonical;  "
              f"packet_loss {pl_msg}")
    print()

print("gap = random drop - attack drop, paired; * = 95% CI excludes zero.")
print("GNN rows: packet_loss here uses the victim's real GNN decision path; the")
print("canonical probe attacked through actor(raw obs), a function the GNN victim never uses.")
