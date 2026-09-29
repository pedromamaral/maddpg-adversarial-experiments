#!/usr/bin/env python3
"""Recompute Delta^FGSM (Miguel's Tables 4.6 / 4.7) against a given FGSM arm.

Delta^FGSM = mean over the 15 shared episodes of (PDR_FGSM - PDR_learned), with a
paired 95% t-interval (df = 14). Positive = the learned adversary did more damage.

The learned-adversary records are Miguel's FGSM-parity evaluations
(learned_adv_parity/<variant>/[eps0.30-tb<b>/]learned_adv_eval_<scope>.json). The
FGSM arm comes from either the original probe (fgsm_tighten) or the re-run that
attacks GNN victims through their real decision path (logit_attack). Every cell
first checks that the clean arms are bit-identical on the shared episodes, which
is what makes the pairing valid.

Run with --arm fgsm_tighten to reproduce the published tables (validation), and
with --arm logit_attack for the corrected GNN rows.

    python tools/recompute_learned_vs_fgsm.py --learned host_data/results/mchen \
        --arm logit_attack
"""
import argparse
import json
import math
import os

VARIANTS = ["CC-Simple", "CC-Duelling", "CC-Simple-GNN", "CC-Duelling-GNN",
            "LC-Simple", "LC-Duelling", "LC-Duelling-GNN"]
BUDGETS = ["0.10", "0.15", "0.20", "0.25", None]
FGSM_KEY = "load2_fail0__packet_loss_eps0.3_steps1"
T975_14 = 2.145


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def learned_record(root, v, scope, b):
    sub = os.path.join(root, "learned_adv_parity", v)
    if b is not None:
        sub = os.path.join(sub, f"eps0.30-tb{b}")
    return load(os.path.join(sub, f"learned_adv_eval_{scope}.json"))


def delta(fgsm, rec):
    """(mean, half-width) of FGSM - learned on the shared episodes, or an error."""
    n = len(fgsm["attacked_pdr_series"])
    if rec["pdr"]["clean_series"][:n] != fgsm["clean_pdr_series"]:
        return "clean arms differ"
    d = [f - l for f, l in zip(fgsm["attacked_pdr_series"],
                               rec["pdr"]["attack_series"][:n])]
    m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5
    return m, T975_14 * sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learned", default="host_data/results/mchen")
    ap.add_argument("--probes", default="host_data/results")
    ap.add_argument("--arm", default="logit_attack",
                    help="probe run that supplies the FGSM arm")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    out = {}
    for scope in ("independent", "coordinated"):
        print(f"\n{scope} scope, FGSM arm = {args.arm}   (* = 95% CI excludes 0)")
        print(f"{'variant':<17}" + "".join(f"{(b or 'ungated'):>10}" for b in BUDGETS))
        for v in VARIANTS:
            probe = load(os.path.join(args.probes, args.arm, v, "fgsm_probe_results.json"))
            fgsm = probe.get(FGSM_KEY) if probe else None
            cells = []
            for b in BUDGETS:
                rec = learned_record(args.learned, v, scope, b)
                if not fgsm or not rec:
                    cells.append("missing"); continue
                r = delta(fgsm, rec)
                if isinstance(r, str):
                    cells.append(r); continue
                m, h = r
                out.setdefault(scope, {}).setdefault(v, {})[b or "ungated"] = [m, h]
                cells.append(f"{m:+.2f}{'*' if abs(m) > h else ' '}")
            print(f"{v:<17}" + "".join(f"{c:>10}" for c in cells))
    if args.json_out:
        json.dump(out, open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
