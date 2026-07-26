"""
FGSM robustness analyzer for Paper 2 / the FGSM thesis.

Reads the probe results produced by `--phase fgsm_probe` — one
`fgsm_probe_results.json` per variant under a run directory (the canonical run
is `fgsm_tighten`, the 15-episode paired-CI sweep that backs figures T3-T7) —
and prints the exploitability picture per variant:

  * clean vs attacked PDR, and the DROP, for the gradient attack (packet_loss);
  * the same for the budget-matched RANDOM control at the identical epsilon;
  * the load-bearing quantity, the NET adversarial gap = gradient drop - random
    drop, with a 95% paired-episode CI, which separates a real adversarial
    signal from mere perturbation jostling;
  * the action-flip rate (decisions changed), reported alongside the PDR drop
    (outcomes changed) because a high flip rate with a flat drop is the network
    ABSORBING the attack, not the attack winning.

This is the text companion to tools/plot_thesis.py (T3-T7), reading the SAME
files. Runs on stdlib only (no numpy), so it works anywhere.

Usage:
    python tools/analyze_fgsm.py [RUN_DIR]
Default RUN_DIR: host_data/results/fgsm_tighten
Override the results root with RESULTS_ROOT (e.g. /workspace/data/results in the
container); RUN_DIR may be absolute or a name/relative path under that root.
"""
import json
import math
import os
import sys

ROOT = os.environ.get("RESULTS_ROOT", os.path.join("host_data", "results"))
RUN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TIGHTEN_RUN", "fgsm_tighten")
RUN_DIR = RUN if os.path.isabs(RUN) else os.path.join(ROOT, RUN)

PDR = "mean_end_to_end_pdr"  # kept for reference; probe cells expose *_pdr directly
VARIANTS = ["CC-Simple", "CC-Duelling", "CC-Simple-GNN", "CC-Duelling-GNN",
            "LC-Simple", "LC-Duelling", "LC-Duelling-GNN"]
EPS = "0.3"          # the probe grid attacks at a single budget
COLLAPSE_PDR = 40.0  # below this clean PDR the network self-collapses; attack signal is noise


def load_variant(v):
    fp = os.path.join(RUN_DIR, v, "fgsm_probe_results.json")
    return json.load(open(fp)) if os.path.exists(fp) else None


def conditions(d):
    """Ordered (key, n_failures) for every failure level present, gradient side."""
    out = []
    for c in d.values():
        if isinstance(c, dict) and c.get("attack_type") == "packet_loss":
            out.append((c.get("condition"), int(c.get("n_failures", 0))))
    return sorted(set(out), key=lambda t: t[1])


def paired_ci(series_a, series_b):
    """95% CI half-width of mean(a-b) over paired episodes; None if unavailable."""
    if not series_a or not series_b or len(series_a) != len(series_b) or len(series_a) < 2:
        return None
    diffs = [a - b for a, b in zip(series_a, series_b)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    return 1.96 * math.sqrt(var) / math.sqrt(n)


def main():
    if not os.path.isdir(RUN_DIR):
        print(f"No probe run directory at {RUN_DIR}\n"
              f"Expected <RUN_DIR>/<variant>/fgsm_probe_results.json "
              f"(produced by --phase fgsm_probe). Pass the run dir as an argument "
              f"or set RESULTS_ROOT.")
        return

    done = [v for v in VARIANTS if load_variant(v) is not None]
    pending = [v for v in VARIANTS if v not in done]
    print(f"FGSM probe analysis — {RUN_DIR}")
    print(f"variants present: {len(done)}/7  {done}")
    if pending:
        print(f"missing: {pending}")
    print("=" * 74)

    verdicts = {}  # variant -> (nominal net gap, verdict)

    for v in done:
        d = load_variant(v)
        conds = conditions(d)
        if not conds:
            print(f"\n### {v} — no gradient cases found (unexpected schema?)")
            continue

        # clean PDR at nominal (no failures) characterises the loaded victim:
        # ~87% = correctly loaded, ~55% = wrong/missing weights.
        nominal = d.get(f"load2_fail0__packet_loss_eps{EPS}_steps1", {})
        print(f"\n### {v}   (clean PDR @ nominal ~ {nominal.get('clean_pdr', float('nan')):.1f}%)")
        print(f"  {'failures':>8}  {'grad drop':>9}  {'rand drop':>9}  "
              f"{'net gap':>16}  {'flip%':>6}")

        for cond_key, nf in conds:
            g = d.get(f"{cond_key}__packet_loss_eps{EPS}_steps1")
            r = d.get(f"{cond_key}__random_eps{EPS}_steps1")
            if not g:
                continue
            gdrop = g["drop_pp"]
            rdrop = r["drop_pp"] if r else float("nan")
            net = gdrop - rdrop if r else float("nan")
            # CI on the net gap: paired (random_attacked - gradient_attacked) per episode
            ci = paired_ci(r.get("attacked_pdr_series") if r else None,
                           g.get("attacked_pdr_series"))
            net_str = (f"{net:+5.1f} ± {ci:.1f}pp" if ci is not None
                       else f"{net:+5.1f}pp" if r else "     n/a")
            flip = (g.get("action_flip_rate") or 0.0) * 100
            note = "  <- self-collapse" if g["clean_pdr"] < COLLAPSE_PDR else ""
            print(f"  {nf:>8}  {gdrop:+7.1f}pp  {rdrop:+7.1f}pp  {net_str:>16}  {flip:5.1f}{note}")

            if nf == 0:  # the exploitability verdict is read at nominal load
                sig = ci is not None and net - ci > 0   # net gap CI clears zero
                verdict = ("EXPLOITABLE" if sig and net >= 5 else
                           "borderline" if sig and net >= 2 else
                           "ROBUST")
                verdicts[v] = (net, ci, verdict)

    if verdicts:
        print("\n" + "=" * 74)
        print("EXPLOITABILITY @ nominal load (net gradient gap above random control):")
        for v in done:
            if v in verdicts:
                net, ci, verdict = verdicts[v]
                ci_str = f" ± {ci:.1f}" if ci is not None else ""
                print(f"  {v:18s} {net:+6.1f}{ci_str}pp   {verdict}")
        print("\nNet gap = gradient PDR drop - random-control PDR drop; a positive gap whose\n"
              "95% CI clears zero is evidence of a real (not budget-matched-noise) attack.\n"
              "Failure rows are shown for context; the nominal row is the headline. Rows\n"
              "marked self-collapse (clean PDR < 40%) are failure-dominated, not attack signal.")


if __name__ == "__main__":
    main()
