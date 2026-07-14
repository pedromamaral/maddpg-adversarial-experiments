"""
Paper 1 seed-variance figure (reviewer M1): clean delivery per architecture across
independent training seeds, with the policy-independent worst-path floor.

Shows that the CC>LC ordering is seed-stable: seed spread (<=1.5pp) is far below
the architecture gap (~6pp). Reads host_data/results/seed_variance_summary.json.

Run (server docker):
    RESULTS_ROOT=/workspace/data/results FIG_DIR=/workspace/data/paper1_figures \
        python tools/plot_seed_variance.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# default alongside the other Paper 1 figures (F1..F11 from plot_paper1.py)
ROOT = os.environ.get("RESULTS_ROOT", os.path.join("host_data", "results"))
FIG_DIR = os.environ.get("FIG_DIR",
                         os.path.join("host_data", "results", "reward_fix", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

summary = json.load(open(os.path.join(ROOT, "seed_variance_summary.json")))
# order best->worst delivery, CC family then LC family
ORDER = ["CC-Simple", "CC-Duelling", "LC-Duelling", "LC-Simple"]
COL = {"CC": "#1f77b4", "LC": "#dd8452"}

fig, ax = plt.subplots(figsize=(6.6, 4.2))
xs = np.arange(len(ORDER))
worst_floor = None
for i, v in enumerate(ORDER):
    s = summary[v]
    fam = "CC" if v.startswith("CC") else "LC"
    m, sd = s["policy_pdr_mean"], s["policy_pdr_sd"]
    ax.bar(i, m, 0.55, color=COL[fam], alpha=0.35,
           yerr=sd, capsize=5, error_kw=dict(ecolor="black", lw=1.2))
    # individual seed points
    pts = [d["policy"] for d in s["seeds"].values()]
    ax.plot([i] * len(pts), pts, "o", color=COL[fam], ms=7, mec="black", mew=0.6, zorder=5)
    ax.text(i, m + sd + 1.0, f"{m:.1f}\n(n={s['n_seeds']})", ha="center", fontsize=8)
    worst_floor = list(s["seeds"].values())[0]["worst"]

if worst_floor is not None:
    ax.axhline(worst_floor, ls="--", color="#999999", lw=1.4)
    ax.text(len(ORDER) - 0.5, worst_floor + 0.6, f"worst-path floor ({worst_floor:.1f})",
            ha="right", fontsize=8, color="#666666")

ax.set_xticks(xs)
ax.set_xticklabels(ORDER, fontsize=9)
ax.set_ylabel("clean end-to-end PDR (%), 2$\\times$ hotspot")
ax.set_ylim(60, 95)
ax.set_title("Delivery ranking is seed-stable: CC $>$ LC across all seeds")
ax.grid(axis="y", alpha=0.3)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"F12_seed_variance.{ext}"), bbox_inches="tight")
print("wrote F12_seed_variance")
