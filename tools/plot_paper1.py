"""
Paper-1 figure generator — MADDPG routing-variant architecture study.

Produces exactly the 10 headline figures (F1–F10) discussed in the results
section: training convergence, load-sweep (normal), hotspot, failure, and the
GNN paired-delta. Training curves are denoised with a rolling mean over the
raw per-episode signal (faint raw trace kept in the background).

Usage:
    python tools/plot_paper1.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("host_data", "results", "main_run")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Variant styling ──────────────────────────────────────────────────────────
# Family colour; GNN variants share their non-GNN family colour but use a dashed
# line so the GNN/non-GNN pairing is visually obvious.
STYLE = {
    "CC-Simple":       dict(color="#1f77b4", ls="-",  marker="o"),
    "CC-Simple-GNN":   dict(color="#1f77b4", ls="--", marker="o"),
    "CC-Duelling":     dict(color="#17becf", ls="-",  marker="s"),
    "CC-Duelling-GNN": dict(color="#17becf", ls="--", marker="s"),
    "LC-Duelling":     dict(color="#2ca02c", ls="-",  marker="^"),
    "LC-Duelling-GNN": dict(color="#2ca02c", ls="--", marker="^"),
    "LC-Simple":       dict(color="#ff7f0e", ls="-",  marker="D"),
    "EVPN_SP":         dict(color="black",   ls=":",  marker="x"),
}
MADDPG_VARIANTS = ["CC-Simple", "CC-Duelling", "LC-Duelling",
                   "CC-Simple-GNN", "CC-Duelling-GNN", "LC-Duelling-GNN", "LC-Simple"]
ALL_METHODS = MADDPG_VARIANTS + ["EVPN_SP"]
# GNN ↔ non-GNN pairs for F10
GNN_PAIRS = [("CC-Simple", "CC-Simple-GNN"),
             ("CC-Duelling", "CC-Duelling-GNN"),
             ("LC-Duelling", "LC-Duelling-GNN")]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 9,
})


def load(fname):
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def rolling_mean(y, window):
    """Centred rolling mean; shrinks window at the edges to avoid NaNs."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.empty(n)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def series(methods_block, scenario_key, metric, loads):
    """Extract [metric] across loads for one method block, returning (x, y)."""
    x, y = [], []
    blk = methods_block[scenario_key] if scenario_key else methods_block
    for L in loads:
        d = blk.get(f"load_{L:.2f}")
        if d and metric in d:
            x.append(L)
            y.append(d[metric])
    return np.array(x), np.array(y)


# ── F1 / F2 : denoised training curves ───────────────────────────────────────
def fig_training(p1, metric_key, ylabel, name, window=41):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for v in MADDPG_VARIANTS:
        raw = np.asarray(p1[v][metric_key], dtype=float)
        epochs = np.arange(len(raw))
        st = STYLE[v]
        ax.plot(epochs, raw, color=st["color"], alpha=0.12, lw=0.8)
        ax.plot(epochs, rolling_mean(raw, window), color=st["color"],
                ls=st["ls"], lw=2.0, label=v)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2, loc="best")
    ax.set_title(f"{ylabel} during training (rolling mean, window={window})")
    save(fig, name)


# ── F3/F4/F5/F7/F8/F9 : load-sweep line plots ────────────────────────────────
def fig_sweep(data, scenario_key, metric, ylabel, name, methods=ALL_METHODS,
              std_metric=None, title="", legend_outside=False):
    loads = data["meta"]["loads"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in methods:
        if m not in data["methods"]:
            continue
        st = STYLE[m]
        x, y = series(data["methods"][m], scenario_key, metric, loads)
        if len(x) == 0:
            continue
        lw = 2.6 if m == "EVPN_SP" else 1.8
        ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=5, lw=lw, label=m)
        if std_metric:
            _, s = series(data["methods"][m], scenario_key, std_metric, loads)
            if len(s) == len(y):
                ax.fill_between(x, y - s, y + s, color=st["color"], alpha=0.08)
    ax.set_xlabel("Offered load factor")
    ax.set_ylabel(ylabel)
    if legend_outside:
        ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, framealpha=0.9)
        fig.subplots_adjust(right=0.78)
    else:
        ax.legend(ncol=2, loc="best")
    if title:
        ax.set_title(title)
    save(fig, name)


# ── F6 : MADDPG − EVPN_SP PDR gap (normal vs hotspot) ─────────────────────────
def fig_gap(ls, hs, name):
    metric = "mean_resolved_pdr"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    panels = [(axes[0], ls, "normal", "Uniform traffic"),
              (axes[1], hs, None, "Hotspot traffic")]
    handles, labels = [], []
    for ax, data, skey, ttl in panels:
        loads = data["meta"]["loads"]
        _, evpn = series(data["methods"]["EVPN_SP"], skey, metric, loads)
        for m in MADDPG_VARIANTS:
            st = STYLE[m]
            x, y = series(data["methods"][m], skey, metric, loads)
            gap = y - evpn
            line, = ax.plot(x, gap, color=st["color"], ls=st["ls"], marker=st["marker"],
                            ms=4, lw=1.8, label=m)
            if skey == "normal":
                handles.append(line)
                labels.append(m)
        ax.axhline(0, color="black", lw=1.2, ls=":")
        ax.set_xlabel("Offered load factor")
        ax.set_title(ttl)
    axes[0].set_ylabel("PDR gap vs EVPN_SP (pp)\n(+ = MADDPG better)")
    fig.legend(handles, labels, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), framealpha=0.9)
    fig.suptitle("MADDPG advantage over shortest-path baseline", y=1.02)
    save(fig, name)


# ── F10 : GNN paired delta (normal vs failure) ───────────────────────────────
def fig_gnn_delta(ls, fs, name, load=2.0):
    metric = "mean_resolved_pdr"
    key = f"load_{load:.2f}"

    def pdr(data, scenario_key, method):
        blk = data["methods"][method]
        blk = blk[scenario_key] if scenario_key else blk
        return blk.get(key, {}).get(metric, np.nan)

    labels = [base for base, _ in GNN_PAIRS]
    normal_delta = [pdr(ls, "normal", g) - pdr(ls, "normal", b) for b, g in GNN_PAIRS]
    failure_delta = [pdr(fs, "uniform", g) - pdr(fs, "uniform", b) for b, g in GNN_PAIRS]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b1 = ax.bar(x - w/2, normal_delta, w, label="Normal (no failure)", color="#4c72b0")
    b2 = ax.bar(x + w/2, failure_delta, w, label="Dual-link failure", color="#c44e52")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GNN effect on PDR (pp)\nGNN − non-GNN")
    ax.set_title(f"Does the GNN help? PDR change from adding a GNN (load={load:.1f}×)")
    ax.legend()
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:+.1f}", (r.get_x() + r.get_width()/2, h),
                        ha="center", va="bottom" if h >= 0 else "top", fontsize=8)
    save(fig, name)


def main():
    p1 = load("phase1_training_results.json")
    ls = load("phase2_load_sweep_results.json")
    hs = load("phase2_hotspot_sweep_results.json")
    fs = load("phase2_failure_sweep_results.json")

    print("Generating Paper-1 figures:")
    # F1, F2 — training
    fig_training(p1, "rewards", "Mean reward", "F1_training_reward")
    fig_training(p1, "pkt_losses", "Packet loss (%)", "F2_training_pkt_loss")
    # F3, F4 — normal load sweep
    fig_sweep(ls, "normal", "mean_resolved_pdr", "Resolved PDR (%)",
              "F3_pdr_vs_load_normal", std_metric="std_resolved_pdr",
              title="Packet delivery vs load — uniform traffic")
    fig_sweep(ls, "normal", "mean_true_pkt_loss", "True packet loss (%)",
              "F4_loss_vs_load_normal",
              title="Packet loss vs load — uniform traffic")
    # F5 — hotspot
    fig_sweep(hs, None, "mean_resolved_pdr", "Resolved PDR (%)",
              "F5_pdr_vs_load_hotspot", std_metric="std_resolved_pdr",
              title="Packet delivery vs load — hotspot traffic")
    # F6 — gap
    fig_gap(ls, hs, "F6_maddpg_vs_evpn_gap")
    # F7, F8 — failure
    fig_sweep(fs, "uniform", "mean_resolved_pdr", "Resolved PDR (%)",
              "F7_pdr_vs_load_failure_uniform", std_metric="std_resolved_pdr",
              title="Packet delivery under dual-link failure — uniform traffic")
    fig_sweep(fs, "hotspot", "mean_resolved_pdr", "Resolved PDR (%)",
              "F8_pdr_vs_load_failure_hotspot", std_metric="std_resolved_pdr",
              title="Packet delivery under dual-link failure — hotspot traffic")
    # F9 — hops under failure
    fig_sweep(fs, "uniform", "mean_hops_mean", "Mean delivered hop count",
              "F9_hops_vs_load_failure",
              title="Path stretch under dual-link failure — uniform traffic",
              legend_outside=True)
    # F10 — GNN paired delta
    fig_gnn_delta(ls, fs, "F10_gnn_paired_delta", load=2.0)
    print("Done.")


if __name__ == "__main__":
    main()
