"""
MSc thesis figure generator — FGSM observation-attack robustness of MADDPG routing.

Produces T1..T7 (see students/goncalo-martins-fgsm-thesis/msc_fgsm_robustness_guide.md)
into students/goncalo-martins-fgsm-thesis/figures/.
Prefers the 15-episode "tightening" run (per-episode series -> paired CIs) and
falls back to the 5-episode "full" run per variant, so it yields a usable draft
before the tightening run finishes and the final figures after.

Run (server docker, matplotlib available):
    RESULTS_ROOT=/workspace/data/results FIG_DIR=/workspace/data/thesis_figures \
        python tools/plot_thesis.py
Locally (if matplotlib present):
    python tools/plot_thesis.py
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("RESULTS_ROOT", os.path.join("host_data", "results"))
FIG_DIR = os.environ.get(
    "FIG_DIR",
    os.path.join("students", "goncalo-martins-fgsm-thesis", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)
# PAPER_MODE: camera-ready figures for the Paper-2 tex — captions carry the message,
# so drop the in-figure titles and bump fonts for legibility at single-column width.
PAPER = bool(os.environ.get("PAPER_MODE"))

VARIANTS = ["CC-Simple", "CC-Duelling", "CC-Simple-GNN", "CC-Duelling-GNN",
            "LC-Simple", "LC-Duelling", "LC-Duelling-GNN"]
GNN = {v for v in VARIANTS if v.endswith("GNN")}
COL = {"grad": "#c44e52", "rand": "#4c72b0", "policy": "#1f77b4",
       "worst": "#bbbbbb", "greedy": "#dd8452"}
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300,
                     "font.size": 13 if PAPER else 11,
                     "axes.labelsize": 14 if PAPER else 11,
                     "xtick.labelsize": 12 if PAPER else 10,
                     "ytick.labelsize": 12 if PAPER else 10,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "legend.fontsize": 11 if PAPER else 9})


def save(fig, name):
    if PAPER:  # captions carry the message in the paper; strip in-figure titles
        for ax in fig.axes:
            ax.set_title("")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def jload(*p):
    fp = os.path.join(ROOT, *p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def variant_probe(v):
    """Prefer tightening (CI series) else full-experiment results for a variant."""
    return (jload("fgsm_tighten", v, "fgsm_probe_results.json"),
            jload("fgsm_full", v, "fgsm_probe_results.json"))


def cell(d, cond, atype, eps):
    if d is None:
        return None
    return d.get(f"{cond}__{atype}_eps{eps}_steps1")


def paired_ci(series_a, series_b):
    """95% CI half-width of mean(a-b); paired episodes."""
    if not series_a or not series_b or len(series_a) != len(series_b) or len(series_a) < 2:
        return None
    d = np.array(series_a) - np.array(series_b)
    return 1.96 * float(d.std(ddof=1)) / math.sqrt(len(d))


# ─── T1: flip rate vs epsilon, gradient vs random (CC-Simple) ────────────────
def t1():
    b = jload("fgsm_probe", "fgsm_probe_results.json")     # budget sweep
    if not b:
        print("  T1 skipped (no budget probe)"); return
    eps = [0.05, 0.10, 0.20, 0.30]
    gf = []
    for e in eps:
        c = b.get(f"packet_loss_eps{e}_steps1")
        gf.append((c["action_flip_rate"] or 0) * 100 if c else np.nan)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(eps, gf, "o-", color=COL["grad"], lw=2.2, ms=6, label="FGSM (gradient)")
    # random control flip points from full experiment (eps 0.1, 0.3, nominal)
    full = jload("fgsm_full", "CC-Simple", "fgsm_probe_results.json")
    rpts = []
    for e in (0.1, 0.3):
        c = cell(full, "load2_fail0", "random", e)
        if c: rpts.append((e, (c["action_flip_rate"] or 0) * 100))
    if rpts:
        xs, ys = zip(*rpts)
        ax.plot(xs, ys, "s--", color=COL["rand"], lw=2.0, ms=6, label="random control")
    ax.set_xlabel(r"perturbation budget $\epsilon$ (L$_\infty$)")
    ax.set_ylabel("decisions flipped (%)")
    ax.set_title("FGSM reliably flips routing decisions (CC-Simple, 2$\\times$ hotspot)")
    ax.legend()
    save(fig, "T1_flip_vs_epsilon")


# ─── T2: decisions changed vs delivery lost, 7 variants @ nominal ────────────
def t2():
    names, flips, drops = [], [], []
    for v in VARIANTS:
        t, f = variant_probe(v)
        c = cell(t, "load2_fail0", "packet_loss", 0.3) or cell(f, "load2_fail0", "packet_loss", 0.3)
        if not c: continue
        names.append(v); flips.append((c["action_flip_rate"] or 0) * 100)
        drops.append(c["drop_pp"])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.barh(y - 0.2, flips, 0.38, color="#8c8c8c", label="decisions flipped (%)")
    ax.barh(y + 0.2, drops, 0.38, color=COL["grad"], label="PDR lost (pp)")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("percent / percentage points")
    ax.set_title("Many decisions change, almost no packets lost (2$\\times$ hotspot, $\\epsilon$0.3)")
    ax.legend(loc="lower right")
    save(fig, "T2_flips_vs_pdr_nominal")


# ─── T3: adversarial gap (grad - rand) @ nominal, 7 variants, 95% CI ─────────
def t3():
    names, gaps, cis = [], [], []
    for v in VARIANTS:
        t, f = variant_probe(v)
        d = t if t else f
        g = cell(d, "load2_fail0", "packet_loss", 0.3)
        r = cell(d, "load2_fail0", "random", 0.3)
        if not g or not r: continue
        names.append(v)
        gaps.append(g["drop_pp"] - r["drop_pp"])
        cis.append(paired_ci(r.get("attacked_pdr_series"), g.get("attacked_pdr_series")))
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xerr = [c if c is not None else 0 for c in cis]
    colors = [COL["grad"] if (c is not None and gaps[i] - c > 0) else "#8c8c8c"
              for i, c in enumerate(cis)]
    ax.barh(y, gaps, xerr=xerr, capsize=3, color=colors, error_kw=dict(ecolor="black", lw=1.1))
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("gradient $-$ random PDR drop (pp)")
    ax.set_title("Only some architectures show a real adversarial signal (nominal, 95% CI)")
    save(fig, "T3_adversarial_gap_by_variant")


# ─── T4: PDR drop vs #failures, gradient vs random, CI bands (CC-Simple) ─────
def t4(variant="CC-Simple"):
    t = jload("fgsm_tighten", variant, "fgsm_probe_results.json")
    if not t:
        print(f"  T4 skipped ({variant} tightening not done)"); return
    ns = [0, 2, 4, 6]
    clean = []  # clean PDR at each n, to expose the self-collapse at n=6
    def drops(atype):
        m, lo, hi = [], [], []
        for n in ns:
            c = t.get(f"load2_fail{n}__{atype}_eps0.3_steps1")
            cs, as_ = (c or {}).get("clean_pdr_series"), (c or {}).get("attacked_pdr_series")
            if cs and as_:
                d = np.array(cs) - np.array(as_)
                mu = d.mean(); ci = 1.96 * d.std(ddof=1) / math.sqrt(len(d))
            else:
                mu = c["drop_pp"] if c else np.nan; ci = 0
            m.append(mu); lo.append(mu - ci); hi.append(mu + ci)
        return np.array(m), np.array(lo), np.array(hi)
    for n in ns:
        c = t.get(f"load2_fail{n}__packet_loss_eps0.3_steps1")
        cs = (c or {}).get("clean_pdr_series")
        clean.append(float(np.mean(cs)) if cs else (c.get("clean_pdr") if c else np.nan))
    gm, glo, ghi = drops("packet_loss")
    rm, rlo, rhi = drops("random")
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.plot(ns, gm, "o-", color=COL["grad"], lw=2.2, label="FGSM (gradient)")
    ax.fill_between(ns, glo, ghi, color=COL["grad"], alpha=0.18)
    ax.plot(ns, rm, "s--", color=COL["rand"], lw=2.2, label="random control")
    ax.fill_between(ns, rlo, rhi, color=COL["rand"], alpha=0.18)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("number of failed links per episode")
    ax.set_ylabel("PDR lost vs clean (pp)")
    # secondary axis: clean (un-attacked) PDR — exposes that the n=6 dip is
    # because the network has already self-collapsed, not renewed robustness.
    ax2 = ax.twinx()
    ax2.plot(ns, clean, "^:", color="#555555", lw=1.6, ms=6, alpha=0.8,
             label="clean PDR (no attack)")
    ax2.set_ylabel("clean PDR (%)", color="#555555")
    ax2.tick_params(axis="y", labelcolor="#555555")
    ax2.set_ylim(0, 100); ax2.grid(False)
    # flag the failure-dominated cell where clean PDR has collapsed on its own
    collapse = [i for i, cp in enumerate(clean) if cp is not None and cp < 20]
    if collapse:
        i = collapse[0]
        ax.annotate("network self-collapsed\n(drop shrinks: little left to lose)",
                    xy=(ns[i], gm[i]), xytext=(ns[i] - 2.4, max(gm) * 0.62),
                    fontsize=8, color="#333333",
                    arrowprops=dict(arrowstyle="->", color="#777777", lw=1))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_title(f"Fragility rises with failures — but random $\\geq$ gradient ({variant})")
    save(fig, "T4_failure_fragility")


# ─── T5: flip rate vs #failures (CC-Simple) ──────────────────────────────────
def t5(variant="CC-Simple"):
    t = jload("fgsm_tighten", variant, "fgsm_probe_results.json")
    if not t:
        print(f"  T5 skipped ({variant} tightening not done)"); return
    ns = [0, 2, 4, 6]
    gf = [((t.get(f"load2_fail{n}__packet_loss_eps0.3_steps1") or {}).get("action_flip_rate") or 0) * 100 for n in ns]
    rf = [((t.get(f"load2_fail{n}__random_eps0.3_steps1") or {}).get("action_flip_rate") or 0) * 100 for n in ns]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ns, gf, "o-", color=COL["grad"], lw=2.2, label="FGSM (gradient)")
    ax.plot(ns, rf, "s--", color=COL["rand"], lw=2.2, label="random control")
    ax.set_xlabel("number of failed links per episode")
    ax.set_ylabel("decisions flipped (%)")
    ax.set_title(f"The attack flips MORE decisions under failure ({variant})")
    ax.legend()
    save(fig, "T5_flip_vs_failures")


# ─── T6: flip rate GNN vs non-GNN @ nominal ──────────────────────────────────
def t6():
    names, flips = [], []
    for v in VARIANTS:
        t, f = variant_probe(v)
        c = cell(t, "load2_fail0", "packet_loss", 0.3) or cell(f, "load2_fail0", "packet_loss", 0.3)
        if not c: continue
        names.append(v); flips.append((c["action_flip_rate"] or 0) * 100)
    colors = ["#55a868" if v in GNN else "#8172b3" for v in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.barh(y, flips, color=colors)
    for i, fl in enumerate(flips):
        ax.text(fl + 0.4, i, f"{fl:.1f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("decisions flipped by FGSM (%)")
    ax.set_title("GNN usually suppresses flips — but not for CC-Duelling (green = GNN)")
    save(fig, "T6_gnn_flip_robustness")


# ─── T7: damage ceiling vs achieved damage, vs load ──────────────────────────
def t7():
    loads = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    pol, worst, greedy = [], [], []
    for l in loads:
        d = jload("reward_fix", "sweep_baselines", f"load_{l:.2f}", "damage_ceiling.json")
        if not d:
            pol.append(np.nan); worst.append(np.nan); greedy.append(np.nan); continue
        pol.append(d["policy"]["mean_end_to_end_pdr"])
        worst.append(d["worst"]["mean_end_to_end_pdr"])
        greedy.append(d["greedy"]["mean_end_to_end_pdr"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(loads, greedy, ":", color=COL["greedy"], lw=1.8, label="greedy (benign best)")
    ax.plot(loads, pol, "o-", color=COL["policy"], lw=2.2, label="policy (clean)")
    ax.plot(loads, worst, "-", color=COL["worst"], lw=1.8, label="worst-path (max damage)")
    ax.fill_between(loads, worst, pol, color=COL["grad"], alpha=0.12,
                    label="damage an ideal obs. attacker could extract")
    ax.set_xlabel("offered load factor (hotspot)")
    ax.set_ylabel("end-to-end PDR (%)")
    ax.set_title("At stake vs achieved: FGSM extracts ~0 of the available damage")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "T7_damage_ceiling_contrast")


if __name__ == "__main__":
    print(f"ROOT={ROOT}  FIG_DIR={FIG_DIR}")
    for fn in (t1, t2, t3, t4, t5, t6, t7):
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__} FAILED: {e}")
    print("done.")
