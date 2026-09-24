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
Naming figure functions regenerates only those, e.g.:
    PAPER_MODE=1 python tools/plot_thesis.py t2 t6 t9
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("RESULTS_ROOT", os.path.join("host_data", "results"))
# Run directory names under ROOT, hoisted (and env-overridable) so renaming a result
# set is a one-line change here. TIGHTEN is the 15-episode paired-CI probe, FULL the
# earlier epsilon/load sweep, CANONICAL the Paper-1 victim run these attack.
CANONICAL = os.environ.get("CANONICAL_RUN", "reward_fix")
TIGHTEN = os.environ.get("TIGHTEN_RUN", "fgsm_tighten")
FULL = os.environ.get("FULL_RUN", "fgsm_full")
# On GNN victims TIGHTEN took the attack gradient through the actor alone,
# bypassing the encoder the victim decides through. LOGIT re-ran every variant
# with the attack aimed through the real decision path (faithful_gnn), plus the
# two logit-margin objectives; on non-GNN variants it reproduces TIGHTEN exactly.
LOGIT = os.environ.get("LOGIT_RUN", "logit_attack")
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
    """Prefer tightening (CI series) else full-experiment results for a variant.

    GNN variants read the faithful re-run instead: their TIGHTEN attack never
    went through the encoder, so its numbers describe a different function.
    """
    t = jload(TIGHTEN, v, "fgsm_probe_results.json")
    if v in GNN:
        faithful = jload(LOGIT, v, "fgsm_probe_results.json")
        if faithful:
            t = faithful
        else:
            print(f"  WARNING: no {LOGIT} run for {v}; its FGSM numbers bypass the encoder")
    return t, jload(FULL, v, "fgsm_probe_results.json")


def cell(d, cond, atype, eps):
    if d is None:
        return None
    return d.get(f"{cond}__{atype}_eps{eps}_steps1")


# Two-sided 95% Student-t quantiles by degrees of freedom. The probes pair 15
# episodes (df = 14), where the normal 1.96 understates the interval by ~9%.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
         15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
         21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
         27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t975(n):
    """95% two-sided t quantile for the mean of n paired differences."""
    return _T975.get(n - 1, 1.96)


def paired_ci(series_a, series_b):
    """95% CI half-width of mean(a-b); paired episodes."""
    if not series_a or not series_b or len(series_a) != len(series_b) or len(series_a) < 2:
        return None
    d = np.array(series_a) - np.array(series_b)
    return t975(len(d)) * float(d.std(ddof=1)) / math.sqrt(len(d))


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
    full = jload(FULL, "CC-Simple", "fgsm_probe_results.json")
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
    # Upper right: the top (LC-Duelling-GNN) row is short, whereas lower right
    # would sit on CC-Simple's 25.7% bar.
    ax.legend(loc="upper right")
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


# ─── T3b: the same gap across independently trained victims ─────────────────
SEED_RUNS = [("canonical", None), ("s1042", "seed_probe/s1042"),
             ("s2042", "seed_probe/s2042")]


def _gap_at_nominal(d):
    """Paired gradient-minus-random gap (pp) at the nominal cell, or None."""
    g = cell(d, "load2_fail0", "packet_loss", 0.3)
    r = cell(d, "load2_fail0", "random", 0.3)
    if not g or not r:
        return None
    return g["drop_pp"] - r["drop_pp"]


def t3_seeds():
    """Replaces T3. T3 ranks architectures by a single trained victim each, and
    that ranking does not replicate: CC-Simple spans +0.37 to +5.11 pp across three
    training seeds. Plotting one marker per seed shows the effect is small
    everywhere, that the seed spread swamps the differences between variants, and
    — because the GNN variants have only one trained seed — exactly which rows
    carry replication and which do not. The canonical GNN markers come from the
    faithful re-run, as everywhere else (variant_probe).
    """
    rows = []
    for v in VARIANTS:
        gaps = []
        for _label, sub in SEED_RUNS:
            d = (variant_probe(v)[0] if sub is None
                 else jload(sub, v, "fgsm_probe_results.json"))
            if not d:
                continue
            g = _gap_at_nominal(d)
            if g is not None:
                gaps.append(g)
        if gaps:
            rows.append((v, gaps))
    if not rows:
        print("  T3b skipped (no data)"); return

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, (v, gaps) in enumerate(rows):
        replicated = len(gaps) > 1
        if replicated:
            m = float(np.mean(gaps))
            sd = float(np.std(gaps, ddof=1))
            ax.barh(i, 2 * sd, left=m - sd, height=0.52,
                    color=COL["grad"], alpha=0.16, zorder=1)
            ax.plot([m, m], [i - 0.26, i + 0.26], color=COL["grad"], lw=2.4, zorder=3)
        ax.scatter(gaps, [i] * len(gaps),
                   s=46 if replicated else 64,
                   facecolor=COL["grad"] if replicated else "none",
                   edgecolor=COL["grad"], linewidth=1.6, zorder=4,
                   label=None)
    ax.axvline(0, color="black", lw=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{v}" if len(g) > 1 else f"{v}  (1 seed)"
                        for v, g in rows], fontsize=9)
    # Keep the label short: at PAPER font sizes a longer one is clipped. The
    # "one marker per trained victim" reading belongs in the caption.
    ax.set_xlabel("gradient $-$ random PDR drop (pp)")
    if not PAPER:
        ax.set_title("The per-architecture ordering does not survive reseeding")
    # Legend built by hand: filled = three seeds, hollow = single seed. Anchored
    # upper-right, where the single-seed GNN rows leave the axes empty; lower-right
    # would cover CC-Simple's +5.11 pp seed.
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker='o', ls='none', color=COL["grad"],
               label='trained victim (3 seeds)'),
        Line2D([], [], marker='o', ls='none', markerfacecolor='none',
               markeredgecolor=COL["grad"], label='single seed — not replicated'),
        Line2D([], [], color=COL["grad"], lw=2.4, label='mean; band = $\\pm$1 sd'),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    ax.margins(x=0.06)
    save(fig, "T3b_gap_across_seeds")


# ─── T4: PDR drop vs #failures, gradient vs random, CI bands (CC-Simple) ─────
def t4(variant="CC-Simple"):
    t = jload(TIGHTEN, variant, "fgsm_probe_results.json")
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
                mu = d.mean(); ci = t975(len(d)) * d.std(ddof=1) / math.sqrt(len(d))
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
    t = jload(TIGHTEN, variant, "fgsm_probe_results.json")
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


# ─── T6: what the GNN encoder does to noise and to the attack @ nominal ──────
def _flip_pct(c):
    return (c["action_flip_rate"] or 0) * 100 if c else None


def t6():
    """Decisions flipped by the random control and by FGSM, per variant.

    Replaces the old T6 ("GNN suppresses flips"), which plotted the canonical
    attack only. On GNN victims that attack bypassed the encoder, so its low flip
    rates said nothing about an attacker who knows the encoder. Showing the
    random control, the bypassing attack and the attack through the encoder side
    by side separates the three things the old figure conflated.
    """
    rows = []
    for v in VARIANTS:
        d = variant_probe(v)[0]
        rnd = _flip_pct(cell(d, "load2_fail0", "random", 0.3))
        aimed = _flip_pct(cell(d, "load2_fail0", "packet_loss", 0.3))
        bypass = (_flip_pct(cell(jload(TIGHTEN, v, "fgsm_probe_results.json"),
                                 "load2_fail0", "packet_loss", 0.3))
                  if v in GNN else None)
        if rnd is None or aimed is None:
            continue
        rows.append((v, rnd, bypass, aimed))
    if not rows:
        print("  T6 skipped (no data)"); return

    h = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for i, (v, rnd, bypass, aimed) in enumerate(rows):
        if v in GNN:
            ax.axhspan(i - 0.5, i + 0.5, color="#55a868", alpha=0.07, zorder=0)
            bars = [(i + h, rnd, dict(color=COL["rand"])),
                    (i, bypass, dict(color="white", edgecolor="0.45", hatch="///")),
                    (i - h, aimed, dict(color=COL["grad"]))]
        else:
            bars = [(i + h / 2, rnd, dict(color=COL["rand"])),
                    (i - h / 2, aimed, dict(color=COL["grad"]))]
        for yy, val, style in bars:
            ax.barh(yy, val, h, zorder=2, **style)
            ax.text(val + 0.4, yy, f"{val:.1f}", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel(r"decisions flipped (%), $\epsilon$ = 0.3")
    ax.margins(x=0.08)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=COL["rand"], label="random control"),
        Patch(facecolor="white", edgecolor="0.45", hatch="///",
              label="FGSM, encoder bypassed (original attack)"),
        Patch(color=COL["grad"], label="FGSM through the victim's decision path"),
    ], loc="upper right", fontsize=8, framealpha=0.95)
    ax.set_title("GNN damps noise, not an attack aimed through it (shaded = GNN)")
    save(fig, "T6_gnn_noise_vs_attack")


# ─── T7: damage ceiling vs achieved damage, vs load ──────────────────────────
def t7():
    loads = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    # The policy line comes from the phase-2 load sweep, as in Paper 1's F5. The
    # "policy" rows of sweep_baselines must not be used: their models/ folders
    # are empty (broken symlink), so no trained weights were loaded. Only the
    # rule rows there, which never touch the policy, are valid.
    sw = jload(CANONICAL, "phase2_hotspot_sweep_results.json")
    pol_by_load = (sw or {}).get("methods", {}).get("CC-Simple", {})
    pol, worst, greedy, rnd = [], [], [], []
    for l in loads:
        d = jload(CANONICAL, "sweep_baselines", f"load_{l:.2f}", "damage_ceiling.json")
        p = pol_by_load.get(f"load_{l:.2f}")
        pol.append(p["mean_end_to_end_pdr"] if p else np.nan)
        if not d:
            for s in (worst, greedy, rnd):
                s.append(np.nan)
            continue
        worst.append(d["worst"]["mean_end_to_end_pdr"])
        greedy.append(d["greedy"]["mean_end_to_end_pdr"])
        rnd.append(d["random"]["mean_end_to_end_pdr"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(loads, greedy, ":", color=COL["greedy"], lw=1.8,
            label="greedy (least-congested path)")
    # The random rule is the reference the victim should at least match: it
    # delivers more than the trained policy at every load.
    ax.plot(loads, rnd, "--", color=COL["rand"], lw=1.6, label="random path (per step)")
    ax.plot(loads, pol, "o-", color=COL["policy"], lw=2.2, label="policy (clean)")
    ax.plot(loads, worst, "-", color=COL["worst"], lw=1.8, label="worst-path (max damage)")
    ax.fill_between(loads, worst, pol, color=COL["grad"], alpha=0.12,
                    label="damage an ideal obs. attacker could extract")
    ax.set_xlabel("offered load factor (hotspot)")
    ax.set_ylabel("end-to-end PDR (%)")
    ax.set_title("At stake: the policy sits between random routing and the worst path")
    # Headroom under the worst-path curve so the five-entry legend clears it.
    ax.set_ylim(np.nanmin(worst) - 12, 101)
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "T7_damage_ceiling_contrast")


# ─── T8: iterated attacks (PGD / MI-FGSM) against single-step FGSM ───────────
def _pgd_pick(rows, kind):
    """Select one attack configuration from a pgd_diagnostic row list.

    Matched on the parameters rather than the display label so the figure does
    not break if a label is reworded.
    """
    for r in rows:
        if kind == "fgsm" and r["n_steps"] == 1:
            return r
        # The shipped PGD: 10 steps, no momentum, objective re-read each step.
        if (kind == "pgd" and r["n_steps"] == 10 and r["momentum"] == 0
                and not r["freeze_weights"]
                and abs(r["step_alpha"] - 0.3 / 10 * 2.5) < 1e-9):
            return r
        # Best-tuned iterated attack: momentum, 20 steps, no random restart.
        if (kind == "mi" and r["momentum"] > 0 and r["n_steps"] == 20
                and not r["random_start"]):
            return r
    return None


def t8():
    data = {}
    for v in VARIANTS:
        d = jload("pgd_diagnostic", f"{v}.json")
        if d:
            data[v] = d
    if not data:
        print("  T8 skipped (no pgd_diagnostic/*.json)"); return

    order = [v for v in VARIANTS if v in data]
    spend = {k: [] for k in ("fgsm", "pgd", "mi")}
    dflip = {k: [] for k in ("pgd", "mi")}
    for v in order:
        rows = data[v]["rows"]
        picks = {k: _pgd_pick(rows, k) for k in ("fgsm", "pgd", "mi")}
        for k in spend:
            spend[k].append(picks[k]["budget_spend"] if picks[k] else np.nan)
        base = picks["fgsm"]["flip_rate"] if picks["fgsm"] else np.nan
        for k in ("pgd", "mi"):
            dflip[k].append((picks[k]["flip_rate"] - base) * 100 if picks[k] else np.nan)

    y = np.arange(len(order))
    h = 0.26
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # Left: how much of the epsilon budget each attack actually spends.
    axL.barh(y + h, spend["fgsm"], h, color=COL["grad"], label="FGSM (1 step)")
    axL.barh(y, spend["pgd"], h, color=COL["rand"], label="PGD (10 steps)")
    axL.barh(y - h, spend["mi"], h, color=COL["greedy"], label="MI-FGSM (20, momentum)")
    axL.axvline(1.0, color="0.4", lw=1.0, ls=":")
    axL.set_yticks(y); axL.set_yticklabels(order)
    axL.set_xlabel(r"budget actually spent  ($\overline{|\delta|}/\epsilon$)")
    axL.set_xlim(0, 1.08)
    axL.legend(loc="lower right", fontsize=9)
    if not PAPER:
        axL.set_title("Plain PGD wastes half its budget; momentum recovers it")

    # Right: flips relative to FGSM. Within-variant differences only -- the
    # absolute rates are not comparable to T2/T6 because the GNN attack path
    # decodes through a zero-filled batch (see the diagnostic's docstring).
    axR.barh(y + h / 2, dflip["pgd"], h, color=COL["rand"], label="PGD (10 steps)")
    axR.barh(y - h / 2, dflip["mi"], h, color=COL["greedy"],
             label="MI-FGSM (20, momentum)")
    axR.axvline(0.0, color=COL["grad"], lw=1.6)
    axR.set_yticks(y); axR.set_yticklabels([])
    axR.set_xlabel("decisions flipped vs FGSM (pp;  0 = FGSM)")
    axR.legend(loc="lower left", fontsize=9)
    if not PAPER:
        axR.set_title("No iterated attack beats the single step")

    save(fig, "T8_iterated_vs_fgsm")


# ─── T9: single-step attack on the logits against FGSM ───────────────────────
def t9():
    """FGSM's objective reads the actor's sigmoid outputs, which the trained
    policy saturates, so its gradient nearly vanishes. The two logit objectives
    act on the pre-sigmoid scores instead. Same epsilon, one sign step, same 15
    paired episodes, GNN variants attacked through their real decision path.

    Left: decisions flipped by each attack and by the random control. Right: the
    delivery each logit attack removed beyond what FGSM removed, paired on the
    same episodes (positive = more damage than FGSM).
    """
    attacks = [("random", "random control", COL["rand"]),
               ("packet_loss", "FGSM", COL["grad"]),
               ("logit_congestion", "logits: push the most congested path up",
                COL["greedy"]),
               ("logit_margin", "logits: push the best alternative up", "#8172b3")]
    rows = []
    for v in VARIANTS:
        d = jload(LOGIT, v, "fgsm_probe_results.json")
        cells = {a: cell(d, "load2_fail0", a, 0.3) for a, _, _ in attacks}
        if any(c is None for c in cells.values()):
            continue
        extra = {}
        for a in ("logit_congestion", "logit_margin"):
            base = np.array(cells["packet_loss"]["attacked_pdr_series"])
            att = np.array(cells[a]["attacked_pdr_series"])
            diff = base - att          # positive: the logit attack delivered less
            ci = t975(len(diff)) * float(diff.std(ddof=1)) / math.sqrt(len(diff))
            extra[a] = (float(diff.mean()), ci)
        rows.append((v, {a: _flip_pct(c) for a, c in cells.items()}, extra))
    if not rows:
        print(f"  T9 skipped (no {LOGIT} run)"); return

    y = np.arange(len(rows))
    names = [r[0] for r in rows]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 5.0), sharey=True,
                                   gridspec_kw=dict(width_ratios=[1.15, 1]))

    # Left: flips, four bars per variant.
    h = 0.2
    for j, (a, label, color) in enumerate(attacks):
        axL.barh(y + (1.5 - j) * h, [r[1][a] for r in rows], h,
                 color=color, label=label)
    axL.set_yticks(y); axL.set_yticklabels(names, fontsize=9)
    axL.set_xlabel(r"decisions flipped (%), $\epsilon$ = 0.3")
    if not PAPER:
        axL.set_title("The logit objectives flip more decisions everywhere")

    # Right: extra delivery lost vs FGSM, with paired 95% CIs. Hollow bars have an
    # interval that includes zero.
    h = 0.32
    for j, (a, label, color) in enumerate(attacks[2:]):
        yy = y + (0.5 - j) * h
        means = [r[2][a][0] for r in rows]
        cis = [r[2][a][1] for r in rows]
        sig = [abs(m) > c for m, c in zip(means, cis)]
        axR.barh(yy, means, h, xerr=cis, capsize=2.5,
                 color=[color if s else "white" for s in sig],
                 edgecolor=color, linewidth=1.4,
                 error_kw=dict(ecolor="0.3", lw=1.0))
    axR.axvline(0.0, color=COL["grad"], lw=1.6)
    axR.text(0.15, len(rows) - 0.45, "FGSM", color=COL["grad"], fontsize=9, va="center")
    axR.set_ylim(-0.6, len(rows) - 0.3)
    axR.set_xlabel("extra delivery lost vs FGSM (pp)")
    if not PAPER:
        axR.set_title("…but cost more delivery only on some victims")

    # The result splits by critic: mark the boundary between the two groups.
    lc = [i for i, v in enumerate(names) if v.startswith("LC")]
    if lc and lc[0] > 0:
        cut = lc[0] - 0.5
        for ax in (axL, axR):
            ax.axhline(cut, color="0.45", lw=0.9, ls="--")
        axR.text(0.98, cut + 0.06, "local critic", transform=axR.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8, color="0.35")
        axR.text(0.98, cut - 0.06, "central critic", transform=axR.get_yaxis_transform(),
                 ha="right", va="top", fontsize=8, color="0.35")

    # One legend for both panels, below them: every in-axes spot covers a bar.
    handles, labels = axL.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    save(fig, "T9_logit_vs_fgsm")


if __name__ == "__main__":
    import sys
    wanted = set(sys.argv[1:])
    print(f"ROOT={ROOT}  FIG_DIR={FIG_DIR}")
    for fn in (t1, t2, t3, t3_seeds, t4, t5, t6, t7, t8, t9):
        if wanted and fn.__name__ not in wanted:
            continue
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__} FAILED: {e}")
    print("done.")
