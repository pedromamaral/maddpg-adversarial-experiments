# MSc Thesis Guide — Adversarial Robustness of MADDPG Routing under FGSM Observation Attacks

*A working guide for the student. It gives the story arc, the concepts to explain,
the results to report, and how to read each figure. All numbers below are from the
final 15-episode "tightening" run (paired episodes, 7 variants, failure sweep
n=0,2,4,6 at 2× hotspot); the plot script `tools/plot_thesis.py` regenerates every
figure from the raw JSON.*

---

## 1. The one-sentence thesis

> A gradient-based observation attack (FGSM/PGD) **successfully changes a large
> fraction of the routing agents' path decisions, yet barely degrades delivery on
> a well-provisioned service-provider topology — because K-shortest-path
> redundancy absorbs the changed decisions. That absorption capacity shrinks as
> link failures remove redundancy, at which point delivery does collapse under
> perturbation — but a budget-matched *random* (non-adversarial) perturbation is
> just as damaging, so the failure-regime fragility is generic observation-noise
> sensitivity, not adversarial exploitability.**

Everything in the thesis is in service of that sentence. The three empirical acts
below each nail one clause of it.

---

## 2. Background the thesis must establish (Chapters 2–3)

The student should be able to explain each of these clearly, because the results
only make sense against them.

**2.1 The routing problem as a multi-agent MDP.** PE (provider-edge) switches are
the learning agents. Each agent, for every destination, chooses one of `K=3`
pre-computed paths (path 0 = shortest). The action is a per-destination `arg max`
over `K` path-logits. This *discrete, per-destination argmax* is the crux of the
whole thesis: the attack has to move a continuous observation enough to flip a
discrete argmax.

**2.2 The observation the agent sees.** The critical part of the state is
`U_i ∈ [0,1]^(|D|×K)` — the *bottleneck (max-over-hops) utilisation* of each of the
`K` candidate paths to each destination. This is what lets an agent prefer a less
congested path. It is also exactly what the attack perturbs.

**2.3 The threat model — observation-space perturbation.** The attacker cannot
change the network, the traffic, or the reward; it can only add a bounded
perturbation to the *observation vector* an agent reads, within an L∞ budget ε
(so no single feature moves by more than ε), with the bandwidth features kept in
their valid `[0,1]` range. This models a compromised telemetry/monitoring channel.

**2.4 FGSM and PGD (how the attack works).** This is Act 1's mechanism; explain it
carefully.
- Build a differentiable **loss** whose gradient points toward "worse routing":
  the *packet-loss objective* weights each path by a sharpened function of its true
  bottleneck utilisation and rewards putting policy probability mass on the
  **congested** paths (`loss = Σ π(path)·sigmoid((util−0.5)·10)`).
- **FGSM** (single step): perturb the observation by `ε·sign(∇_obs loss)` — one
  step to the corner of the ε-ball that most increases the congestion objective.
- **PGD** (iterated): repeat the gradient step, re-projecting into the ε-ball, to
  find a better corner. PGD is the "we tried harder" upper bound on attack
  strength.
- Both are **white-box**: they need the actor's gradients.

**2.5 The random control (the non-adversarial baseline).** Same threat model, same
ε, same domain clamping, but the perturbation direction is a **random** ±ε per
feature instead of the gradient direction. It is *not* an adversarial attack — it
is the **blind, zero-knowledge** member of the same attacker family. Framing to use
in the thesis:

| control | attacker knowledge | perturbation direction |
|---|---|---|
| **random** | blind (no policy access) | random corner of the ε-ball |
| FGSM | white-box, 1 gradient step | optimised corner |
| PGD | white-box, iterated | best optimised corner |

The single most important derived quantity in the thesis is:

> **adversarial-specific effect = (drop under gradient attack) − (drop under random control).**
> It isolates *the value of gradient knowledge to the attacker*. ≈0 means the attack
> is no better than blind noise; ≫0 means a genuine targeted vulnerability.

**2.6 The two robustness diagnostics.**
- **Action-flip rate**: the fraction of per-(agent, destination) argmax path
  choices that the perturbation changes. Measures whether the attack moves the
  *policy*. (Distinct from PDR, which measures whether it moves the *outcome*.)
- **Damage ceiling**: replace the policy with fixed oracle routing rules on the
  same traffic — `worst` (always most-congested path) is the maximum damage *any*
  observation attacker could cause by steering decisions; `greedy` (least-utilised)
  is a strong benign reference. The clean→worst gap is how much delivery is
  *theoretically* at stake at each operating point.

---

## 3. The results, act by act (Chapters 4–5)

### Act 1 — FGSM works: it flips decisions (mechanism validated)

**Claim:** the attack is not broken or under-powered; it genuinely moves the policy.

**Evidence (CC-Simple, 2× hotspot, no failures):**
- Action-flip rate rises monotonically with budget: ~8% at ε0.05 → ~14% at ε0.1 →
  ~22% at ε0.2 → **~26% at ε0.3** (25.7% in the tightening run). PGD flips a
  comparable amount.
- At the *same* ε0.3 the **random control flips only ~8%**. So the gradient attack
  flips ~**3×** more decisions than blind noise — proof the gradient is doing
  targeted work, finding and crossing argmax boundaries, not jostling.

**Figure T1** — *flip rate vs ε*, gradient vs random, CC-Simple. Two rising curves;
the gradient curve sits ~3× above random. Caption: "FGSM reliably flips routing
decisions, and the gradient direction matters — it flips far more decisions than a
budget-matched random perturbation."

**What the student explains here:** *why* single-step FGSM can still flip a discrete
argmax (it optimises the softmax mass toward a congested path until it crosses the
decision boundary), and why PGD helps only marginally (the margins it needs to
cross are small; one good step already crosses most of them).

### Act 2 — The topology absorbs the flips (robustness at nominal)

**Claim:** flipping a fifth-to-a-quarter of all path decisions costs at most a few
points of delivery — a small fraction of what is theoretically at stake.

**Evidence (2× hotspot, no failures, 15 paired episodes):**
- The *absolute* PDR loss under FGSM (ε0.3) is small on **every** architecture: from
  ~0pp (CC-Simple, CC-Simple-GNN, LC-Duelling) up to **2.8pp** (LC-Simple). Compare
  that to the ~21pp damage ceiling (T7): even the most-affected variant reaches
  ~13% of the extractable damage; the robust ones reach ~0.
- The paired gradient−random gap, however, is **not zero on all architectures.** On
  four of seven variants FGSM loses *significantly* more delivery than the
  budget-matched random control — a small but genuine adversarial signal:
  **LC-Simple +3.3pp (CI [+2.0,+4.6])**, **CC-Duelling +2.5pp ([+1.1,+3.9])**,
  **LC-Duelling +1.2pp ([+0.6,+1.9])**, **CC-Duelling-GNN +1.0pp ([0.0,+2.0])**.
  On the other three — **CC-Simple (+0.4pp, CI [−0.3,+1.0])**, **CC-Simple-GNN
  (~0)**, **LC-Duelling-GNN (~0)** — the gap includes zero: no better than noise.
- So the honest Act-2 statement has two clauses: (i) *operationally* the topology
  absorbs the attack everywhere (≤2.8pp of 21pp), and (ii) *statistically* the
  gradient direction does buy a real, if small, advantage on the duelling / LC-Simple
  architectures. Do not overstate to "no adversarial effect" — say "a real but
  operationally minor adversarial effect, capped by topology redundancy."

**Mechanism (the heart of the thesis):** a flipped decision moves a flow from its
chosen K-path to a *different* K-shortest path. On a well-provisioned SP topology
with `K=3` hop-diverse paths and spare capacity, the alternate path is of similar
quality and still delivers. Redundancy converts most "changed decisions" into "same
outcome"; the residual few that the gradient targets well produce the small
significant gap on the exploitable architectures. The damage ceiling confirms the
headroom exists but is barely tapped: an omniscient `worst`-path attacker could
extract ~21pp here; FGSM extracts ≤2.8pp.

**Figure T2** — *decisions changed vs delivery lost*, all 7 variants at nominal: a
bar/scatter showing high flip rate (15–26%) against ~0 PDR drop. Caption: "Adversarial
perturbation changes many decisions but almost no packets: the routing structure
absorbs the flips."

**Figure T7 (optional but strong)** — *damage ceiling vs achieved damage*: the
clean→worst gap (~21pp, what's at stake) next to the FGSM/PGD achieved drop (~0pp)
and the greedy reference. Caption: "The vulnerability exists in principle but the
gradient attack reaches ~2% of it at nominal load."

### Act 3 — Absorption erodes with failures, but the damage is not adversarial

**Claim:** remove the redundancy (link failures) and the same perturbation starts to
degrade delivery — but a *random* perturbation degrades it as much or more, so it is
noise-fragility, not adversarial exploitation.

**Evidence (CC-Simple, 2× hotspot, random link failures n = 0,2,4,6):**
- Clean PDR falls as failures remove capacity: ~88% (n0) → ~86% (n2) → ~78% (n4) →
  **~9% (n6, network essentially collapses on its own)**. The attack-relevant window
  is **n = 0–4**.
- Attack-induced drop grows with failures: negligible at n0, then large at n2–n4.
- **The nominal adversarial signal does not survive the failure regime.** At n2 the
  gap goes *negative and significant* — gradient drops only ~0.3pp while the random
  control drops **~6.5pp** (gap −6.1pp, CI excludes 0): blind noise is *worse* than
  the attack. At n4 both collapse together (~33pp gradient, ~30pp random); the +3pp
  gap is **not significant** at 15 episodes — well inside the failure-driven variance.
  So the direction the gradient found at nominal stops mattering once redundancy is
  gone; magnitude (any perturbation) dominates.
- This pattern is consistent across architectures: at n2 the CC variants all show
  random ≥ gradient; at n4 all seven collapse ~28–33pp with no significant gap; n6
  is degenerate (identical numbers for every variant — the network is failure-dead
  regardless of policy or attack).
- Interpretation: with few surviving paths, *any* ε-perturbation that changes a
  decision can shove a flow onto an overloaded survivor; the gradient's targeting
  buys nothing over random flailing — in fact the random direction is slightly
  worse because it is unstructured.

**Figure T4** — *the money figure*: PDR drop vs number of failed links, one curve
for gradient, one for random, with 95% CI bands, CC-Simple. The two curves rise
together as failures increase; random ≥ gradient throughout. Caption: "Robustness
to decision-flips decreases as failures remove path redundancy; but a
non-adversarial random perturbation is equally damaging, so the failure-regime
fragility is generic observation-noise sensitivity, not adversarial exploitability."

**Figure T5** — *flip rate vs failures*: the flip rate actually **rises** under
failure (post-failure states are more sensitive), yet — cross-referencing T4 — the
extra flips are not adversarially productive. Caption: "The attack flips *more*
decisions under failure, yet gains no adversarial advantage over random noise."

### Act 4 — Architecture and the GNN (secondary, ties to Paper 1)

Two architecture-level observations worth a short chapter/section:

- **Adversarial-gap ranking at nominal (Figure T3).** With paired CIs, four of seven
  architectures show a small *significant* adversarial gap and three do not. Ranked:
  **LC-Simple +3.3pp**, **CC-Duelling +2.5pp**, **LC-Duelling +1.2pp**,
  **CC-Duelling-GNN +1.0pp** (all CIs exclude 0); **CC-Simple +0.4pp**,
  **CC-Simple-GNN ~0**, **LC-Duelling-GNN ~0** (CIs include 0). The pattern: the
  **duelling** head is consistently more exploitable than the **simple** head under
  a central critic, and CC-Simple — the simplest, best-delivering architecture from
  Paper 1 — is also among the most adversarially robust. (Caveat for the student:
  these gaps are real but small; frame them as an architecture *ranking of a minor
  effect*, not a "vulnerability," since absolute loss is ≤2.8pp everywhere.)
- **GNN decision-level robustness (Figure T6), with one exception.** Two of the
  three GNN variants flip only **1.4–2.6%** of decisions (CC-Simple-GNN 1.4%,
  LC-Duelling-GNN 2.6%) vs 18–26% for non-GNN — the message-passing encoder smooths
  the observation so per-agent perturbations barely change decisions. But
  **CC-Duelling-GNN is an exception: it still flips 12.6%** and carries the small
  significant adversarial gap noted above, so the GNN's decision-robustness is
  *base-architecture-dependent, not a property of GNN encoding per se* — warn the
  student not to state it as universal. Where it does hold, the encoder that *cost*
  delivery in Paper 1 *buys* decision robustness here: a security/performance
  trade-off to discuss.

---

## 4. Figure list (what the plot script produces)

All written by `tools/plot_thesis.py` into `thesis/figures/` as PNG + PDF.

| fig | file | shows | data source |
|---|---|---|---|
| T1 | `T1_flip_vs_epsilon` | flip rate vs ε, gradient vs random (CC-Simple) | `fgsm_probe` (budget) |
| T2 | `T2_flips_vs_pdr_nominal` | decisions changed vs delivery lost, 7 variants @ nominal | `fgsm_tighten` (fail0) |
| T3 | `T3_adversarial_gap_by_variant` | gradient−random gap @ nominal, 7 variants, 95% CI | `fgsm_tighten` (fail0) |
| T4 | `T4_failure_fragility` | PDR drop vs #failures, gradient vs random, CI bands | `fgsm_tighten` (n=0,2,4,6) |
| T5 | `T5_flip_vs_failures` | flip rate vs #failures | `fgsm_tighten` |
| T6 | `T6_gnn_flip_robustness` | flip rate GNN vs non-GNN | `fgsm_tighten`/`fgsm_full` |
| T7 | `T7_damage_ceiling_contrast` | achievable (worst) vs achieved (FGSM) damage vs load | `sweep_baselines` + probes |

---

## 5. Suggested thesis chapter structure

1. **Introduction** — routing as RL; why adversarial robustness matters for
   learned control planes; the question: *can an observation attacker degrade a
   MADDPG routing policy, and if so, how and when?*
2. **Background** — MADDPG/CTDE, the PE/K-path routing model, observation-space
   adversarial attacks (FGSM/PGD), the SP topology. (§2 above.)
3. **Methodology** — threat model, the packet-loss objective, FGSM/PGD, the random
   control, the flip-rate and damage-ceiling diagnostics, the evaluation grid
   (load × failures), paired-episode statistics and CIs.
4. **Results I — the attack works but the network absorbs it** (Acts 1–2, T1–T2, T7).
5. **Results II — failure-regime fragility and the random-control verdict**
   (Act 3, T4–T5).
6. **Results III — architecture and the GNN** (Act 4, T3, T6).
7. **Discussion** — redundancy as an implicit adversarial defence; the difference
   between changing *decisions* and changing *outcomes*; implications for deploying
   learned routing (robust while provisioned, fragile to *any* observation noise
   once degraded — argues for telemetry integrity + fast reconvergence, not
   adversarial training).
8. **Conclusion & future work** — black-box/transfer attacks, attacks that target
   the *worst* path directly (closing the gap to the damage ceiling), online graph
   reconstruction for the GNN, multi-step temporal attacks.

---

## 6. Key numbers to quote (final, from the 15-episode tightening run)

- Flip rate at ε0.3: gradient ~26% vs random ~8% (≈3×). *(CC-Simple, nominal.)*
  Across variants, non-GNN flip 18–26%; GNN flip 1.4–2.6% **except CC-Duelling-GNN
  at 12.6%** (GNN robustness is not universal).
- Nominal absolute PDR loss under FGSM (ε0.3): **≤2.8pp on every architecture**
  (LC-Simple worst at 2.8pp; CC-Simple/CC-Simple-GNN/LC-Duelling ~0).
- Nominal gradient−random gap (adversarial-specific effect), with 95% CI:
  **LC-Simple +3.3 [+2.0,+4.6]**, **CC-Duelling +2.5 [+1.1,+3.9]**,
  **LC-Duelling +1.2 [+0.6,+1.9]**, **CC-Duelling-GNN +1.0 [0.0,+2.0]** → significant;
  **CC-Simple +0.4 [−0.3,+1.0]**, **CC-Simple-GNN ~0**, **LC-Duelling-GNN ~0** → n.s.
- Damage ceiling at 2× hotspot: policy ~87% vs worst-path ~66% → **~21pp at stake**,
  ≤2.8pp reached by FGSM (≤13% of the ceiling).
- Failure sweep gaps (grad−rand): **n2 gap negative** (CC-Simple −6.1pp, CI excludes
  0 — random *worse*); **n4 both collapse ~28–33pp, gap n.s.**; **n6 degenerate**
  (clean ≈ 9%, identical for every variant).
- Architecture takeaway: duelling > simple in exploitability under a central critic;
  CC-Simple robust; GNN encoder makes decisions near-unflippable (1.4–2.6% flips) for
  two of three bases — CC-Duelling-GNN excepted (12.6%) — at the Paper-1 cost of
  delivery: a base-dependent security/performance trade-off.

---

## 7. Pitfalls to warn the student about

- **Do not report the −25/−30pp failure-regime drop as an "adversarial" result**
  without the random control beside it — that is the exact overclaim this thesis
  exists to prevent.
- **Flip rate ≠ vulnerability.** A high flip rate with flat PDR is *robustness*, not
  a successful attack. Keep the two axes (decisions vs outcomes) separate throughout.
- **`n=6` (and `n=8`) failures collapse the network on their own** (clean PDR ≈ the
  same tiny number for every variant) — these points are failure-dominated, not
  attack-informative. Report them only to show where the usable window ends.
- **"random" is a control, not an attack** in the adversarial sense — call it the
  non-adversarial / blind baseline (see §2.5).
- Always use **paired** episodes (same traffic seed + same failure draw for clean,
  gradient, and random) so the gap CIs are valid.
