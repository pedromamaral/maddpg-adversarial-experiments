# Paper 2 — road to journal submission

Manuscript: `paper/paper2_robustness.tex` (target: IEEE TNSM). It is the journal reframe
of the July short draft `paper2_fgsm.tex`, which it supersedes.

State as of 29 Sep 2026: the draft predates the 24 Sep corrections (gradient masking,
mis-aimed GNN attack, per-victim ceilings, broken T7 policy line). Every number and
claim below comes from `students/goncalo-martins-fgsm-thesis/RESULTS_OUTLINE.md`
(FGSM side) and `students/miguel-chen-learned-adversary/FEEDBACK_2026-09-24.md`
(learned-adversary side), which are the authoritative corrected sources.

---

## 1. The headline has changed

**Old (in the draft):** FGSM is the strongest myopic attack; it flips a quarter of
decisions yet takes at most 2.8 of ~21 pp; a learned adversary does no better, so
K-path redundancy is a structural defence (H1).

**New (what the evidence supports):** — *every attack number below was measured under
the legacy threat model (all 94 features perturbed; see §2) and will be replaced by the
`path_util` re-runs. The structure of the argument should survive; the numbers will not.*

1. Observation attacks change many decisions and a minority of outcomes. Measured
   against each victim's own damage ceiling (7.3–21.4 pp), FGSM extracts 0–30 %, the
   best gradient attack at most ~35 % (CC-Duelling, 5.8 of 16.5 pp), and the learned
   adversary at most about half, on one victim with every step attacked (LC-Simple,
   5.90 of 11.3 pp); no timing-gated attack exceeds 13 %.
2. FGSM is gradient-masked: the chosen outputs are saturated (σ > 0.99 in 82–96 % of
   decisions). Iterating the same objective does not help; moving it to the logits
   does, on three of seven victims. The limit was the objective, not the step count.
3. A GNN encoder damps undirected noise (0.6–1.9 % flips) but not an attack aimed
   through it (7–19 %).
4. Under link failures, FGSM does no better than random noise. (Untested for the
   logit attack; see E4.)
5. No architectural robustness ordering survives retraining.

The durable contribution is the **methodology** (decision vs outcome, budget-matched
random control, per-victim ceiling, seed replication, masking check), and a result
that is weaker than H1 but defensible: no attacker in a family of increasing power
extracted a majority of the available damage.

---

## 2. Experiments

### Agreed plan (29 Sep 2026)

**Threat-model fix — found 29 Sep, everything else depends on it.** The attack
perturbed all 94 observation features by ±ε and clamped only the first four to [0,1]:
it also moved the timestep, the queue state and the one-hot destination flags, which no
compromised telemetry channel can reach, and left most features free to leave their
physical range. The paper describes a utilisation-only attacker, and that is now the
only threat model in the code: every attacker perturbs just the 61 per-path utilisations
(`NetworkEngine.path_util_slots`, slots 33–93), clamped to [0,1]. The all-feature
attacker was removed rather than kept as an option. Every reported attack number must
be re-measured; the legacy numbers (fgsm_tighten, logit_attack, seed_probe,
partial_compromise*, mchen) can serve as a contrast showing how much of the old damage
came from unreachable features.

| Stage | Run | Status |
|---|---|---|
| A1 | `threat_util`: budget sweep ε ∈ {0.1, 0.2, 0.3, 0.5, 1.0} × {logit_margin, logit_congestion, FGSM, random}, 7 canonical victims, nominal cell (`configs/probe_threat_util_sweep.json`) | running since 29 Sep 18:41 UTC, ~12 h |
| A2 | `threat_util_iter`: MI-FGSM (n=20, α=ε/8, μ=1) on both logit objectives, ε=0.3 (E3) | queued after A1 |
| B | Replication under `path_util`: 2 extra seeds × 7 variants (GNN seeds are trained) (E1, E2); logit vs random under failures n=0,2,4 (E4); partial compromise re-run | after A1 is checked |
| C | Learned adversary, re-implemented as PA-AD (director over target paths + logit-margin actor); nominal, full compromise, 4 non-GNN + CC-Duelling-GNN, 2 seeds | after A1 |
| D (stretch) | Link-level telemetry compromise (consistent across agents) + cross-agent consistency defence | after C |

Not planned: a second topology (keep for revision: 2 variants on one contrasting
topology, if a reviewer asks), certified bounds (vacuous at ε=0.3), adversarial training.

Runs go through `tools/run_probe_grid.sh RUN configs/probe_*.json [MODELS] [VARIANTS]`
on the server, from the repo root.

### Original gap list

In priority order. E1–E4 close gaps a reviewer will find; E5–E6 tighten numbers.

| # | Experiment | Why | Cost |
|---|---|---|---|
| E1 | Attack the GNN seed victims (`seeds_gnn/`, s1042, s2042) with `faithful_gnn`: FGSM, random, both logit arms, nominal cell | GNN rows of T3b rest on one victim; last open item of the 24 Sep list | Small, once training is done (4/6 finished on 24 Sep — check) |
| E2 | Logit attacks on the non-GNN seed victims (s1042, s2042) | "FGSM is masked on 3/7" is a single-victim claim, the same trap the per-variant ordering fell into | Small |
| E3 | Iterated attack on the **logit-margin** objective (PGD / MI-FGSM), all 7 victims, 15 paired episodes | First reviewer question: we iterated the masked objective only. This defines "strongest gradient attack" | Small; PGD machinery exists |
| E4 | Logit attack under failures (n = 0, 2, 4) | The failure-regime conclusion is only established for FGSM, which we now know is masked | Medium |
| E5 | Damage ceilings on the same 15 episodes as the attack arms (and on seed victims) | Fractions of ceiling are "approximate" (20 vs 15 episodes, different draws) | Small; rule rollouts only |
| E6 | Nominal probe under a second traffic seed | Threats-to-validity lists a single traffic seed | Small (optional) |

Not planned: a learned adversary built on the logit margin (Miguel's SAJA direction);
it goes in future work.

---

## 3. Manuscript revisions (`paper2_robustness.tex`)

- [ ] **Abstract.** Rewrite around §1. Drop "open scaffold" / "central open question"
      (the learned results exist). Per-victim ceilings; masking; learned-adversary count.
- [ ] **Contributions.** Remove "FGSM is at or near the strongest myopic attacker".
      Add the masking finding. Learned adversary is a result, not a scaffold.
- [ ] **Taxonomy table.** All rows completed; add logit (1-step) and iterated-logit rows;
      add partial compromise on the scope axis.
- [ ] **Threat model.** State exactly which features the attacker reaches (the 61 per-path
      utilisations, clamped to [0,1]) and why local switch state is out of reach; add the
      legacy all-feature attack as a contrast if the difference is large.
- [ ] **Method.** Per-victim ceiling table; t-intervals (t₀.₉₇₅,₁₄ = 2.145); white-box
      GNN attack needs the encoder and every agent's clean observation.
- [ ] **§iterated → rewrite:** iterating doesn't help (T8); new subsection on gradient
      masking and the logit attack (T9, CW-style margin loss).
- [ ] **Redundancy absorbs flips.** T2 numbers (flips 7.2–25.7 %, ≤ 2.8 pp,
      CC-Simple-GNN −1.39). Promote the path-diversity mechanism into results: 8–22 %
      of flips are vacuous, 34–47 % of real flips go to a worse path, 36–48 % of
      decisions already sit on the worst path.
- [ ] **GNN subsection (new).** T6_gnn_noise_vs_attack.
- [ ] **Partial compromise (new).** 1/4/7/14 agents, 4 draws each: damage grows with the
      number compromised; one agent ≈ nothing.
- [ ] **Failure regime.** Scope to FGSM unless E4 is run.
- [ ] **Learned-adversary results.** Replace GNN rows (corrected FGSM arm, via
      `tools/recompute_learned_vs_fgsm.py`). Recount: 11 of 70, 10 of them on
      CC-Simple-GNN where FGSM itself improves delivery; once where FGSM does damage.
      Weaken the H1 argument: on CC-Duelling(-GNN) a better-aimed single step beats
      both FGSM and the learned adversary.
- [ ] **Threats to validity.** GNN seeds (E1); ceiling episodes (E5); keep the rest.
- [ ] **Discussion / conclusion.** New headline. Add the methodological guidance: robustness
      evaluations of these controllers need an unsaturated (logit/margin) objective.
- [ ] **Figures.** Regenerate T1–T9 with `PAPER_MODE=1 FIG_DIR=paper/figures`; use T3b
      (not T3); new T6; add T8, T9. Fix T5's 6-link point (collapse regime).
- [ ] **Related work and bibliography.** Currently ~11 refs, four of them "Anon."
      placeholders. Needs real author lists, and at minimum: Carlini & Wagner 2017 (margin
      loss), Athalye et al. 2018 (obfuscated gradients), Madry et al. 2018 (PGD),
      Dong et al. 2018 (MI-FGSM), Tramèr et al. 2020 (adaptive attacks), and the
      DRL/GNN-routing line. A TNSM paper expects ~35–50 references.

## 4. Decisions for Pedro

- **Authorship.** Section 6 reports Miguel Chen's 28 trained adversaries; the author list
  is Amaral and Martins only.
- **Companion citation.** Status of Paper 1 (submitted / accepted) for `\cite{companion}`.
- **Where the paper compiles.** Paper 1 moved out of the repo for compilation. If Paper 2
  follows, figures must travel with it; if it stays, `paper/figures/` should be tracked
  (it is gitignored today).
