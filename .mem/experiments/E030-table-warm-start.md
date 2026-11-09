# E030 — Table warm start under own-flow credit (A036, D23, T1)
Status: DONE 2026-11-06 02:48, analysed 2026-11-09, code 5599288. Avenue A036. Last Paper 1 experiment beyond E034.
Smoke (4 eps, 3 warm, 1-ep tracks): table agree 0.97-1.00 during warm, PDR ~86; one post-warm episode dropped agree to 0.87 (warm far shorter than the real 50 eps, not a result; check the validity gate and spW first).

Hypothesis H1: own-flow credit actively undoes the system-optimal table on Deltacom (K17, K18), so a learner handed the
 E026 table drifts back toward ECMP. H0: the table holds, so credit does not undo it and the problem is reaching it.
Change: --warm-table + --warm-margin (src/idqn_control.py): during --sp-pretrain the behaviour is the table and a hinge makes
 its path the argmax for all (agent, dest); --track-every snapshots greedy n=0 PDR + argmax mean-Q table, RNG restored.
Config: tools/run_E030.sh. Flow-credit DQN, train n=0, fixed hotspot (table's own conditions; E028's failure/random-hotspot
 training would confound), 750 eps incl. 50 warm, track every 100 + at warm end (5 eps), final eval n=0/4/8 (20 paired eps), 3 seeds.
Arms: dcW Deltacom warm, flat eps 0.05 · spW SP-86 warm, flat eps 0.05 · dcD Deltacom warm + eps 1.0->0.05 · dcC Deltacom cold.
Validity gate: track at warm end must be near the table (Deltacom ~87, SP-86 ~97; table_agree high). Otherwise E030 is void and the warm start needs fixing.
Reading, on final n=0 (noise: ~2 pp seed spread on Deltacom):
 - dcW <=82 and spW ~97 -> credit undoes the optimum (H1 supported).
 - dcW <=82 and spW also drifts -> warm start / Q recalibration forgetting, no conclusion on credit.
 - dcW >=86 -> credit holds it; compare dcD: dcD low too -> exploration phase erases it; dcD holds -> cold-start search is the problem.
 - dcW between 82 and 86 -> inconclusive; K18 retention (replay E026_dc/search_log changes against track tables) and A041 next.
Cost: CPU only, ~6 h/cell (E028 scale), 2 chains per seed.

## Results (2026-11-09; final eval 20 paired eps n=0/4/8; track = 5-ep greedy n=0 PDR / table_agree)
| arm | final n=0 (s0 s1 s2) | mean n=0/4/8 | track ep49 -> ep99 -> ep699 (PDR / agree) |
|---|---|---|---|
| dcW warm, eps 0.05 | 80.33 80.92 79.65 | 80.30/72.76/65.58 | 86.3/1.00 -> 78.5-79.8/0.78-0.87 -> 78.5-80.0/0.41-0.51 |
| dcD warm, eps 1->0.05 | 79.92 80.47 80.88 | 80.42/72.66/65.54 | 86.3/1.00 -> 78.0-80.3/0.53-0.55 -> 77.7-79.6/0.46-0.52 |
| dcC cold | 80.48 80.50 80.62 | 80.53/73.01/65.64 | - -> 78.6-79.5 -> 78.5-79.3 |
| spW warm, eps 0.05 | 97.29 96.18 97.37 | 96.95/91.96/84.29 | 97.2/1.00 -> 96.8-97.2/0.56-0.58 -> 97.1-97.4/0.56-0.60 |
Refs: Deltacom table 87.35/77.6/68.0, ECMP 80.4/72.8/66.7; SP-86 table 96.98/91.5/83.8.

## Interpretation
- Validity gate passed: at warm end every seed is on the table (agree 1.00; PDR 86.3 Deltacom, 97.2 SP-86).
- Deltacom: within 50 episodes of own-flow training the warm learner falls to ~79 (-7 pp) and keeps leaving the table (agree 0.4-0.5 by ep 699). Flat or decayed exploration and a cold start all end at the same point, ~ECMP (80.3-80.5, spread <0.6 pp). Start point does not matter.
- SP-86 control: table agreement also falls to ~0.57, but PDR stays at 97: the learner rewrites the table into an equally good one. So the drift is not generic forgetting that costs PDR; it costs PDR only where own-flow and system optima differ (K17, K18).
- Caveat: the hinge (margin 0.05) fixes argmax, not Q calibration; fast first-update rewriting is partly expected, but the SP-86 arm shows it is not destructive by itself.
## Verdict
H1 supported (medium-high, 3 seeds/arm, effect 7 pp vs ~0.6 pp spread): own-flow credit actively undoes the system-optimal table on Deltacom; the learner's fixed point is ~ECMP whatever the start. The Deltacom miss is credit, not search/exploration.
