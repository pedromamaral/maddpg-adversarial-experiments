# E030 — Table warm start under own-flow credit (A036, D23, T1)
Status: RUNNING since 2026-11-05 14:58, code 5599288. Avenue A036. Last Paper 1 experiment beyond E034.
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
