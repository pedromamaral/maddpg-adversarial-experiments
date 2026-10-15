# E025 — Does the own-flow table help the system when every agent uses it?
Status: DONE 2026-10-15 (analysed). Training/eval sha d209c69. Avenue A029; decision D14; follows E024 (K16); gates A025.

Hypothesis: on Deltacom 8x, the E024 own-flow table (+7.6 pp own-flow over ECMP under an ECMP
background) gives system PDR ~ ECMP (80.4) when all agents use it -> its gain is eaten by
externality (congestion on the shared "best" paths). On SP-86 3x it should match the learners (~96-97).

Change (minimal, behind a switch; baselines unchanged):
  1. tools/supervised_signal.py: --export-table <path> -> JSON {agent: {dest: k}} built exactly like the
     'table' rule (best path per (agent,dest) by shrunk mean r on TRAIN decisions; prior weight 5).
  2. new rule 'table' in the rule-action path used by tools/run_eval.sh (runner._rule_action), reading the
     file from env TABLE_FILE; missing (agent,dest) -> path 0. Respect pinning per flow like other rules.
  3. Optional: agreement of the trained learners' argmax (dc_ac_local_rhf etc.) with the exported table.
Config: configs/v2_deltacom.json (LOAD=8), configs/v2_pilot4_lc.json (LOAD=3); paired eval, 20 eps,
  traffic seed 20240601, n = 0/4/8; RULES="table random+flow greedy greedy_stale2 sp".
Success criterion / reading (Deltacom n=0, seed noise ~1.3 pp; Deltacom rhf seeds 3-5 pp):
  - table-for-all <= ECMP + 1.3      -> externality confirmed -> run A025 (difference reward).
  - table-for-all >= ECMP + 2.6, ~greedy -> learners under-optimised (failed to find the table);
    A025 is the wrong target -> investigate optimisation instead.
  - in between -> inconclusive; report, rerun with more eps.
  (strategy-plan KILL-1 'table >= greedy - 1 SE -> drop A025' is the upper branch above; these thresholds rule, D15.)
  - SP-86 sanity: table-for-all within ~1.3 pp of learners (96.2-97.4); if far below, the export is wrong.
Implemented 544df54 (smoke OK on SP-86: export 14 agents/137 entries; 'table' rule matches file, 0/294 mismatches; local python = /usr/bin/python3).
Usage: supervised_signal.py --config C --train-episodes 150 --export-table T.json; TABLE_FILE=T.json RULES="table ..." tools/run_eval.sh RUN CFG VARIANT "0 4 8" (needs a trained RUN dir for the policy/env).
Results (system PDR %, n=0/4/8; results/E025-dc.json, E025-sp.json; 1 table export, 20 paired eps):
  Deltacom 8x: table 78.0/70.9/64.6 | ECMP 80.4/72.8/66.7 | greedy 84.9/76.5/68.9 | stale2 84.1/75.8/68.5 | sp 74.4/68.2/63.3
  SP-86 3x:    table 95.3/90.2/82.7 | ECMP 91.4/89.7/81.7 | greedy 95.7/93.6/89.1 | stale2 93.7/90.7/86.1 | sp 75.8/75.7/70.7
Interpretation:
  - Deltacom: table-for-all = ECMP -2.4 (n=0), -1.9, -2.1 -> BELOW ECMP at every n (> seed noise 1.3, eval noise <=1).
    The own-flow-best table (+7.6 pp own-flow vs ECMP background, E024) loses system PDR when everyone uses it:
    selfish routing / price of anarchy. Learners (rhf 80.3) sit 2.3 above the table, i.e. at ECMP: they do not
    converge to the off-policy own-flow table but to something spread-out.
  - SP-86 sanity: table 95.3 vs learners 96.2-97.4 at n=0 (0.9-2.1 below; within the card's ~1.3 band of the low
    seed) -> export OK. Under failures table drops faster than learners (n=4 90.2 vs 92.7-93.0; n=8 82.7 vs 86.7-87.0).
    Caveat: not checked how the 'table' rule handles a failed chosen path (k fixed vs masked) -> may explain gap.
  - KILL-1 not triggered (table far below greedy - 1 SE).
Verdict: hypothesis SUPPORTED (stronger than predicted: below, not ~equal to, ECMP) -> externality confirmed; A025 gate open (D14, D16).
Limits: one table export (export noise unmeasured), one eval seed; the -2.4 is consistent across n, so sign is safe.
