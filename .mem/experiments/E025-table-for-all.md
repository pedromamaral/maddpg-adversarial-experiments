# E025 — Does the own-flow table help the system when every agent uses it?
Status: PLANNED 2026-10-14. Avenue A029; decision D14; follows E024 (K16); gates A025.

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
  - SP-86 sanity: table-for-all within ~1.3 pp of learners (96.2-97.4); if far below, the export is wrong.
Implemented 544df54 (smoke OK on SP-86: export 14 agents/137 entries; 'table' rule matches file, 0/294 mismatches; local python = /usr/bin/python3).
Usage: supervised_signal.py --config C --train-episodes 150 --export-table T.json; TABLE_FILE=T.json RULES="table ..." tools/run_eval.sh RUN CFG VARIANT "0 4 8" (needs a trained RUN dir for the policy/env).
Results: TODO -> results/E025.json
