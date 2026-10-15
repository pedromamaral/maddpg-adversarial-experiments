# STATE — 2026-10-15

branch main pushed to origin; server code at d209c69 (current).
phase just done: research-think strategy (gate = ITERATE, D15). Before that: implement+deploy E025.

## What changed this session (strategy)
- strategy.md rebuilt from lit-graph strategy-plan rev 0 digest (A027 done): novelty narrowed, KILL-1..4, DRL-OR = A028 target
- avenues: A027 tried; A028 -> DRL-OR port; new A030 (static-table optimiser, needed-for-paper), A031 dropped, A032 parked; A003/A026 dropped
- D15; project.md C3/C5 kill criteria; E025 card note (its thresholds rule)

## Earlier this session (E025 implement)
- src/standalone_experiment_runner.py: rule 'table' (_table_choice, env TABLE_FILE; missing/out-of-range -> k=0); 'table' in _parse_rule_spec
- tools/supervised_signal.py: --export-table (JSON {agent_idx:{dest_idx:k}}, same shrunk-mean rule as 'table')
- tools/run_eval.sh: TABLE_FILE host path mounted at /workspace/table.json
- smoke: SP-86 export 14 agents/137 entries; rule matches file 0/294 mismatches (local python: /usr/bin/python3)
- .mem: E025 card (impl note), runs.md row, STATE

## Active runs
none. E025 fetched 2026-10-15 (both done markers present): .mem/results/E025-dc.json, E025-sp.json (summaries sha=144de42, NOT training sha).
Raw: host_data/results/E025/, v2_dc_lc/E025_v2_deltacom, v2_p3_lc_w05/E025_v2_pilot4_lc. runs.md rows marked done (delete once LEDGER row added).

## Latest results
- E024 IPS own-flow Deltacom: ECMP 88.2, greedy 94.1, table 95.8 (K16). E020-E022 Deltacom 8x system PDR: learners ~ECMP 80.4 vs greedy 84.9 (K10).

## Open problem
Why do Deltacom learners stay at ECMP while greedy gains 4.5 pp? Hypothesis (K16): externality; E025 tests it.

## EXACT next step
1. research-think analyze E025 (use card thresholds; numbers below are system PDR %, n=0/4/8):
   Deltacom: greedy 84.9/76.5/68.9 | random+flow 80.4/72.8/66.8 | table 78.0/70.9/64.6 | sp 74.4/68.2/63.3
   SP-86:    greedy 95.7/93.6/89.1 | random+flow 91.4/89.7/81.7 | table 95.3/90.2/82.7 | sp 75.8/75.7/70.7
   (no interpretation done here; add LEDGER row E025 with training sha d209c69)
2. then research-think ideate -> cards for A030 (M1 wk 1-2) and A028 (M2 wk 2-5).
Paper deadline target ~2027-01-14 (TNSM). Note: ssh alias 'server' not resolvable locally; use pedroamaral@10.26.110.14.

## Read next
experiments/E025-table-for-all.md, decisions.md D14/D15, strategy.md (KILL criteria), conclusions.md K10/K16.
