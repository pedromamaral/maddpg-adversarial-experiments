# STATE — 2026-10-14 (evening)

branch main @ 76a7f90 (local ahead of origin by 1 = .mem only; server at d209c69 = code current).
phase just done: research-do implement E025 + deploy E025.

## What changed this session
- src/standalone_experiment_runner.py: rule 'table' (_table_choice, env TABLE_FILE; missing/out-of-range -> k=0); 'table' in _parse_rule_spec
- tools/supervised_signal.py: --export-table (JSON {agent_idx:{dest_idx:k}}, same shrunk-mean rule as 'table')
- tools/run_eval.sh: TABLE_FILE host path mounted at /workspace/table.json
- smoke: SP-86 export 14 agents/137 entries; rule matches file 0/294 mismatches (local python: /usr/bin/python3)
- .mem: E025 card (impl note), runs.md row, STATE

## Active runs
E025 on server, launched 10-14 ~22:00 via ~/idqn/run_E025.sh (export table -> run_eval.sh, n=0/4/8,
RULES="table random+flow greedy greedy_stale2 sp"). Containers maddpg-E025-exp-{v2_deltacom,v2_pilot4_lc}
(evals follow). Done marker ~/idqn/run_E025.done (2 lines when finished). ETA ~1h.
Out: host_data/results/E025/{table_,ss_}<cfg>.json; evals in v2_dc_lc/E025_v2_deltacom and v2_p3_lc_w05/E025_v2_pilot4_lc.
Caveat: SP-86 policy dir v2_p3_lc_w05 is a guess (rules don't use the policy); if its eval fails, pick another SP-86 LC-Simple dir.

## Latest results
- E024 IPS own-flow Deltacom: ECMP 88.2, greedy 94.1, table 95.8 (K16). E020-E022 Deltacom 8x system PDR: learners ~ECMP 80.4 vs greedy 84.9 (K10).

## Open problem
Why do Deltacom learners stay at ECMP while greedy gains 4.5 pp? Hypothesis (K16): externality; E025 tests it.

## EXACT next step
0. Pedro: `git push` (.mem commit 76a7f90). Server code already current.
1. later: research-do status -> fetch E025 (rsync json only; summarize_run.py -> .mem/results/E025.json;
   check runs.md/LEDGER row E025 with sha d209c69).
2. research-think analyze E025 using the card's thresholds (<=ECMP+1.3 externality -> A025; >=+2.6 -> optimisation; SP-86 sanity within 1.3 pp of learners). Decides A025 per D14.
Parallel: A027 lit-graph redo in Cowork.

## Read next
experiments/E025-table-for-all.md, decisions.md D14, conclusions.md K10/K16.
