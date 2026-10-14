# STATE — 2026-10-14 (evening)

branch main pushed to origin; server code at d209c69 (current).
phase just done: research-do implement + deploy + status E025.

## What changed this session
- src/standalone_experiment_runner.py: rule 'table' (_table_choice, env TABLE_FILE; missing/out-of-range -> k=0); 'table' in _parse_rule_spec
- tools/supervised_signal.py: --export-table (JSON {agent_idx:{dest_idx:k}}, same shrunk-mean rule as 'table')
- tools/run_eval.sh: TABLE_FILE host path mounted at /workspace/table.json
- smoke: SP-86 export 14 agents/137 entries; rule matches file 0/294 mismatches (local python: /usr/bin/python3)
- .mem: E025 card (impl note), runs.md row, STATE

## Active runs
E025 (server, launched 10-14): runs.md rows E025-dc, E025-sp (split so tools/runs_status.sh parses them).
Status at last check (~22:00 server clock): SP-86 table exported, evals n=0/4/8 running (~7 min in);
Deltacom export still running (~47 min), its evals follow. Est. all done ~1.5-2 h after the check.
Done marker ~/idqn/run_E025.done gets 2 lines (runs_status shows SP-86 row 'DONE? no container' during evals = false alarm).
Out: host_data/results/E025/{table_,ss_}<cfg>.json; evals in v2_dc_lc/E025_v2_deltacom, v2_p3_lc_w05/E025_v2_pilot4_lc.
Caveat: SP-86 policy dir v2_p3_lc_w05 is a placeholder (rules ignore the policy); if its eval fails, use another SP-86 LC-Simple dir.

## Latest results
- E024 IPS own-flow Deltacom: ECMP 88.2, greedy 94.1, table 95.8 (K16). E020-E022 Deltacom 8x system PDR: learners ~ECMP 80.4 vs greedy 84.9 (K10).

## Open problem
Why do Deltacom learners stay at ECMP while greedy gains 4.5 pp? Hypothesis (K16): externality; E025 tests it.

## EXACT next step
1. after the .done marker has 2 lines (SP-86 may finish first; can fetch it alone): research-do status -> fetch E025 (rsync json only; summarize_run.py -> .mem/results/E025.json;
   check runs.md/LEDGER row E025 with sha d209c69).
2. research-think analyze E025 using the card's thresholds (<=ECMP+1.3 externality -> A025; >=+2.6 -> optimisation; SP-86 sanity within 1.3 pp of learners). Decides A025 per D14.
Parallel: A027 lit-graph redo in Cowork.

## Read next
experiments/E025-table-for-all.md, decisions.md D14, conclusions.md K10/K16.
