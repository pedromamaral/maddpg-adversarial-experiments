# STATE — 2026-10-14 20:40 UTC

branch main @ 6e29524 (server same). Everything below is UNCOMMITTED.
phase just done: E024 status+fetch -> analyze E024 -> strategy gate (iterate) -> ideate -> A029/E025 planned.

## What changed this session
- tools/summarize_run.py: parses supervised_signal.py format ({rule:{own_delivery_pct,se}, agree_greedy})
- results/E024.json written (NOTE: .gitignore `results/` also ignores .mem/results/ -> add `!.mem/results/`?)
- E024 card (results+verdict), LEDGER E024, runs.md (E024 -> Finished)
- conclusions.md K16; project.md C4 supported / C5 reframed; avenues A024 tried, A025 gated, A027/A028 needed-for-paper, A029 active
- decisions.md D14 (supersedes D11 gate); strategy.md gate = ITERATE; new card experiments/E025-table-for-all.md

## Active runs
none

## Latest results
- E024 IPS own-flow Deltacom: ECMP 88.2, greedy 94.1, table 95.8, util 96.7, full 94.8 (SE ~1.4);
  SP-86 all rules 99.3-100.7 vs ECMP 93.3 (K16)
- E020-E022 Deltacom 8x system PDR: learners ~ECMP 80.4 vs greedy 84.9 (K10)

## Open problem
Why do Deltacom learners stay at ECMP while greedy gains 4.5 pp?
Best hypothesis (K16): own-flow objective has no state headroom; greedy's gain is externality, which
per-decision own-flow credit cannot see. Untested caveat: E024 is off-policy (ECMP background).

## EXACT next step
0. Pedro: commit (`git add .mem tools/summarize_run.py && git commit -m "E024 analysed (K16); E025 planned (D14)"`).
1. /model cheaper -> research-do implement E025 (card: --export-table in tools/supervised_signal.py,
   'table' rule via TABLE_FILE in runner._rule_action / tools/run_eval.sh; smoke test).
2. research-do deploy E025 -> fetch -> research-think analyze E025 (decides A025 per D14).
Parallel: A027 lit-graph redo in Cowork (academic-research + research-strategist); Pedro agrees D13 reframing.

## Read next
experiments/E025-table-for-all.md, codebase.md, decisions.md D14; src by rg -n "_rule_action" first.
