# STATE — 2026-10-17

branch main @ d0e5689 (pushed through b0581e1); server at b0581e1.
phase just done: research-do implement+deploy E027, summarize_run reader for E026.

## What changed
- tools/summarize_run.py: reads static_table_search search_log.json (partial logs ok) -> metrics search/<start>/{J0,J,gain_pp,evals,sweeps} (1ce92a4)
- E027 (bafe41b, b0581e1): supervised_signal.py --train-seed; env EVAL_TRAFFIC_SEED; run_eval.sh DOCKER_EXTRA; tools/run_E027.sh

## Plan (D17)
E026 (A030) -> E027 (A034b) -> E028 (A028 DRL-OR-style) -> E029 (A025 diff reward).

## Active runs (server, see runs.md)
E026: maddpg-E026-sp86 + -dc, up ~44 h since 10-15 21:30 (ETA 40-80 h). Partial search_log fetched to host_data/results/E026_{sp,dc} (not interpreted).
E027: tools/run_E027.sh, started 10-17 17:30, 3 Deltacom eval cells, 4 cpus; marks in ~/idqn/run_E027.done.

## Open problem
Does Deltacom have static headroom that learners miss (E026 > 81.9)? Is K17 (table < ECMP) robust to export and eval seed (E027)?

## EXACT next step
research-do status; when E027 .done has 3 lines -> research-do fetch E027 (summarize each v2_dc_lc/E027_* dir); when E026 containers exit -> fetch E026.

## Read next
runs.md, experiments/E027-e025-hardening.md, experiments/E026-system-static-table.md.
