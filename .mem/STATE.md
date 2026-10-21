# STATE — 2026-10-17

branch main @ "E027: analysed" commit (local, not pushed); server at b0581e1.
phase just done: research-think analyze E027 (K17 hardening).

## What changed
- experiments/E027-e025-hardening.md: results table, interpretation, verdict SUPPORTED
- experiments/LEDGER.md: E027 line
- avenues.md: A034 -> tried
- conclusions.md: K17 revised -> high (Deltacom), evidence E025,E027
- runs.md: E027 moved to Finished

## Latest results
E027: Deltacom 8x table - ECMP < 0 in 12/12 cells (2 exports x 2 eval seeds x n=0/4/8), -1.9..-3.7 pp;
  export spread <=0.12 pp (exports share one IPS data source -> estimation noise only).

## Plan (D17)
E026 (A030) -> E027 done -> E028 (A028 DRL-OR-style) -> E029 (A025 diff reward).

## Active runs (server, see runs.md)
E026: finished (sp86 10-19 09:55, dc 10-21 09:40), fetched 10-21 -> host_data/results/E026_{sp,dc}, .mem/results/E026_{sp,dc}.json (not interpreted). Server idle.

## Open problem
Does E026's Deltacom static headroom (+7 pp train J) hold on held-out eval and under failures (E026b)?

## EXACT next step
E028 implemented (9ff3439, pushed + pulled on server; card experiments/E028-drlor-style.md). E026 has exited (hold lifted); server idle. Pick margin/pretrain cells -> research-do deploy E028 (check tools/runs_status.sh first).
E026 analysed (D18): Deltacom static table 87.4 >> ECMP 80.4 on train J -> research-do implement/deploy E026b (held-out eval of host_data/results/E026_{sp,dc}/table_best.json via run_eval.sh TABLE_FILE, n=0/4/8). E028 may run in parallel.

## Read next
runs.md, experiments/E026-system-static-table.md (fetch/analyse) or avenues.md A028 line + codebase.md (E028).
