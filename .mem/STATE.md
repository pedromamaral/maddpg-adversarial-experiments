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
E026: maddpg-E026-sp86 + -dc, started 10-15 21:30 (ETA 40-80 h). Partial search_log in host_data/results/E026_{sp,dc} (not interpreted).

## Open problem
Does Deltacom have static headroom that learners miss (E026 > 81.9 / > ECMP by >1 pp -> KILL-4)?

## EXACT next step
E028 implemented (9ff3439, pushed + pulled on server; card experiments/E028-drlor-style.md). Next: pick margin/pretrain cells, then research-do deploy E028 (check tools/runs_status.sh first; server load ~11 from E026).
E026 still running -> when containers exit: research-do fetch E026, research-think analyze E026.

## Read next
runs.md, experiments/E026-system-static-table.md (fetch/analyse) or avenues.md A028 line + codebase.md (E028).
