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
tools/runs_status.sh; when E026 containers exit -> research-do fetch E026, then research-think analyze E026.
If E026 still running: research-do implement E028 (A028 DRL-OR-style port) on a cheaper model.

## Read next
runs.md, experiments/E026-system-static-table.md (fetch/analyse) or avenues.md A028 line + codebase.md (E028).
