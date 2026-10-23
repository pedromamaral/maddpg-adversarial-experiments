# STATE — 2026-10-23

branch main @ cfa143c (pushed). Server code at 6d946c8 (tools-only diff since; src unchanged). Server idle.
phases this session: research-do implement + deploy + status + fetch E026b. No interpretation done.

## What changed
- tools/table_change_credit.py (new, 598d49d), tools/run_E026b.sh (new, 6d946c8); tools/runs_status.sh unchanged
- experiments/E026b card: IMPLEMENTED + launch cmd (Results/Interpretation/Verdict still empty)
- runs.md: E026b run rows removed (finished, fetched); not yet in LEDGER
- results/E026b_{dc,sp}.json (eval rules n=0/4/8), results/E026b_credit_{dc,sp}.json (per-change sys/own/entry gains); raw in host_data/results/E026b + v2_dc_lc/E026b_dc + v2_p4_lc/E026b_sp

## Latest results
E026 (train J, n=0): SP-86 best static table 97.0 ~ learners 96.2-97.4 (C3 ok);
  Deltacom 87.4 (4 starts 86.9-87.4) vs ECMP 80.4, learners 80.3, greedy 84.9 -> +7 pp static headroom (K11 revised).
E026b ANALYSED (D20, K11 high@n=0, K18): Deltacom held-out 87.35 vs ECMP 80.4 (n=4/8: 77.6/68.0 vs 72.8/66.7); SP-86 96.98. Table decays under failures (below greedy on SP-86). Own-agent net -1.1 vs system +9.4 on Deltacom (coordinated pair), SP-86 own>=0.

## Plan (D19)
E026b done -> E028 (A028 DRL-OR-style, GPU, cells not picked) -> E030 (A036 table warm-start) -> E029 (A025 diff reward).

## Active runs
none.

## EXACT next step
research-do: pick E028 cells (A028 DRL-OR-style, GPU) and deploy; optionally A037 failure-aware table search (CPU) alongside. Card: experiments/E028-drlor-style.md.

## Open problem
Is Deltacom's missed headroom credit (own-flow blind) or optimisation (needs a non-greedy first step)? A036/E030 separates. Is the table failure-robust? (no: A037)

## Read next
experiments/E026b-table-eval-credit.md, results/E026b_*.json, conclusions.md (K11, K17), decisions.md D18-D19.
