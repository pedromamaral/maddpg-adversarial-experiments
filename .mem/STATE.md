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
E026b fetched, NOT analysed (table n=0: dc 87.35, sp 96.98 raw eval numbers in results/E026b_*.json).

## Plan (D19)
E026b analyse -> E028 (A028 DRL-OR-style, GPU, cells not picked) -> E030 (A036 table warm-start) -> E029 (A025 diff reward).

## Active runs
none.

## EXACT next step
research-think analyze E026b (strong model): H1 from results/E026b_{dc,sp}.json, H2 from results/E026b_credit_{dc,sp}.json;
fill card Results/Interpretation/Verdict, LEDGER line, K11 -> final, A030/A035 status. Then pick E028 cells and research-do deploy.
Note: tools/runs_status.sh needs one name/glob per runs.md container cell (no braces or ';').

## Open problem
Does the Deltacom +7 pp hold on held-out eval and n=4/8 (H1), and is it invisible to own-flow credit (H2, A035)?

## Read next
experiments/E026b-table-eval-credit.md, results/E026b_*.json, conclusions.md (K11, K17), decisions.md D18-D19.
