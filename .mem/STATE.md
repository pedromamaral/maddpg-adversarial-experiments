# STATE — 2026-10-29

branch main @ 2da2600 (pushed; only .mem/STATE.md + runs.md modified locally). Server code at 3f02e7a. Server idle.
phase this session: research-think analyze E028 (K19) + analyze E031 (K20, A037 done, K11 edited) + ideate -> D21, A039 active, E032 card.

## What changed
- tools/table_change_credit.py (new, 598d49d), tools/run_E026b.sh (new, 6d946c8); tools/runs_status.sh unchanged
- experiments/E026b card: IMPLEMENTED + launch cmd (Results/Interpretation/Verdict still empty)
- runs.md: E026b run rows removed (finished, fetched); not yet in LEDGER
- results/E026b_{dc,sp}.json (eval rules n=0/4/8), results/E026b_credit_{dc,sp}.json (per-change sys/own/entry gains); raw in host_data/results/E026b + v2_dc_lc/E026b_dc + v2_p4_lc/E026b_sp

## Latest results
E026 (train J, n=0): SP-86 best static table 97.0 ~ learners 96.2-97.4 (C3 ok);
  Deltacom 87.4 (4 starts 86.9-87.4) vs ECMP 80.4, learners 80.3, greedy 84.9 -> +7 pp static headroom (K11 revised).
E026b ANALYSED (D20, K11 high@n=0, K18): Deltacom held-out 87.35 vs ECMP 80.4 (n=4/8: 77.6/68.0 vs 72.8/66.7); SP-86 96.98. Table decays under failures (below greedy on SP-86). Own-agent net -1.1 vs system +9.4 on Deltacom (coordinated pair), SP-86 own>=0.

## Latest: E028 (K19) DRL-OR-style ~ECMP on Deltacom (79.5); SP-86 94.5 = ECMP+3.1 but ~2 pp < flow-credit DQN.

## Latest: E031 (K20) failure-aware table search: no held-out gain; SP-86 greedy/K14 DQN beat best table at n=4/8 by 1.7/4-5 pp.

## Plan (D19)
E026b done -> E028 (A028 DRL-OR-style, GPU, cells not picked) -> E030 (A036 table warm-start) -> E029 (A025 diff reward).

## Active runs
E028: FINISHED 2026-10-24 04:50, FETCHED 2026-10-28 (results/E028_{dc,sp}.json; raw host_data/results/E028/{dc,sp}_s*). Note: ssh alias "server" is unresolved locally, use pedroamaral@10.26.110.14 for rsync. E031: FINISHED 2026-10-25 09:54, FETCHED 2026-10-28 (results/E031_{dc,sp,eval_dc,eval_sp}.json; raw host_data/results/E031_{dc,sp} + v2_dc_lc/E031_dc, v2_p4_lc/E031_sp).

## EXACT next step
research-do implement E032 (D21; card experiments/E032-shuffle-under-failures.md): first check E018 SP-86 checkpoints exist on server and that eval shuffle works at n>0; add fallback deviation-rate logging. Then E028 margin=0 cell -> E030 (A036) -> E029 (A025).

## Open problem
Is Deltacom's missed headroom credit (own-flow blind) or optimisation (needs a non-greedy first step)? A036/E030 separates. Is the table failure-robust? (no: A037)

## Read next
experiments/E026b-table-eval-credit.md, results/E026b_*.json, conclusions.md (K11, K17), decisions.md D18-D19.
