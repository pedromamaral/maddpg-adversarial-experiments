# STATE — 2026-10-22

branch main @ a10017b (local; 5 .mem-only commits ahead of origin — push before deploy). Server code at 9ff3439 (E028).
phases this session: research-do status + fetch E026, research-think analyze E026, research-think ideate.

## What changed
- runs.md: E026 (sp86 + dc) -> Finished, fetched 10-21; server idle
- results/E026_{sp,dc}.json (gitignored) + host_data/results/E026_{sp,dc}/ (JSON only)
- experiments/E026-system-static-table.md: results, interpretation, verdict (provisional); LEDGER: E026 line
- conclusions.md: K11 revised (Deltacom HAS static headroom; medium, pending E026b)
- decisions.md: D18 (eval E026 before claims), D19 (ideate order)
- avenues.md: A030 updated; A025 KILL-2 also vs E026 gap; new A035 active, A036-A038 proposed
- project.md / strategy.md: KILL-2 reference revised (D19)
- experiments/E026b-table-eval-credit.md: new card (PROPOSED)

## Latest results
E026 (train J, n=0): SP-86 best static table 97.0 ~ learners 96.2-97.4 (C3 ok);
  Deltacom 87.4 (4 starts 86.9-87.4) vs ECMP 80.4, learners 80.3, greedy 84.9 -> +7 pp static headroom (K11 revised).

## Plan (D19)
E026b (A030 eval + A035 credit, CPU) || E028 (A028 DRL-OR-style, GPU) -> E030 (A036 table warm-start) -> E029 (A025 diff reward).

## Active runs
none (server idle).

## Open problem
Does the Deltacom +7 pp hold on held-out eval and n=4/8 (H1), and is it invisible to own-flow credit (H2, A035)?

## EXACT next step
research-do implement E026b (cheaper model): eval = run_eval.sh TABLE_FILE=host_data/results/E026_{sp,dc}/table_best.json
(no code); write tools/table_change_credit.py (per accepted change: system vs own-flow PDR, paired, eval seeds; reuse
own-flow accounting of tools/own_flow_effect.py). Then research-do deploy E026b + E028 (pick E028 margin/pretrain cells).

## Read next
experiments/E026b-table-eval-credit.md, codebase.md, infra.md; for E028: experiments/E028-drlor-style.md.
