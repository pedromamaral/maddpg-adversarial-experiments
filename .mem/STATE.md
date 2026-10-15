# STATE — 2026-10-15

branch main @ 46f4985 (+ this STATE commit), not pushed; server code at d209c69.
phase just done: research-do implement E026 (tools/static_table_search.py; step 0 = no change needed).

## What changed
- D17: order E026 (A030) -> E027 (A034b, alongside) -> E028 (A028 DRL-OR-style) -> E029 (A025 diff reward)
- avenues: A030 active (E026), A034 new active (E026 step 0, E027), A028 -> E028, A025 -> E029
- cards: experiments/E026-system-static-table.md, experiments/E027-e025-hardening.md

## Plan (D17)
- E026: best static table by SYSTEM PDR, simulator-only first-improvement coordinate descent (Pedro: no fluid
  surrogate). Step 0 = fix/record 'table' rule failed-path semantics. Tests KILL-4 (SP-86) and K11 (Deltacom > 81.9?).
- E027: K17 with 2nd export + 2nd eval seed (CPU, while E026 runs).

## Active runs
none.

## Open problem
Does Deltacom have static headroom that learners miss (E026 > 81.9)? If yes, A025 is retargeted and C4 reworded.

## EXACT next step
research-do deploy E026: push, git pull on server; time Deltacom with --sweeps 0 (set M/eps as in card); launch SP-86
(M=16, eps 0.5, starts sp/E025-table/random:1/random:2, 8 workers) and Deltacom (LOAD 8) detached, CPU only.
Then E027 alongside on CPU. Paper deadline ~2027-01-14. ssh: pedroamaral@10.26.110.14.

## Read next
experiments/E026-system-static-table.md ('Implemented' block), infra.md, runs.md.
