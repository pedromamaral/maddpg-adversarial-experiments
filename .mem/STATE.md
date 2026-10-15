# STATE — 2026-10-15

branch main (+ this ideate commit), not pushed; server code at d209c69.
phase just done: research-think ideate (after E025).

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
research-do implement E026 (card = spec; start with step 0, then tools/static_table_search.py; time one episode to set M, eps).
Paper deadline target ~2027-01-14 (TNSM). ssh alias 'server' not resolvable locally; use pedroamaral@10.26.110.14.

## Read next
experiments/E026-system-static-table.md, codebase.md, E025 card ('table' rule + export usage).
