# E027 — Harden E025/K17: second table export + second eval traffic seed
Status: PLANNED 2026-10-15. Avenue A034b; decision D17. CPU only; run alongside E026 (after E026 step 0 fixes 'table' failure semantics).

Hypothesis: K17's sign holds: own-flow table-for-all < ECMP on Deltacom 8x at n=0/4/8, with a 2nd export
(supervised_signal.py, different train seed) and a 2nd eval traffic seed (≠ 20240601, both rules paired on it).
Change: none expected. Only run E025's pipeline with new seeds (if step 0 changed semantics, SP-86 failure numbers re-run too).
Success criterion: Deltacom table - ECMP < 0 at n=0 for both exports x both eval seeds -> K17 confidence -> high.
  Any table >= ECMP -> K17 back to medium; report the spread.
Results:
Verdict:
