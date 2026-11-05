# E036 — Searched static table on unseen demand (A044, D24)
Status: SKELETON 2026-11-05, not implemented. Avenue A044. Follows E026b (K11).

Hypothesis H1: the E026 tables were optimised on one hotspot and lose their gain on other demand, as fixed-hotspot learners do (K09).
 H0: the gain over ECMP survives on shifts A and B, so the Deltacom headroom is not one-matrix traffic engineering.
Change: none in src expected. Eval-only: run_eval.sh with TABLE_FILE = E026 table_best (confirm path; same files as E026b).
Config: configs/v2_pilot4_lc_shift{A,B}.json (3x), configs/v2_deltacom_shift{A,B}.json (8x), n=0/4/8, 20 paired eps, seed 20240601.
 Check at implement: ECMP, greedy and learner shift cells must share this traffic seed and n set (existing eval_shiftA/B dirs); rerun rules if not.
Reading (noise <=2.3 pp on shifts, REF), n=0 headline, per topology and shift:
 - table >= ECMP + 2.3 on both shifts -> headroom holds across demand; H0.
 - table within 2.3 of ECMP or below it on a shift -> H1; C4 says "on the training demand".
 - Also report table vs changing-hotspot learners on each shift (is the learned table more demand-robust than the searched one?).
Limit: one searched table per topology, so no seed spread for the table itself.
Cost: CPU only, 2 topologies x 2 shifts x 3 n (ETA at implement).
