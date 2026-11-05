# E037 — Step credit on Deltacom, 2 more seeds (A045, D24)
Status: SKELETON 2026-11-05, not implemented. Avenue A045. Follows E020 (1 seed, 80.9), K10.

Hypothesis H1: with 3 seeds, step MADDPG-LC on Deltacom 8x stays within noise of flow-credit LC, so credit has no detectable PDR effect there.
Change: none in src expected. Same recipe as E020, new seeds only.
Config: v2_dc_lc recipe. Check at implement: E020's exact command, seed and hotspot regime, to pick the matching comparator (E021 fix or E022 rhf) and report it in the card.
 Eval n=0/4/8, 20 paired eps, seed 20240601, least-loaded share as in E020.
Reading (noise: 3-5 pp Deltacom rhf; fixed-hotspot spread to be read from E021): judged on 3-seed mean and seed range.
 - step within noise of flow LC -> no credit effect detected at this power; C1 stays scoped to SP-86.
 - step clearly below flow LC (gap > noise, seed ranges separate) -> credit helps on Deltacom too; revisit D24 wording.
Limit: 3 seeds with this noise cannot show equality; say "not detected", not "no effect".
Cost: CPU, ~6 h per seed (estimate from E028 scale; confirm at implement). Runs alongside E030 if cores allow.
