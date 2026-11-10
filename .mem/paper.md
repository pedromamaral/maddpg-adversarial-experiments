# Paper plan (research-think paper-plan 2026-11-05; T3 of D23)

Paper 1 v2: paper/paper1_v2/paper1_v2.tex (IEEEtran journal, TNSM, target ~2027-01-14). Bib: Routing_v2.bib.
All numbers: tools/paper1_tables.py -> tables/ figures/ numbers.json (never hand-type numbers).
Status: plan agreed (D24); tex not yet changed (title and text change at T5).

## Framing (D24)
Title: "Credit Assignment, not Architecture: What MARL Routing Agents Learn in Service-Provider Networks" (plural).
One sentence: credit decides whether agents learn at all; what they then learn is a static table plus, if trained on
failures, a failure reaction; and they reach static headroom only where own-flow credit points at it.
Consequence: per-decision credit is necessary, NOT sufficient. On Deltacom flow credit does not beat step credit or ECMP (K10).
The tex says "per-decision credit makes both learners work" unqualified: that sentence must be scoped to SP-86.
Risk (strategy.md): must read as "what is learned + why", not as a negative result.

## Claims -> evidence -> artefact -> status
| claim | artefact | experiments | K | status |
|---|---|---|---|---|
| v1 step-reward MADDPG does not route (C1) | T1_v1 | E001,E002 | K01,K02 | ready |
| baseline realism, operating point | T2_realism, T3, F1 | E008,E009 | K13,K04 | ready |
| pilots fail; credit decides learning on SP-86 (C1) | T4_pilots, T5_credit, F2 | E004-E013 | K05 | ready |
| step vs flow credit on Deltacom (C1 scope) | row in T11 | E020-E022,E037 | K10 | WEAK: step = 1 seed until E037 |
| architecture secondary (C2) | T6_architecture (drop the 4 GNN rows, D25) | E015,E018 | K06 | ready; method text: why no GNN + future work (A046) |
| static table at n=0 (C3) | T7_ablation (extend: add E018) | E012-E017,E032 | K08 | ready, tex needs fix (F1) |
| demand generalisation of learned tables (C3) | T8, F3 | E012-E015 | K09 | ready |
| best static table vs learners, both topologies (C3,C4) | NEW T9_static | E026,E026b,E031,E036 | K11,K20,K25 | ready: table does not transfer to shifts (E036 branch 2); add shift rows to T9 |
| failure-trained learners react to failures (C3) | NEW T10_failure | E018,E032,E034 | K14,K20,K21,K22 | ready: reaction shared (E034 branch 1), magnitude per learner |
| Deltacom: learners ~ECMP, ~7 pp table headroom missed (C4) | NEW T11_deltacom, F4_headroom | E019-E022,E026b | K10,K11 | numbers ready |
| why: own-flow optimum is selfish; coordinated changes (C4) | NEW T12_mechanism | E025,E027,E026b | K17,K18 | K17 ready; K18 low-medium |
| does credit undo the optimum, or is it not found? (C4) | in T12, NEW F5_warmstart (track PDR + agree vs episode, 4 arms) | E030 | K23 | ready: credit undoes it |
| published-method-shaped learner (G1) | NEW T13_drlor | E028,E032,E033 | K19 | numbers ready; E033 per-seed signs unchecked |
| C5 difference reward | future work only | — | K16 | out of paper (D23) |

## Outline (section -> claims -> artefacts; * = new text)
I Introduction: rewrite contributions, drop "1.5 to 5" as a general claim (F2).
II State of the art: keep; *add G4 (static vs state headroom never separated) and DRL-OR positioning.
III MMDP: keep; *state the own-flow vs system objective explicitly (needed for C4).
IV Learning models: keep; *one paragraph on the DRL-OR-style learner ("style", not a port).
V Environment and methodology: *Deltacom conversion, ttl 50, 8x (E019, D09); *static table search; *own-table extraction.
VI Results
 A step reward does not route -> T1 | B why: credit over sparse decisions | C realism -> T2,T3,F1
 D credit decides -> T4,T5,F2 | E architecture secondary -> T6
 F what is learned: a static table in normal operation -> T7,T8,F3 (fix F1)
 G* how good is that table: searched table vs learners, SP-86 -> T9 (K11; decays under failures, K20)
 H* failures: state use where tables stop -> T10 (K14,K21; K22 only as observation)
 I* second topology: credit no longer points at the headroom -> T11,F4,T9 (K10,K11)
 J* why -> T12,F5 (K17,K18,E030)
 K* a published-method-shaped learner -> T13 (K19)
VII Discussion: *rewrite (F3): credit necessary not sufficient; reporting checklist stays; limits below.
VIII Conclusions: *rewrite; future work = difference reward (A025), third topology (A043), feature-wise reaction (A040).
K12 (E023) left out: own_flow/ is server-only, so it breaks the numbers rule; K17+K18 carry the argument. K16: one sentence at most.

## Manuscript sentences now contradicted (fix at T5)
F1 sec:what line ~496: "for every learner ... less than one point ... at four failed links". T7 omits E018, which pays 1.38 (K21).
F2 abstract, contributions, conclusions: "beat ECMP by 1.5 to 5 points" is SP-86-specific (K10).
F3 discussion: "adaptivity has a small and fragile value", "needs no telemetry at decision time", "did not find a regime in which
   a learner adapts", "one topology" and second-topology future work: all stale against K11,K19,K20,K21.
F4 "demand-robust" for the searched table: REFUTED by E036 (K25), say "on the training demand" in C4 text and F4_headroom caption. Keep the word for learned changing-hotspot tables only (K09), with the failure caveat (D20,D22).
F5 any sentence that the learned policy "ignores" or "does not read" state: say static over realised telemetry (shuffle, K08); out-of-range path telemetry still moves it (E035, K24). One sentence in discussion pointing to future attack work.

## Branches for pending results
| result | outcome | what changes |
|---|---|---|
| E030 | gate fails (warm end not near table) | void; fix warm start or drop J's causal claim |
| E030 | dcW <=82, spW ~97 | mechanism firm: credit undoes the optimum; title stands |
| E030 | dcW and spW both drift | forgetting; no conclusion on credit; K17/K18 stated as tentative |
| E030 | dcW >=86 | credit holds it, cannot find it: C4 becomes search + credit; dcD splits exploration vs cold start; abstract softens, title still stands on SP-86 |
| E030 | dcW 82-86 | inconclusive; report as such, A041 stays parked (D23) |
| E034 | >1 pp, 3/3 seeds, own table >1 pp lower, n=8 | reaction shared: C3 general for failure-trained learners |
| E034 | <=1 pp, own table ~ learner | C3 scoped to E018. Read DQN rhf first: E018 is fixed-hotspot, so E018 vs MADDPG rhf confounds algorithm and demand regime |
| E036 | table >= ECMP + 2.3 on both shifts | headroom holds across demand: C4 stated without demand qualifier |
| E036 | table within noise of or below ECMP on a shift | headroom is one-matrix traffic engineering: C4 and F4_headroom say "on the training demand" |
| E037 | step within noise of flow LC | no credit effect detected on Deltacom at this power; C1 stays scoped to SP-86 |
| E037 | step clearly below flow LC | credit helps on Deltacom too: C1 widens, D24 wording revisited |
Bonus: E030 dcC is the first plain flow-credit DQN on Deltacom; add it to T11 (K10 is MADDPG only).

## To do in tools/paper1_tables.py (T4; after E030, E034, E036, E037)
T9_static, T10_failure, T11_deltacom, T12_mechanism, T13_drlor, F4_headroom (ECMP/learner/table/greedy per topology, n=0/4/8),
F5_warmstart (track PDR + table_agree vs episode). Extend T7 with E018. E033 per-seed signs: if not 3/3, do not quote the +1.00.
Length: 13 tables + 5 figures is heavy. Candidates to merge: T2+T3, T1+T4. Check TNSM page limit before T5.

## Also pending for T6
Pedro rewrites in own voice; Routing.bib, author photos, acknowledgement placeholders. SP-86 release wording per strategy.md.

Paper 2 (paper/paper2_robustness.tex, roadmap paper/PAPER2_ROADMAP.md): ON HOLD (D03); gate = A042 (E035 not in Paper 1).
