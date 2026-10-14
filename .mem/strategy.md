# Strategy (2026-10-14, from strategy-plan rev 0)

Source: OneDrive-FCTNOVA/Docencia/Research/lit-graph/drl-routing-adversarial/strategy-plan.md (2026-10-14 22:57).
Digest copied verbatim below; full plan has feasibility matrix, open-source unlocks, ranking, avoid-list.

date 2026-10-14 · topic drl-routing-adversarial · status: new (revision 0) · chosen: AVENUE #1 (+#6 folded in) now; AVENUE #3 as gated C5; AVENUE #2 next paper
RQ: in online per-flow MARL routing, does reward granularity decide whether agents learn, what do they learn, and can counterfactual credit make them use current state?
C1 step/team reward -> no learning; per-decision flow credit -> learning, for DQN and MADDPG (K05) [have]
C2 once credit is fixed, architecture is secondary (K06,K07) [have]
C3 learned policy = demand-robust static table (shuffle <1 pp, K08); NEW: compare with a stochastic-optimal static table (AVENUE #6)
C4 learners capture static headroom only (SP-86 yes, Deltacom no) (K10-K12,K16) [have, E025/A029 pending]
C5 (gated) per-flow difference reward recovers part of greedy's state-dependent gain on Deltacom AND makes telemetry shuffle cost > noise (A025)
gaps: GAP#1 no state-use tests in DRL routing (0/15 papers) · GAP#2 no ECMP+adaptive+stale baselines together (0/15) · GAP#3 credit schemes not compared in online per-flow routing (partially filled by CMRL/Teal) · GAP#4 static vs state headroom never separated
novelty (narrowed): first controlled comparison of credit schemes for the same online per-flow routing task, with observation ablations showing what is learned and a two-topology static-vs-state decomposition
venues: IEEE TNSM (primary, no mandatory APC); IEEE/ACM ToN if C5 passes
topologies: SP-86 = primary (C1-C3; static-headroom case of C4; 3x) · Deltacom = contrast (C4 no static headroom; C5 target; 8x) · every new run (A028, #6, C5) reports both
SP-86 release: operator anonymised ("a real 86-node service-provider topology (operator anonymised) with synthetic link capacities"); released with code
baselines (ours, PDR %): ECMP 91.4 / greedy 95.7 / greedy_stale2 93.7 SP-86 3x n=0; ECMP 80.4 / greedy 84.9 Deltacom 8x
literature (not comparable: MLU, fluid): MAGNNETO within 2% of SRLS, 10.1109/tccn.2023.3235719 · CMRL diff-reward MARL TE, 10.1109/tnsm.2024.3454282 · DRL-OR per-flow MAPPO, 10.1109/infocom42981.2021.9488736 · Dealing With Changes MA-DQN vs ECMP (dest-only obs), 10.1109/tnsm.2023.3287936
A028 target: DRL-OR, https://github.com/netlab-lcy/DRL-OR (MIT) -> port reward+observation+pretraining into our simulator; NOT its Mininet testbed on the shared server
other code: MAGNNETO-TE (MIT) · Teal (MIT) · Repetita (no licence) · Topology Zoo (cite)
KILL-1 (A029/E025, wk 1-2): Deltacom system PDR of table-for-all >= greedy - 1 SE -> externality story wrong: drop A025, rethink C4/C5
KILL-2 (A025, end wk 5): diff-reward Deltacom 8x, 3 seeds. PASS = PDR > 80.4 by >= 1.5 pp (>2 SE) AND telemetry shuffle > 1 pp. FAIL -> C5 as open problem, submit at month 3
KILL-3 (A028, wk 4): DRL-OR port not end-to-end -> report "DRL-OR-style" learner (its reward+obs in our learner)
KILL-4 (A030): stochastic-optimal static table beats learners by > 1 pp -> C3 becomes "learners approximate the robust table imperfectly"
do-not-pursue: imported into avenues.md as dropped (A001-A009, A019, A003/A026, Teal/MAGNNETO training now)
constraints: Pedro + Claude only; 1 shared server (16 cores, 1x 2080 Ti); submit Paper 1 ~2027-01-14 (3 mo), slip to 4-5 mo only if KILL-2 passes

## Risk
Reviewers read C3/C4 as a negative result; framing must be "what is learned + why".
Literature numbers are MLU/fluid -> no direct head-to-head; A028 (in-simulator port) is the only fair comparison.

## Gate 2026-10-14 (after lit digest) -> ITERATE
Missing for paper: A029/E025 (KILL-1, running) · A028 DRL-OR-style port (G1, reviewer question) ·
A030 stochastic-optimal static table (C3 strength) · A025 only if KILL-1 passes (C5, optional for TNSM).
