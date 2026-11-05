# E034 — Failure reaction across learners (A042a, T2, K22)
Status: RUNNING since 2026-11-05 14:58, code 5599288. Avenue A042. Launcher tools/run_E034.sh (shared with E035).

Hypothesis H1: the failure reaction of K21 is not DQN-specific: failure-trained MADDPG and DQN rhf also pay >1 pp shuffle at n=8.
Change: none in src. Eval-only on existing checkpoints: idqn_p4_rhf (dqn), ac_local_rhf (lc), ac_central_rhf (cc); duelling skipped (K06).
Config: SP-86 3x, shuffle {none,path,all} at n=0/4/8, 20 paired eps seed 20240601, own table via --export-tables + run_eval TABLE_FILE. 3 seeds. Same protocol as E032, so E018 is the comparison row.
Reading (noise ~1 pp): shuffle cost >1 pp at n=8 on 3/3 seeds AND own table >1 pp below learner -> reaction shared (C3 generalises). Cost <=1 pp and own table ~ learner -> E018's reaction is specific; C3 stays scoped to it.
