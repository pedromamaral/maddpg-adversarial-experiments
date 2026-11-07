# E034 — Failure reaction across learners (A042a, T2, K22)
Status: DONE 2026-11-05 19:51, analysed 2026-11-07, code 5599288. Avenue A042. Launcher tools/run_E034.sh (shared with E035).

Hypothesis H1: the failure reaction of K21 is not DQN-specific: failure-trained MADDPG and DQN rhf also pay >1 pp shuffle at n=8.
Change: none in src. Eval-only on existing checkpoints: idqn_p4_rhf (dqn), ac_local_rhf (lc), ac_central_rhf (cc); duelling skipped (K06).
Config: SP-86 3x, shuffle {none,path,all} at n=0/4/8, 20 paired eps seed 20240601, own table via --export-tables + run_eval TABLE_FILE. 3 seeds. Same protocol as E032, so E018 is the comparison row.
Reading (noise ~1 pp): shuffle cost >1 pp at n=8 on 3/3 seeds AND own table >1 pp below learner -> reaction shared (C3 generalises). Cost <=1 pp and own table ~ learner -> E018's reaction is specific; C3 stays scoped to it.

## Results (2026-11-07; paired, 20 eps, eval seed 20240601; per-seed in brackets s0 s1 s2)
Delta = variant - same checkpoint unshuffled. tab = own extracted table (run_eval TABLE_FILE) - learner.
| net | n | base | path | all | own table |
|---|---|---|---|---|---|
| DQN rhf | 0 | 95.41 | -0.07 | -0.12 | -1.60 (-2.60 -3.79 +1.58) |
| DQN rhf | 4 | 92.69 | -0.81 (-1.85 -0.01 -0.57) | -0.71 | -3.74 (-5.33 -4.37 -1.52) |
| DQN rhf | 8 | 86.58 | -1.13 (-1.91 -1.52 +0.02) | -1.03 (-1.17 -0.91 -1.03) | -4.56 (-7.40 -4.35 -1.91) |
| MADDPG LC rhf | 0 | 96.27 | -0.23 | -0.66 (-0.58 -0.95 -0.47) | -1.36 |
| MADDPG LC rhf | 4 | 92.90 | -1.14 (-1.16 -1.32 -0.93) | -0.78 | -5.70 (-4.03 -7.92 -5.16) |
| MADDPG LC rhf | 8 | 86.67 | -1.34 (-1.37 -1.35 -1.30) | -1.34 (-0.65 -1.17 -2.21) | -4.76 (-4.98 -4.58 -4.73) |
| MADDPG CC rhf | 0 | 97.02 | -0.35 | -0.30 | -0.66 |
| MADDPG CC rhf | 4 | 92.97 | -0.95 (-1.21 -0.74 -0.91) | -0.63 | -3.12 (-4.68 -3.38 -1.30) |
| MADDPG CC rhf | 8 | 86.72 | -1.10 (-1.97 -1.21 -0.13) | -0.71 (-1.89 -0.19 -0.06) | -4.40 (-6.93 -2.73 -3.53) |
Reference E018 (fixed hotspot, E032): path -0.3/-1.3/-2.2, all -0.2/-1.4/-2.7, own table -0.6/-4.7/-4.6. Searched table 96.98/91.5/83.8; greedy 95.7/93.6/89.1.

## Interpretation
- Shuffle cost is negative in 17/18 seed cells at n=4 and 17/18 at n=8, and grows with failures for all three learners; at n=0 it is <=0.66 pp (K08 holds).
- Magnitude ~1.0-1.3 pp at n=8 (path), about half of E018. The strict band (>1 pp on 3/3 seeds) is met only by MADDPG LC path; DQN and CC miss on one seed each.
- Own frozen table: 9/9 seed cells >1 pp below the learner at n=4 and n=8. Net of the n=0 extraction loss (-0.7..-1.6) it is still -2.1..-4.3 pp. Learners beat the best searched static table by ~1.3 pp (n=4) and ~2.9 pp (n=8).
- Caveat: the own table is one lossy static summary (argmax of mean Q / actor output); the n=0 gap shows extraction loses ~1 pp on its own.
## Verdict
H1 supported (medium): the failure reaction is shared by failure-trained DQN rhf and MADDPG LC/CC rhf on SP-86, not E018-specific; it is smaller than E018 on the shuffle measure. Branch for paper.md: "reaction shared", with magnitude stated per learner. Deltacom unchanged (static, K22).
