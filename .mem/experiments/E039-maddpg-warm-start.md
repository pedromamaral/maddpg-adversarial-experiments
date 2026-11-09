# E039 — Table warm start with MADDPG LC (A049, D27)
Status: SKELETON 2026-11-09, not implemented. Avenue A049. Replicates E030 (K23) on the second learner.

Hypothesis H1: own-flow credit undoes the system-optimal Deltacom table for MADDPG LC too: a warm-started learner falls from ~86 to ~80 (ECMP), while the SP-86 control keeps ~97.
 H0: MADDPG LC holds the table (>=86) -> K23 is DQN-specific; C1/C4 wording scoped to DQN.
Change: warm start for the actor-critic learner (src/idqn_control.py, AC path): during --sp-pretrain behaviour = --warm-table, and a hinge on the actor scores (margin --warm-margin) makes the table path the argmax per (agent, dest); critic trains on the table behaviour as usual. Reuse --track-every (PDR + table_agree). No change for DQN or for runs without --warm-table.
Arms: dcW_lc (Deltacom 8x warm, flat exploration as E030 dcW), spW_lc (SP-86 3x warm, control). Cold reference: E021 dc_ac_local_fix (LC 80.2); check at implement that its protocol matches (fixed hotspot, n=0 training, same eval); if not, add a dcC_lc arm.
Config: flow credit, critic local, train n=0, fixed hotspot, 750 eps incl. 50 warm, track every 100, final eval n=0/4/8 20 paired eps, 3 seeds. Table files: data/results/E026_{dc,sp}/table_best.json.
Validity gate: at warm end agree ~1.00 and PDR ~86 (Deltacom) / ~97 (SP-86). Otherwise void, fix the hinge.
Reading (spread ~0.6-2 pp): dcW_lc final n=0 <=82 and spW_lc ~97 -> H1, K23 generalises. >=86 -> H0. 82-86 -> inconclusive.
Cost: ~60 lines; 2 arms x 3 seeds x ~6 h CPU; can share the server with E036/E037 tails.
