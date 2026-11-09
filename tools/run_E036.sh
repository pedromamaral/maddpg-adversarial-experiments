#!/bin/bash
# E036 (A044, D24): E026 searched static table on unseen demand (shift A/B), eval-only, CPU.
# Rules table + ECMP (random+flow) + greedy + the LC-Simple learner, all on the same 20 paired eps
# (seed 20240601), n=0/4/8, SP-86 3x and Deltacom 8x. Run from the repo root on the server.
cd ~/maddpg-adversarial-experiments; R=$(pwd)
ev() { c=$1; ld=$2; tag=$3; run=$4; sh=$5
  TABLE_FILE=$R/host_data/results/E026_$tag/table_best.json DOCKER_EXTRA="--cpus 1.5" LOAD=$ld \
    EVAL_NAME=E036_${tag}_shift$sh NO_ATTR=1 RULES="table random+flow greedy policy" TQDM_DISABLE=1 \
    bash tools/run_eval.sh $run configs/${c}_shift$sh.json LC-Simple "0 4 8" > ~/idqn/maddpg-E036-$tag-$sh.log 2>&1
  echo "$(date -u +%m-%d\ %H:%M) E036 $tag shift$sh done" >> ~/idqn/run_E036.done; }
(ev v2_deltacom 8 dc v2_dc_lc A; ev v2_deltacom 8 dc v2_dc_lc B) &
(ev v2_pilot4_lc 3 sp v2_p4_lc A; ev v2_pilot4_lc 3 sp v2_p4_lc B) &
wait
echo "$(date -u +%m-%d\ %H:%M) E036 all done" >> ~/idqn/run_E036.done
