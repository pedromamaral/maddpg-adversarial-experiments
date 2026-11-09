#!/bin/bash
# E037 (A045, D24): E020 recipe (step-credit MADDPG-LC, Deltacom 8x, fixed hotspot) with 2 more training seeds.
# E020 = configs/v2_deltacom.json, training.seed default 42 -> run v2_dc_lc. New seeds 1, 2 via configs/v2_deltacom_s{1,2}.json
# (only training.seed differs). Out: host_data/results/v2_dc_lc_s{1,2}. Train on GPU 0, then eval n=0/4/8 with rules
# (random+flow = ECMP, greedy, policy), 20 paired eps seed 20240601, 8x. Run from the repo root on the server.
cd ~/maddpg-adversarial-experiments; R=$(pwd); mkdir -p ~/idqn
chain() { s=$1; run=v2_dc_lc_s$s; cfg=configs/v2_deltacom_s$s.json; mkdir -p host_data/results/$run
  docker run --rm --name maddpg-E037-train-s$s --gpus all -e TQDM_DISABLE=1 -v $R/host_data:/workspace/data -v $R/host_logs:/workspace/logs \
    -v $R/src:/workspace/src -v $R/configs:/workspace/configs maddpg-exp:latest python src/standalone_experiment_runner.py \
    --config $cfg --gpu 0 --phase train --results-dir data/results/$run --variants LC-Simple > ~/idqn/maddpg-E037-train-s$s.log 2>&1
  LOAD=8 EVAL_NAME=E037_eval DOCKER_EXTRA="--cpus 2" NO_ATTR=1 RULES="policy random+flow greedy" TQDM_DISABLE=1 \
    bash tools/run_eval.sh $run $cfg LC-Simple "0 4 8" > ~/idqn/maddpg-E037-eval-s$s.log 2>&1
  echo "$(date -u +%m-%d\ %H:%M) E037 s$s done" >> ~/idqn/run_E037.done; }
for s in 1 2; do chain $s & done
wait
