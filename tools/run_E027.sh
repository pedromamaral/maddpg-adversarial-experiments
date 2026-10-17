#!/bin/bash
# E027: harden K17 on Deltacom 8x. 2nd table export (train seed 3000) x 2nd eval traffic seed (20240602).
# Cells: B@20240601, A@20240602, B@20240602 (A@20240601 = E025). Sequential, 4 cpus. Run from the repo root on the server.
cd ~/maddpg-adversarial-experiments; R=$(pwd); O=host_data/results/E027; mkdir -p $O
C=v2_deltacom; RUN=v2_dc_lc; LOAD=8
docker run --rm --name maddpg-E027-exp -e TQDM_DISABLE=1 --cpus 4 -v $R/host_data:/workspace/data -v $R/src:/workspace/src -v $R/configs:/workspace/configs -v $R/tools:/workspace/tools -w /workspace --entrypoint python3 maddpg-exp:latest tools/supervised_signal.py --src /workspace/src --config configs/$C.json --train-episodes 150 --train-seed 3000 --out data/results/E027/ss_$C.json --export-table data/results/E027/table_B.json > ~/idqn/maddpg-E027-exp.log 2>&1
docker run --rm -v $R/host_data/results/E027:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
ev() { t=$1; seed=$2; tag=$3   # t = table file (repo-relative), seed = eval traffic seed
  TABLE_FILE=$R/$t EVAL_TRAFFIC_SEED=$seed DOCKER_EXTRA="--cpus 4" LOAD=$LOAD EVAL_NAME=E027_$tag NO_ATTR=1 RULES="table random+flow" TQDM_DISABLE=1 \
    bash tools/run_eval.sh $RUN configs/$C.json LC-Simple "0 4 8" > ~/idqn/maddpg-E027-eval-$tag.log 2>&1
  echo "$(date -u +%m-%d\ %H:%M) E027 $tag done" >> ~/idqn/run_E027.done; }
ev $O/table_B.json 20240601 B_s601
ev host_data/results/E025/table_v2_deltacom.json 20240602 A_s602
ev $O/table_B.json 20240602 B_s602
