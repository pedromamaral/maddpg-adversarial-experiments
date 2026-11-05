#!/bin/bash
# E030 (A036, D23): does own-flow credit keep the E026 system table once it is handed to it?
# All arms: flow-credit DQN, n=0 training, fixed hotspot (the conditions the table was optimised for), 750 eps, 3 seeds.
#  dcW  Deltacom warm start, flat eps 0.05 (behaviour stays near the table)
#  spW  SP-86 warm start, flat eps 0.05 (control: credit aligned, K18)
#  dcD  Deltacom warm start, then the usual eps decay 1.0 -> 0.05 (does the exploration phase erase it?)
#  dcC  Deltacom cold start, usual eps decay (reference; plain DQN has no Deltacom run)
# CPU only. Run from the repo root on the server. Two chains per seed (6 parallel).
cd ~/maddpg-adversarial-experiments; R=$(pwd); RES=host_data/results; O=$RES/E030; mkdir -p $O ~/idqn
docker run --rm -v $R/$O:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
dr() { n=$1; shift; docker run --rm --name maddpg-E030-$n -e TQDM_DISABLE=1 --cpus 2 -v $R/host_data:/workspace/data -v $R/src:/workspace/src \
  -v $R/configs:/workspace/configs -w /workspace --entrypoint python3 maddpg-exp:latest src/idqn_control.py --learner dqn \
  --out data/results/E030/$n --eval-failures 0,4,8 --track-every 100 "$@" > ~/idqn/maddpg-E030-$n.log 2>&1; }
DC="--config configs/v2_deltacom.json"; SP="--config configs/v2_pilot4_lc.json"
WDC="--warm-table data/results/E026_dc/table_best.json --sp-pretrain 50"
WSP="--warm-table data/results/E026_sp/table_best.json --sp-pretrain 50"
low() { s=$1
  dr dcW_s$s $DC $WDC --eps-start 0.05 --seed $s
  dr spW_s$s $SP $WSP --eps-start 0.05 --seed $s
  echo "$(date -u +%m-%d\ %H:%M) E030 low s$s done" >> ~/idqn/run_E030.done; }
ref() { s=$1
  dr dcD_s$s $DC $WDC --seed $s
  dr dcC_s$s $DC --seed $s
  echo "$(date -u +%m-%d\ %H:%M) E030 ref s$s done" >> ~/idqn/run_E030.done; }
for s in 0 1 2; do low $s & ref $s & done
wait
docker run --rm -v $R/$O:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
