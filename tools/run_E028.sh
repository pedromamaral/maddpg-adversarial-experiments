#!/bin/bash
# E028: DRL-OR-style DQN (flow credit, --sp-pretrain 50, --fallback-margin 0.05) on Deltacom 8x + SP-86 3x, 3 seeds. CPU only (idqn_control has no GPU).
# 3 seed chains in parallel (dc then sp), 2 cpus each. Run from the repo root on the server. Eval: failures 0,4,8, stale 0,2; compare vs E026b rules.
cd ~/maddpg-adversarial-experiments; R=$(pwd); mkdir -p host_data/results/E028
docker run --rm -v $R/host_data/results/E028:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
X="--train-random-hotspot --train-failures 0,2,4,6,8 --sp-pretrain 50 --fallback-margin 0.05 --eval-failures 0,4,8 --eval-stale 0,2"
dr() { n=$1; shift; docker run --rm --name $n -e TQDM_DISABLE=1 --cpus 2 -v $R/host_data:/workspace/data -v $R/src:/workspace/src -v $R/configs:/workspace/configs -w /workspace --entrypoint python3 maddpg-exp:latest src/idqn_control.py --learner dqn "$@" > ~/idqn/$n.log 2>&1; }
for s in 0 1 2; do (
  dr maddpg-E028-dc-s$s --config configs/v2_deltacom.json --seed $s --out data/results/E028/dc_s$s --batch 256 --updates 20 $X
  echo "$(date -u +%m-%d\ %H:%M) E028 dc s$s done" >> ~/idqn/run_E028.done
  dr maddpg-E028-sp-s$s --config configs/v2_pilot4_lc.json --seed $s --out data/results/E028/sp_s$s $X
  echo "$(date -u +%m-%d\ %H:%M) E028 sp s$s done" >> ~/idqn/run_E028.done ) &
done
wait
