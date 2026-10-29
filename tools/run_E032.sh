#!/bin/bash
# E032: shuffle ablation under failures (A039 E018 SP-86 failure-trained DQN; A028 E028 cells) + static tables
# extracted from the E018 nets (argmax mean Q per failure count) evaluated via run_eval.sh. CPU, eval-only.
# Run from the repo root on the server. One chain per seed (3 parallel): p4f -> E028 sp -> E028 dc.
cd ~/maddpg-adversarial-experiments; R=$(pwd); RES=host_data/results; O=$RES/E032; mkdir -p $O ~/idqn
NS="0,4,8"
ev() { # name seed cfg src_ckpt_dir extra-args...   -> shuffle none (+tables) / path / all
  name=$1; s=$2; cfg=$3; src=$4; shift 4; d=$O/${name}_s$s; mkdir -p $d
  cp $RES/$src/qnet_seed$s.pt $d/
  for sh in base path all; do
    extra=""; [ $sh = base ] && extra="--export-tables"; [ $sh != base ] && extra="--eval-shuffle $sh"
    docker run --rm --name maddpg-E032-$name-s$s-$sh -e TQDM_DISABLE=1 --cpus 4 -v $R/host_data:/workspace/data -v $R/src:/workspace/src \
      -v $R/configs:/workspace/configs -w /workspace --entrypoint python3 maddpg-exp:latest src/idqn_control.py --eval-only --seed $s \
      --config configs/$cfg.json --out data/results/E032/${name}_s$s --eval-failures $NS --eval-tag $sh $extra "$@" \
      > ~/idqn/maddpg-E032-$name-s$s-$sh.log 2>&1
  done
}
tab() { # seed: evaluate the extracted E018 tables (paired with the in-script DQN eval, same seed 20240601)
  s=$1
  for n in 0 4 8; do
    TABLE_FILE=$R/$O/p4f_s$s/table_idqn_seed${s}_fail$n.json DOCKER_EXTRA="--cpus 2" LOAD=3 EVAL_NAME=E032_tab_p4f_s${s}_n$n NO_ATTR=1 \
      RULES="table sp" TQDM_DISABLE=1 bash tools/run_eval.sh v2_p4_lc configs/v2_pilot4_lc.json LC-Simple "$n" > ~/idqn/maddpg-E032-tab-s$s-n$n.log 2>&1
  done
}
chain() { s=$1
  ev p4f $s v2_pilot4_lc idqn_p4_fail; tab $s
  ev sp $s v2_pilot4_lc E028/sp_s$s --fallback-margin 0.05
  ev dc $s v2_deltacom E028/dc_s$s --fallback-margin 0.05
  echo "$(date -u +%m-%d\ %H:%M) E032 s$s done" >> ~/idqn/run_E032.done; }
for s in 0 1 2; do chain $s & done
wait
docker run --rm -v $R/$O:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
