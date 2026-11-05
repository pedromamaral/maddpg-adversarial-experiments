#!/bin/bash
# E034 (A042a, T2): shuffle at n=0/4/8 + own extracted table for failure-trained SP-86 learners that E032 did not cover.
# E035 (A042b, T7): telemetry lies on failure-trained learners, n=0/4/8, against the same paired base (attack gate).
# Eval-only, CPU. Run from the repo root on the server. One chain per seed (3 parallel): E034 -> E035 -> ceiling.
cd ~/maddpg-adversarial-experiments; R=$(pwd); RES=host_data/results; O4=$RES/E034; O5=$RES/E035; mkdir -p $O4 $O5 ~/idqn
docker run --rm -v $R/$RES:/r maddpg-exp:latest sh -c "chown -R $(id -u):$(id -g) /r/E034 /r/E035" >/dev/null 2>&1
NS="0,4,8"; CFG=configs/v2_pilot4_lc.json
# net: name | source dir | checkpoint template | learner flags
args() { case $1 in
  dqn)  echo "--learner dqn";;
  lc)   echo "--learner ac --critic local";;
  cc)   echo "--learner ac --critic central";;
  p4f)  echo "--learner dqn";; esac; }
src() { case $1 in dqn) echo idqn_p4_rhf;; lc) echo ac_local_rhf;; cc) echo ac_central_rhf;; p4f) echo idqn_p4_fail;; esac; }
ck()  { case $1 in dqn|p4f) echo qnet_seed$2.pt;; lc) echo ac_local_seed$2.pt;; cc) echo ac_central_seed$2.pt;; esac; }
pfx() { case $1 in dqn|p4f) echo idqn;; lc) echo ac_local;; cc) echo ac_central;; esac; }
dr() { n=$1; shift; docker run --rm --name $n -e TQDM_DISABLE=1 --cpus 1.5 -v $R/host_data:/workspace/data -v $R/src:/workspace/src \
  -v $R/configs:/workspace/configs -w /workspace --entrypoint python3 maddpg-exp:latest src/idqn_control.py --eval-only \
  --config $CFG --eval-failures $NS "$@" > ~/idqn/$n.log 2>&1; }
e34() { net=$1; s=$2; d=$O4/${net}_s$s; mkdir -p $d; cp $RES/$(src $net)/$(ck $net $s) $d/
  dr maddpg-E034-$net-s$s-base $(args $net) --seed $s --out data/results/E034/${net}_s$s --eval-tag base --export-tables
  for sh in path all; do
    dr maddpg-E034-$net-s$s-$sh $(args $net) --seed $s --out data/results/E034/${net}_s$s --eval-tag $sh --eval-shuffle $sh
  done
  for n in 0 4 8; do
    TABLE_FILE=$R/$d/table_$(pfx $net)_seed${s}_fail$n.json DOCKER_EXTRA="--cpus 1.5" LOAD=3 EVAL_NAME=E034_tab_${net}_s${s}_n$n NO_ATTR=1 \
      RULES="table" TQDM_DISABLE=1 bash tools/run_eval.sh v2_p4_lc $CFG LC-Simple "$n" > ~/idqn/maddpg-E034-tab-$net-s$s-n$n.log 2>&1
  done; }
e35() { net=$1; s=$2; d=$O5/${net}_s$s; mkdir -p $d; cp $RES/$(src $net)/$(ck $net $s) $d/
  o="$(args $net) --seed $s --out data/results/E035/${net}_s$s"
  dr maddpg-E035-$net-s$s-lure   $o --eval-tag lure --eval-lie lure
  dr maddpg-E035-$net-s$s-repel  $o --eval-tag repel --eval-lie repel
  dr maddpg-E035-$net-s$s-margin $o --eval-tag margin03 --eval-lie margin --lie-eps 0.3
  dr maddpg-E035-$net-s$s-rnd03  $o --eval-tag random03 --eval-lie random --lie-eps 0.3
  dr maddpg-E035-$net-s$s-rnd10  $o --eval-tag random10 --eval-lie random --lie-eps 1.0; }
chain() { s=$1
  for net in dqn lc cc; do e34 $net $s; done
  echo "$(date -u +%m-%d\ %H:%M) E034 s$s done" >> ~/idqn/run_E034.done
  for net in p4f lc cc; do e35 $net $s; done
  echo "$(date -u +%m-%d\ %H:%M) E035 s$s done" >> ~/idqn/run_E034.done; }
for s in 0 1 2; do chain $s & done
wait
# Victim-independent damage ceiling, and greedy under a lure: the worst-path rule.
for n in 0 4 8; do
  DOCKER_EXTRA="--cpus 1.5" LOAD=3 EVAL_NAME=E035_worst_n$n NO_ATTR=1 RULES="worst" TQDM_DISABLE=1 \
    bash tools/run_eval.sh v2_p4_lc $CFG LC-Simple "$n" > ~/idqn/maddpg-E035-worst-n$n.log 2>&1
done
echo "$(date -u +%m-%d\ %H:%M) E035 ceiling done" >> ~/idqn/run_E034.done
docker run --rm -v $R/$RES:/r maddpg-exp:latest sh -c "chown -R $(id -u):$(id -g) /r/E034 /r/E035" >/dev/null 2>&1
