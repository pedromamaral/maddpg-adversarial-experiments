#!/bin/bash
# E038 (A048, D27): in-range telemetry lie. E035's lure / repel and a uniform control, clamped per (agent, slot) to the
# [p5,p95] range of the unattacked rollout (src/idqn_control.py --lie-range). Eval-only, CPU, SP-86 3x, n=0/4/8, 20 paired eps.
# Victims = E035's: p4f (E018 DQN), lc, cc (rhf); checkpoints copied from host_data/results/E035/<net>_s<seed>.
# One chain per seed (3 parallel). Run from the repo root on the server.
cd ~/maddpg-adversarial-experiments; R=$(pwd); RES=host_data/results; O=$RES/E038; mkdir -p $O ~/idqn
docker run --rm -v $R/$RES:/r maddpg-exp:latest sh -c "chown -R $(id -u):$(id -g) /r/E038" >/dev/null 2>&1
CFG=configs/v2_pilot4_lc.json
args() { case $1 in p4f) echo "--learner dqn";; lc) echo "--learner ac --critic local";; cc) echo "--learner ac --critic central";; esac; }
ck()  { case $1 in p4f) echo qnet_seed$2.pt;; lc) echo ac_local_seed$2.pt;; cc) echo ac_central_seed$2.pt;; esac; }
dr() { n=$1; shift; docker run --rm --name $n -e TQDM_DISABLE=1 --cpus 1.5 -v $R/host_data:/workspace/data -v $R/src:/workspace/src \
  -v $R/configs:/workspace/configs -w /workspace --entrypoint python3 maddpg-exp:latest src/idqn_control.py --eval-only \
  --config $CFG --eval-failures 0,4,8 "$@" > ~/idqn/$n.log 2>&1; }
cell() { net=$1; s=$2; d=$O/${net}_s$s; mkdir -p $d; cp $RES/E035/${net}_s$s/$(ck $net $s) $d/
  o="$(args $net) --seed $s --out data/results/E038/${net}_s$s --lie-range 5,95"
  for lie in lure repel random; do
    dr maddpg-E038-$net-s$s-$lie $o --eval-tag ${lie}_in --eval-lie $lie
  done; }
chain() { s=$1; for net in p4f lc cc; do cell $net $s; done
  echo "$(date -u +%m-%d\ %H:%M) E038 s$s done" >> ~/idqn/run_E038.done; }
for s in 0 1 2; do chain $s & done
wait
docker run --rm -v $R/$RES:/r maddpg-exp:latest sh -c "chown -R $(id -u):$(id -g) /r/E038" >/dev/null 2>&1
echo "$(date -u +%m-%d\ %H:%M) E038 all done" >> ~/idqn/run_E038.done
