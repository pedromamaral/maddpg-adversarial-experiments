#!/bin/bash
# E033: DRL-OR-style fallback ablation (A028, D22). Does the 0.05 safe-fallback margin explain the SP-86 ~2 pp gap to plain DQN (K19)?
#  A (eval-only): E028 {sp,dc} checkpoints evaluated with no fallback (paired with E032 base, same eval seed 20240601).
#  B (train): E028 recipe without --fallback-margin (keep --sp-pretrain 50, rhf, train failures 0-8), SP-86 3x, 3 seeds.
# CPU only. Run from the repo root on the server. One chain per seed (3 parallel): A sp -> A dc -> B sp.
cd ~/maddpg-adversarial-experiments; R=$(pwd); RES=host_data/results; O=$RES/E033; mkdir -p $O ~/idqn
docker run --rm -v $R/$O:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
dr() { n=$1; shift; docker run --rm --name $n -e TQDM_DISABLE=1 --cpus 4 -v $R/host_data:/workspace/data -v $R/src:/workspace/src \
  -v $R/configs:/workspace/configs -w /workspace --entrypoint python3 maddpg-exp:latest src/idqn_control.py --learner dqn "$@" > ~/idqn/$n.log 2>&1; }
ev() { # name seed cfg: E028 checkpoint, no fallback margin
  name=$1; s=$2; cfg=$3; d=$O/nofb_${name}_s$s; mkdir -p $d; cp $RES/E028/${name}_s$s/qnet_seed$s.pt $d/
  dr maddpg-E033-nofb-$name-s$s --eval-only --seed $s --config configs/$cfg.json --out data/results/E033/nofb_${name}_s$s \
    --eval-failures 0,4,8 --eval-tag nofb; }
chain() { s=$1
  ev sp $s v2_pilot4_lc; ev dc $s v2_deltacom
  dr maddpg-E033-train-sp-s$s --config configs/v2_pilot4_lc.json --seed $s --out data/results/E033/train_sp_s$s \
    --train-random-hotspot --train-failures 0,2,4,6,8 --sp-pretrain 50 --eval-failures 0,4,8 --eval-stale 0,2
  echo "$(date -u +%m-%d\ %H:%M) E033 s$s done" >> ~/idqn/run_E033.done; }
for s in 0 1 2; do chain $s & done
wait
docker run --rm -v $R/$O:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
