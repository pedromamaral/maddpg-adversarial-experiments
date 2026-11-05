#!/bin/bash
# E035: after the run_E034 chains end, rerun MADDPG margin cells that have no n=8 result
# (they crashed at n=4 before the fix to Policy.grad_score). Run from the repo root on the server.
cd ~/maddpg-adversarial-experiments; R=$(pwd); O=host_data/results/E035
until [ "$(grep -c 'E035 s[012] done' ~/idqn/run_E034.done 2>/dev/null)" = 3 ]; do sleep 120; done
one() { net=$1; s=$2; critic=$3; f=$O/${net}_s$s/ac_${critic}_seed${s}_margin03.json
  python3 -c "import json,sys; sys.exit(0 if 'fail8' in json.load(open('$f'))['eval'] else 1)" 2>/dev/null && return
  n=maddpg-E035-$net-s$s-margin-rerun
  docker run --rm --name $n -e TQDM_DISABLE=1 --cpus 2 -v $R/host_data:/workspace/data -v $R/src:/workspace/src \
    -v $R/configs:/workspace/configs -w /workspace --entrypoint python3 maddpg-exp:latest src/idqn_control.py --eval-only \
    --config configs/v2_pilot4_lc.json --eval-failures 0,4,8 --learner ac --critic $critic --seed $s \
    --out data/results/E035/${net}_s$s --eval-tag margin03 --eval-lie margin --lie-eps 0.3 > ~/idqn/$n.log 2>&1
  echo "$(date -u +%m-%d\ %H:%M) E035 rerun $net s$s done" >> ~/idqn/run_E034.done; }
for s in 0 1 2; do (one lc $s local; one cc $s central) & done
wait
docker run --rm -v $R/$O:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
echo "$(date -u +%m-%d\ %H:%M) E035 reruns finished" >> ~/idqn/run_E034.done
