#!/bin/bash
# E040 stage 1b (A043, D29): finer loads for the two graphs whose only in-band load sat at the band edge in stage 1.
# Same rule as stage 1 (tools/e040_screen.py: ECMP in [80,92] closest to 86; GO = greedy-ECMP >= 3 and share >= 1).
# Geant2012 9,10,11x; TataNld 6,7x. Two chains at 1.5 cpus. Run from the repo root on the server.
cd ~/maddpg-adversarial-experiments; R=$(pwd); RES=host_data/results; O=$RES/E040; mkdir -p $O ~/idqn
dr() { n=$1; shift; docker run --rm --name maddpg-E040-$n -e TQDM_DISABLE=1 --cpus 1.5 -v $R/host_data:/workspace/data -v $R/src:/workspace/src \
  -v $R/configs:/workspace/configs -v $R/tools:/workspace/tools -w /workspace --entrypoint python3 maddpg-exp:latest \
  tools/own_flow_effect.py --src src --episodes 20 --out data/results/E040/$n.json "$@" > ~/idqn/maddpg-E040-$n.log 2>&1; }
chain() { g=$1; shift; for L in "$@"; do dr ${g}_L$L --config configs/v2_zoo_$g.json --load $L; done
  echo "$(date -u +%m-%d\ %H:%M) E040b $g done" >> ~/idqn/run_E040b.done; }
chain geant2012 9 10 11 &
chain tatanld 6 7 &
wait
docker run --rm -v $R/$O:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
echo "$(date -u +%m-%d\ %H:%M) E040b all done" >> ~/idqn/run_E040b.done
