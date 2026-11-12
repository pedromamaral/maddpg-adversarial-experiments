#!/bin/bash
# E040 stage 1 (A043, D29): rules-only screen of 4 Topology Zoo graphs for the credit test.
# Per (graph, load): tools/own_flow_effect.py (ECMP + greedy PDR, own-flow gain vs system gain), 20 paired eps.
# Calibration rows with the same code: SP-86 3x and Deltacom 8x. Selection: tools/e040_screen.py (ECMP-only load choice).
# CPU only, one chain per graph (+1 calibration chain). Run from the repo root on the server.
cd ~/maddpg-adversarial-experiments; R=$(pwd); RES=host_data/results; O=$RES/E040; mkdir -p $O ~/idqn
docker run --rm -v $R/$RES:/r maddpg-exp:latest sh -c "mkdir -p /r/E040 && chown -R $(id -u):$(id -g) /r/E040" >/dev/null 2>&1
dr() { n=$1; shift; docker run --rm --name maddpg-E040-$n -e TQDM_DISABLE=1 --cpus 1.5 -v $R/host_data:/workspace/data -v $R/src:/workspace/src \
  -v $R/configs:/workspace/configs -v $R/tools:/workspace/tools -w /workspace --entrypoint python3 maddpg-exp:latest \
  tools/own_flow_effect.py --src src --episodes 20 --out data/results/E040/$n.json "$@" > ~/idqn/maddpg-E040-$n.log 2>&1; }
LOADS=${LOADS:-"3 5 8 12 16 24"}
chain() { g=$1; for L in $LOADS; do dr ${g}_L$L --config configs/v2_zoo_$g.json --load $L; done
  echo "$(date -u +%m-%d\ %H:%M) E040 $g done" >> ~/idqn/run_E040.done; }
calib() { dr sp86_L3 --config configs/v2_pilot4_lc.json --load 3; dr deltacom_L8 --config configs/v2_deltacom.json --load 8
  echo "$(date -u +%m-%d\ %H:%M) E040 calib done" >> ~/idqn/run_E040.done; }
for g in interoute gtsce tatanld geant2012; do chain $g & done
calib &
wait
docker run --rm -v $R/$O:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
echo "$(date -u +%m-%d\ %H:%M) E040 all done" >> ~/idqn/run_E040.done
