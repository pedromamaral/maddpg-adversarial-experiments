#!/bin/bash
# E039 (A049, D27): E030 on the second learner. Does own-flow credit undo the system-optimal table for MADDPG LC too?
# Flow-credit MADDPG, local critic, n=0 training, fixed hotspot, 750 eps incl. 50 warm, flat eps 0.05, 3 seeds.
#  dcW_lc  Deltacom warm start (E026_dc table) · spW_lc  SP-86 warm start (control). Cold ref: E021 dc_ac_local_fix (same protocol).
# CPU only. Run from the repo root on the server. One chain per (arm, seed), 6 parallel.
cd ~/maddpg-adversarial-experiments; R=$(pwd); RES=host_data/results; O=$RES/E039; mkdir -p $O ~/idqn
docker run --rm -v $R/$O:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
dr() { n=$1; shift; docker run --rm --name maddpg-E039-$n -e TQDM_DISABLE=1 --cpus 2 -v $R/host_data:/workspace/data -v $R/src:/workspace/src \
  -v $R/configs:/workspace/configs -w /workspace --entrypoint python3 maddpg-exp:latest src/idqn_control.py --learner ac --critic local \
  --out data/results/E039/$n --eval-failures 0,4,8 --track-every 100 --eps-start 0.05 "$@" > ~/idqn/maddpg-E039-$n.log 2>&1; }
arm() { n=$1; s=$2; shift 2; dr ${n}_s$s "$@" --seed $s; echo "$(date -u +%m-%d\ %H:%M) E039 $n s$s done" >> ~/idqn/run_E039.done; }
for s in 0 1 2; do
  arm dcW_lc $s --config configs/v2_deltacom.json --warm-table data/results/E026_dc/table_best.json --sp-pretrain 50 &
  arm spW_lc $s --config configs/v2_pilot4_lc.json --warm-table data/results/E026_sp/table_best.json --sp-pretrain 50 &
done
wait
docker run --rm -v $R/$O:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
echo "$(date -u +%m-%d\ %H:%M) E039 all done" >> ~/idqn/run_E039.done
