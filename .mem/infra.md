# Infrastructure (2026-10-14)

## Hosts
- server: ssh pedroamaral@10.26.110.14 — NOVA VPN required. Shared (another user; netslice-drl
  project + images: DO NOT TOUCH). 16 cores, 1x RTX 2080 Ti, ~113 GB free. Auth: ssh key (BatchMode ok).
- local Mac: analysis, figures (.venv-plots/bin/python), paper. host_data/ is gitignored.

## Code deploy
Server repo ~/maddpg-adversarial-experiments is a CLEAN git checkout. Deploy = commit locally,
git push, then `ssh server 'cd ~/maddpg-adversarial-experiments && git pull'`. Never scp code.

## Docker
Image maddpg-exp:latest (36 GB; baked src stale -> ALWAYS mount src at /workspace/src).
Mounts: host_data->/workspace/data, src->/workspace/src, configs->/workspace/configs, host_logs->/workspace/logs.
Learner (idqn_control.py, DQN or --learner ac), per run:
  docker run --rm --name <n> -v $R/host_data:/workspace/data -v $R/src:/workspace/src \
    -v $R/configs:/workspace/configs -w /workspace --entrypoint python3 maddpg-exp:latest \
    src/idqn_control.py --learner ac --critic local --seed 0 --config configs/<c>.json \
    --out data/results/<dir> [--train-random-hotspot --train-failures 0,2,4,6,8] > ~/idqn/<n>.log 2>&1
  eval-only variants: --eval-only --eval-tag shiftA --config configs/<c>_shiftA.json; --eval-shuffle all
Step-reward MADDPG (old pipeline): ... --gpus all maddpg-exp:latest python src/standalone_experiment_runner.py
  --config <c> --gpu 0 --phase train --results-dir data/results/<dir> --variants LC-Simple
Rule baselines / policy eval: tools/run_eval.sh RUN CONFIG VARIANT ["0 4"] with env LOAD, EVAL_NAME,
  RULES ("policy random random+flow greedy greedy_stale2 sp worst"), NO_ATTR=1 (run from repo root on server).
Campaign scripts: ~/idqn/run_*.sh (subshells + `&` + wait); completion markers ~/idqn/*.done; logs ~/idqn/*.log.

## Launch / monitor / fetch
- Launch long jobs: `ssh server 'cd ~/maddpg-adversarial-experiments && setsid nohup bash ~/idqn/run_X.sh < /dev/null > ~/idqn/run_X.out 2>&1 &'`
- Monitor: tools/runs_status.sh (reads active rows of runs.md); or `ssh server 'cat ~/idqn/*.done | tail'`.
- Fetch JSON only: `rsync -a --include='*/' --include='*.json' --include='*.log' --exclude='*' server:~/maddpg-adversarial-experiments/host_data/results/<dir>/ host_data/results/<dir>/`

## Gotchas
- Container output is root-owned: `docker run --rm -v <dir>:/r maddpg-exp:latest chown -R 1003:<gid> /r`.
- Never `pkill -f` a pattern contained in the ssh command line itself.
- Never delete exited containers or other projects' images (D12).
- run_eval.sh defaults to 2x load — set LOAD=3 (SP-86) / LOAD=8 (Deltacom).
- All evals paired: traffic seed 20240601 in StandaloneExperimentRunner._attack_episodes.
- Results: host_data/results (active), host_data/archive_v1 (v1 attacks, theses; never read by default).
- own_flow/ results exist only on the server (not yet rsynced).
- Literature: publisher sites through Claude in Chrome (VPN); terminal is bot-blocked.
