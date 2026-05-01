# MADDPG Adversarial Robustness Framework

This repository accompanies the paper(s) on adversarial robustness of multi-agent deep reinforcement learning for network routing. It trains six hop-by-hop MADDPG routing variants on a real service-provider topology, evaluates them under clean operating conditions (Phase 2), and then stress-tests them with FGSM adversarial attacks (Phase 3).

All results, figures, and ranking artefacts used in the papers are produced by the three-phase pipeline below and can be fully reproduced from source using Docker.

## Architecture Variants

Six MADDPG variants are trained and evaluated, crossing three orthogonal design axes — critic scope, Q-network architecture, and use of a GNN encoder:

| Variant | Critic scope | Q-network | GNN encoder |
|---|---|---|---|
| CC-Simple | Centralised | Simple DQN | No |
| CC-Duelling | Centralised | Duelling DQN | No |
| CC-Simple-GNN | Centralised | Simple DQN | Yes |
| CC-Duelling-GNN | Centralised | Duelling DQN | Yes |
| LC-Duelling | Local | Duelling DQN | No |
| LC-Duelling-GNN | Local | Duelling DQN | Yes |

**Topology**: 86-node real service-provider network with 32 trainable routing agents and 63 possible next-hop actions per agent.

## Pipeline Overview

| Phase | Purpose | Main outputs | Typical runtime (GPU) |
|---|---|---|---|
| Phase 1 | Train all six variants | `phase1_training_results.json`, model checkpoints | 8–12 h |
| Phase 2 | Clean evaluation & ranking | `phase2_maddpg_results.json`, `phase2_rankings.json` | 1–2 h |
| Phase 3 | FGSM adversarial evaluation | `phase3_fgsm_results.json`, `phase3_rankings.json` | 4–6 h |

FGSM attacks evaluated: **packet-loss maximisation**, **reward minimisation**, and **policy confusion**, each swept over ε ∈ {0.05, 0.1, 0.15, 0.2}.

Plots are generated automatically at the end of each phase and saved to `host_data/results/main_run/figures/`.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support
- A CUDA-capable GPU (tested on RTX 2080 Ti and above)
- Git

Optional, for regenerating plots locally outside Docker:

- Python 3 with `numpy` and `matplotlib`

## Reproducing the Paper Results

### 1. Clone and build

```bash
git clone https://github.com/pedromamaral/maddpg-adversarial-experiments.git
cd maddpg-adversarial-experiments
docker build -t maddpg-exp:latest .
```

The image packages PyTorch 2.1.0, CUDA 11.8, and PyTorch Geometric. The first build takes 15–30 minutes depending on your connection.

### 2. Smoke test (optional but recommended)

Runs a shortened version of all three phases to verify your environment before the multi-hour main run:

```bash
./test_quick.sh
./check_progress_quick.sh   # stream logs; Ctrl-C to detach
```

Results land in `host_data/results/quick_test/`. A passing smoke test confirms GPU access, model initialisation, and plotting are all working.

### 3. Phase 1 — Train all variants

```bash
./run_training.sh
./check_progress_training.sh
```

Trains all six variants for up to 200 epochs with early stopping. Best checkpoints are saved throughout training.

Outputs in `host_data/results/main_run/`:

```
phase1_training_results.json
models/<variant-name>/          ← model checkpoints
figures/phase1_*.{png,pdf}
```

> Model weights are not stored in this repository (they total ~2 GB). They are produced by Phase 1 and consumed in-place by Phases 2 and 3. To move them between machines see the `save_weights.sh` / `load_weights.sh` scripts below.

### 4. Phase 2 — Clean evaluation

```bash
./run_maddpg_eval.sh
./check_progress_maddpg.sh
```

Evaluates all trained variants against OSPF across normal-traffic and dual-link-failure scenarios (30 episodes each). Produces composite rankings across multiple KPI scoring profiles.

Outputs:

```
phase2_maddpg_results.json
phase2_rankings.json
figures/phase2_*.{png,pdf}
```

### 5. Phase 3 — Adversarial evaluation

```bash
./run_fgsm_eval.sh
./check_progress_fgsm.sh
```

Evaluates all variants under FGSM perturbations: three attack objectives × four epsilon values, SLO-based critical epsilon detection, and tier-level attack-surface sensitivity (core / distribution / access).

Outputs:

```
phase3_fgsm_results.json
phase3_rankings.json
figures/phase3_*.{png,pdf}
```

### Running all phases sequentially

```bash
./run_training.sh
# wait for Phase 1 to finish, then:
./run_maddpg_eval.sh
# wait for Phase 2 to finish, then:
./run_fgsm_eval.sh
```

Each script launches a detached Docker container and returns immediately. Monitor progress with the corresponding `check_progress_*.sh` script, then trigger the next phase once the container exits.

## Output Layout

```
host_data/
  results/
    main_run/
      phase1_training_results.json
      phase2_maddpg_results.json
      phase2_rankings.json
      phase3_fgsm_results.json
      phase3_rankings.json
      experiment_summary_rankings.json
      models/
        <variant-name>/         ← .pth checkpoints, one directory per variant
      figures/
        phase1_*.{png,pdf}
        phase2_*.{png,pdf}
        phase3_*.{png,pdf}
host_logs/                      ← container stdout/stderr
```

## Understanding the JSON Outputs

### `phase1_training_results.json`

Per variant: `rewards` and `pkt_losses` (full training trajectories), `final_reward`, `final_pkt_loss`.

### `phase2_maddpg_results.json`

Per variant × scenario: `mean_reward`, `mean_pkt_loss`, `mean_delivery_rate`, `mean_goodput_per_step`, `mean_delay_p95`, `mean_backlog_end`, `mean_util_p95`.

### `phase2_rankings.json`

Composite rankings by scenario and scoring profile. Includes `seed_convergence` telemetry for adaptive rank-stability tracking.

### `phase3_fgsm_results.json`

Per variant × attack case: `clean`, `attacked`, `metrics`, `slo`, `run_config`, `attack_summary`, `surface` (tier-level sensitivity breakdown).

### `phase3_rankings.json`

Overall robustness ranking and per-attack-type breakdown.

## Plot Generation

Plots are produced automatically at the end of each phase. To regenerate them manually from existing result files:

```bash
# All phases
python tools/plot_results.py --results-dir host_data/results/main_run

# Selected phases only
python tools/plot_results.py --results-dir host_data/results/main_run --phases 1 2
```

This works both inside and outside Docker as long as `numpy` and `matplotlib` are available.

**Phase 1**: reward/packet-loss curves per variant, final reward ranking.  
**Phase 2**: composite score bars by scenario and profile, KPI heatmaps, delivery heatmap, rank-stability curve.  
**Phase 3**: packet-loss vs ε, degradation curves, critical-ε plot, attack-surface sensitivity, overall robustness ranking.

## Pulling Results from a Remote Server

If you ran the experiments on a remote GPU server:

```bash
scp -r user@server:/path/to/maddpg-adversarial-experiments/host_data/results/main_run ./main_run
python tools/plot_results.py --results-dir ./main_run
```

## Configuration Reference

All runtime behaviour is controlled by `experiment_config.json`. Key knobs:

### Training

| Key | Description | Default |
|---|---|---|
| `training.epochs` | Maximum training epochs | 200 |
| `training.episodes_per_epoch` | Episodes per epoch | 5 |
| `training.timesteps_per_episode` | Steps per episode | 128 |
| `training.early_stopping.patience_checks` | Early-stopping patience | 12 |
| `training.best_checkpoint.validation_interval_epochs` | Checkpoint validation frequency | 10 |

### Clean evaluation

| Key | Description | Default |
|---|---|---|
| `paper1_eval.evaluation_episodes` | Episodes per scenario | 30 |
| `paper1_eval.link_failure_scenarios` | Link-failure counts to test | [0, 2] |

### Reward shaping

`reward.delivery_weight`, `reward.max_util_penalty`, `reward.var_util_penalty`, `reward.drop_penalty`, `reward.backlog_penalty`

### FGSM evaluation

| Key | Description |
|---|---|
| `attack_configs` | List of `{attack_type, epsilon}` cases to run |
| `fgsm_slo.*` | SLO thresholds for critical-epsilon detection |
| `runtime_control.phase3_enable_slo_pruning` | Skip epsilon values past SLO breach |
| `runtime_control.phase3_skip_after_critical_epsilon` | Stop a variant once critical ε is found |
| `runtime_control.phase3_consecutive_fail_limit` | Consecutive SLO failures before skipping |

### Adaptive seed expansion (Phase 2 rank stability)

| Key | Description |
|---|---|
| `runtime_control.seed_expansion.enable_adaptive` | Enable adaptive seed expansion |
| `runtime_control.seed_expansion.initial_seeds` | Starting number of evaluation seeds |
| `runtime_control.seed_expansion.max_seeds_for_ranking` | Maximum seeds before forcing a decision |
| `runtime_control.seed_expansion.stability_threshold` | Rank-stability coefficient threshold |

## Scripts Reference

| Script | Purpose |
|---|---|
| `test_quick.sh` | Smoke test — all phases, reduced budget |
| `check_progress_quick.sh` | Stream smoke-test container logs |
| `run_training.sh` | Phase 1 — train all variants |
| `check_progress_training.sh` | Stream Phase 1 logs |
| `run_maddpg_eval.sh` | Phase 2 — clean evaluation |
| `check_progress_maddpg.sh` | Stream Phase 2 logs |
| `run_fgsm_eval.sh` | Phase 3 — FGSM adversarial evaluation |
| `check_progress_fgsm.sh` | Stream Phase 3 logs |
| `reset_results.sh` | Wipe results only; preserve model weights |
| `reset.sh` | Full reset — wipe results and model weights |
| `save_weights.sh` | Rsync model weights from this machine to a remote server |
| `load_weights.sh` | Rsync model weights from a remote server to this machine |

### Transferring weights between machines

Model weights are not tracked in Git. Use the helper scripts to move them between servers:

```bash
# Push weights from local → remote
bash save_weights.sh user@remote-server

# Pull weights from remote → local
bash load_weights.sh user@remote-server
```

Both scripts transfer `host_data/results/main_run/models/` and `phase1_training_results.json`, which is all Phases 2 and 3 need.

## Troubleshooting

### No plots were generated

- Check the phase logs first: `./check_progress_fgsm.sh`
- If you see "Plotting utilities unavailable", rebuild the image: `docker build --no-cache -t maddpg-exp:latest .`
- Regenerate manually: `python tools/plot_results.py --results-dir host_data/results/main_run`

### Phase 2 or Phase 3 cannot find trained models

Phase 1 must complete first and all phases must use the same `--results-dir` (default: `data/results/main_run`).

### GPU not detected inside Docker

Verify NVIDIA Container Toolkit is installed:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Rankings look unstable between runs

Increase `paper1_eval.evaluation_episodes` and/or `runtime_control.seed_expansion.max_seeds_for_ranking`, or reduce Phase 3 SLO pruning aggressiveness.
