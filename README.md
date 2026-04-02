# MADDPG Adversarial Robustness Framework

This repository trains hop-by-hop MADDPG routing variants, compares them under clean operating conditions, and then stress-tests them with FGSM attacks. The workflow is organized into three phases plus post-run result inspection.

Plots are generated automatically at the end of each phase and saved into the active results directory under `figures/`.

## Pipeline Overview

| Phase | Purpose | Main JSON outputs | Main figures | Typical runtime |
|---|---|---|---|---|
| Phase 1 | Train all MADDPG variants | `phase1_training_results.json` | reward curves, packet-loss curves, final reward ranking | 8-12 h on GPU |
| Phase 2 | Clean evaluation and ranking | `phase2_maddpg_results.json`, `phase2_rankings.json` | composite rankings, KPI heatmaps, scenario heatmap, rank-stability curve | 1-2 h |
| Phase 3 | FGSM robustness evaluation | `phase3_fgsm_results.json`, `phase3_rankings.json` | epsilon curves, degradation curves, critical-epsilon plot, attack-surface plot | 4-6 h |
| Summary | Consolidated ranking view | `experiment_summary_rankings.json` | derived from ranking artifacts | included in full run |

## Prerequisites

- Docker
- NVIDIA Container Toolkit if you want GPU acceleration
- Git
- Python 3 with `numpy` and `matplotlib` if you want to replot locally outside Docker

## Quick Start

Build the image:

```bash
docker build -t maddpg-exp:latest .
```

Run the smoke test first:

```bash
./test_quick.sh
./check_progress_quick.sh
```

If the smoke test passes, continue with the standard workflow.

## Standard Workflow

### 1. Train Models

```bash
./run_training.sh
./check_progress_training.sh
```

Outputs:

- `host_data/results/main_run/phase1_training_results.json`
- `host_data/models/<variant-name>/`
- `host_data/results/main_run/figures/phase1_*.png`
- `host_data/results/main_run/figures/phase1_*.pdf`

### 2. Run Clean Evaluation

```bash
./run_maddpg_eval.sh
./check_progress_maddpg.sh
```

This evaluates all trained variants plus OSPF under:

- normal traffic
- dual-link failure

Outputs:

- `host_data/results/main_run/phase2_maddpg_results.json`
- `host_data/results/main_run/phase2_rankings.json`
- `host_data/results/main_run/figures/phase2_*.png`
- `host_data/results/main_run/figures/phase2_*.pdf`

### 3. Run FGSM Evaluation

```bash
./run_fgsm_eval.sh
./check_progress_fgsm.sh
```

This evaluates:

- configured FGSM epsilon sweeps
- SLO-based critical epsilon
- attack-surface sensitivity for core, distribution, and access tiers
- GNN embedding attacks for GNN variants

Outputs:

- `host_data/results/main_run/phase3_fgsm_results.json`
- `host_data/results/main_run/phase3_rankings.json`
- `host_data/results/main_run/figures/phase3_*.png`
- `host_data/results/main_run/figures/phase3_*.pdf`

### 4. Run Everything End-to-End

```bash
./run_full_experiment.sh
```

Use this only after the smoke test succeeds.

## Output Layout

```text
host_data/
	models/
		<variant-name>/
	results/
		main_run/
			phase1_training_results.json
			phase2_maddpg_results.json
			phase2_rankings.json
			phase3_fgsm_results.json
			phase3_rankings.json
			experiment_summary_rankings.json
			figures/
				phase1_*.png
				phase1_*.pdf
				phase2_*.png
				phase2_*.pdf
				phase3_*.png
				phase3_*.pdf
host_logs/
```

## How To Read The JSON Outputs

### `phase1_training_results.json`

Per variant:

- `rewards`: training reward trajectory
- `pkt_losses`: training packet-loss trajectory
- `final_reward`: final reward summary
- `final_pkt_loss`: final packet-loss summary

### `phase2_maddpg_results.json`

Per variant and scenario:

- `mean_reward`
- `mean_pkt_loss`
- `mean_delivery_rate`
- `mean_goodput_per_step`
- `mean_delay_p95`
- `mean_backlog_end`
- `mean_util_p95`

### `phase2_rankings.json`

Contains composite rankings by scenario and scoring profile, plus `seed_convergence` telemetry for adaptive ranking stability.

### `phase3_fgsm_results.json`

Per variant and attack case:

- `clean`
- `attacked`
- `metrics`
- `slo`
- `run_config`
- `attack_summary`
- `surface`

### `phase3_rankings.json`

Contains overall robustness ranking and per-attack-type ranking.

## Plot Generation

Plots are generated automatically at the end of each phase.

### Phase 1 plots

- reward evolution curves
- packet-loss evolution curves
- final reward ranking

### Phase 2 plots

- composite score bars by scenario and profile
- KPI heatmaps
- delivery heatmap across scenarios
- rank-stability curve when adaptive seed expansion is active

### Phase 3 plots

- packet loss vs epsilon by attack type
- degradation curves
- critical epsilon plot
- attack-surface sensitivity plot
- overall robustness ranking bars

### Re-generate plots manually

```bash
python tools/plot_results.py --results-dir host_data/results/main_run
```

Selected phases only:

```bash
python tools/plot_results.py --results-dir host_data/results/main_run --phases 1 2
```

## Can Plots Be Updated Every Epoch?

Yes, but that is intentionally not the default because it adds extra rendering and file I/O inside the longest-running loops.

Current implementation:

- generates plots once per completed phase
- keeps overhead negligible
- preserves one stable figure set per run

If you later want live monitoring, a safer compromise is checkpoint plotting every 10 or 20 epochs during training, not every epoch and not every evaluation episode.

## Running With Different Budgets

### Smoke test

```bash
./test_quick.sh
```

### Custom results directory

```bash
python src/standalone_experiment_runner.py --config experiment_config.json --phase train --results-dir data/results/custom_run
python src/standalone_experiment_runner.py --config experiment_config.json --phase paper1 --results-dir data/results/custom_run
python src/standalone_experiment_runner.py --config experiment_config.json --phase paper2 --results-dir data/results/custom_run
```

### Quick CLI run

```bash
python src/standalone_experiment_runner.py --config experiment_config.json --phase all --quick --results-dir data/results/quick_run
```

## Important Configuration Knobs

All runtime behavior is controlled by `experiment_config.json`.

### Training

- `training.epochs`
- `training.episodes_per_epoch`
- `training.timesteps_per_episode`

### Clean evaluation

- `paper1_eval.evaluation_episodes`

### Reward shaping

- `reward.delivery_weight`
- `reward.max_util_penalty`
- `reward.var_util_penalty`
- `reward.drop_penalty`
- `reward.backlog_penalty`

### FGSM evaluation

- `attack_configs`
- `fgsm_slo.max_pkt_loss_pct`
- `fgsm_slo.min_delivery_rate_pct`
- `fgsm_slo.max_reward_degradation_pct`
- `fgsm_slo.max_delay_p95`

### Runtime control and pruning

- `runtime_control.max_attack_cases_per_variant`
- `runtime_control.max_attack_variants`
- `runtime_control.phase3_enable_slo_pruning`
- `runtime_control.phase3_consecutive_fail_limit`
- `runtime_control.phase3_skip_after_critical_epsilon`

### Adaptive seed expansion

- `runtime_control.seed_expansion.enable_adaptive`
- `runtime_control.seed_expansion.initial_seeds`
- `runtime_control.seed_expansion.max_seeds_for_ranking`
- `runtime_control.seed_expansion.rank_stability_check_interval`
- `runtime_control.seed_expansion.top_k_for_stability`
- `runtime_control.seed_expansion.stability_threshold`

## Troubleshooting

### No plots were generated

Likely causes:

- `matplotlib` is unavailable in the runtime environment
- the phase failed before writing its JSON files

Check the phase logs first, then re-run plotting manually.

### Phase 2 or Phase 3 says no trained model

Run Phase 1 first, or keep the same `--results-dir` across phases so checkpoints and results stay together.

### GPU not detected

Verify Docker GPU support and confirm `nvidia-smi` works inside containers.

### Rankings look unstable

Increase:

- `paper1_eval.evaluation_episodes`
- `runtime_control.seed_expansion.max_seeds_for_ranking`

or reduce Phase 3 pruning aggressiveness.

## Scripts Reference

| Script | Purpose |
|---|---|
| `test_quick.sh` | detached smoke test |
| `check_progress_quick.sh` | tail smoke-test logs |
| `run_training.sh` | run Phase 1 |
| `check_progress_training.sh` | tail Phase 1 logs |
| `run_maddpg_eval.sh` | run Phase 2 |
| `check_progress_maddpg.sh` | tail Phase 2 logs |
| `run_fgsm_eval.sh` | run Phase 3 |
| `check_progress_fgsm.sh` | tail Phase 3 logs |
| `run_full_experiment.sh` | detached all-in-one run |
| `reset_results.sh` | delete results only |
| `reset.sh` | delete results and models |

## Local Result Inspection

```bash
scp -r user@server:/path/to/maddpg-adversarial-experiments/host_data/results/main_run ./main_run
python tools/plot_results.py --results-dir ./main_run
```