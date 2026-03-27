# MADDPG Adversarial Robustness Framework

Self-contained MADDPG adversarial robustness evaluation framework, designed to run on a remote GPU server via Docker in detached mode.

---

## Prerequisites

- Remote host with Docker + NVIDIA Container Toolkit
- Git

---

## Workflow

### 1. Clone & build

```bash
git clone https://github.com/pedromamaral/maddpg-adversarial-experiments.git
cd maddpg-adversarial-experiments
docker build -t maddpg-adversarial:latest .
```

The build takes ~5 minutes and installs all dependencies inside the image.

### 2. Validate (optional but recommended)

```bash
./test_quick.sh
```

Launches a quick smoke-test in the background (~2 min). Check its output with:

```bash
./check_progress_quick.sh
```

### 3. Run the full experiment

```bash
./run_full_experiment.sh
```

Runs all 6 architecture variants with all 3 attack types in the background. Monitor progress with:

```bash
./check_progress_full.sh
```

Results are written to `host_data/results/` on the host.

---

## Re-running experiments

If results are incomplete, wrong, or you've made code changes:

```bash
# 1. Pull latest code (if you made changes)
git fetch origin && git reset --hard origin/main

# 2. Rebuild the image (if code changed)
docker build -t maddpg-adversarial:latest .

# 3. Wipe previous run and reset workspace
./reset.sh

# 4. Re-run
./run_full_experiment.sh
./check_progress_full.sh
```

`reset.sh` stops any running containers and clears `host_data/results/` and `host_logs/`. It preserves `host_data/models/` and the Docker image.

---

## Scripts reference

| Script | What it does |
|---|---|
| `test_quick.sh` | Launches quick smoke-test in detached mode |
| `run_full_experiment.sh` | Launches full experiment in detached mode |
| `check_progress_quick.sh` | Tails logs from the quick-test container |
| `check_progress_full.sh` | Tails logs from the full experiment container |
| `reset.sh` | Stops containers, wipes results, resets workspace |

---

## Output

All results are written to the host under `host_data/`:

```
host_data/
  results/   # JSON metrics, CSVs, plots
  models/    # Saved model checkpoints
host_logs/   # Container stdout/stderr logs
```

---

## Generating Graphs

After the experiment has finished and results are available in `host_data/results/`, pull them to your local machine and run the plotting tools. All outputs are saved to `tools/figures/` as both `.pdf` and `.png`.

### Prerequisites (local machine, no Docker needed)

```bash
pip install matplotlib numpy networkx
```

### 1. Pull results from the server

```bash
scp -r user@your-server:/path/to/maddpg-adversarial-experiments/host_data/results ./host_data/results
```

### 2. Plot experiment results

Generates reward curves, packet-loss comparison, and robustness-degradation figures:

```bash
python tools/plot_results.py --results host_data/results/
```

Outputs saved to `tools/figures/`:

| File | Description |
|---|---|
| `reward_curves.pdf/.png` | Episode reward vs. training step per architecture |
| `packet_loss_comparison.pdf/.png` | Packet loss bar chart per architecture per attack |
| `robustness_degradation.pdf/.png` | (clean − attacked) / clean performance degradation |

### 3. Plot network topology

Generates figures of the 65-node service-provider topology used in the experiments:

```bash
python tools/plot_topology.py
```

Outputs saved to `tools/figures/`:

| File | Description |
|---|---|
| `topology_full.pdf/.png` | Full 65-node service-provider graph |
| `topology_tiers.pdf/.png` | Same graph, nodes coloured by tier |
| `degree_distribution.pdf/.png` | Node degree histogram |

### 4. Plot traffic demand model

Generates figures of the traffic demand model used in the experiments:

```bash
python tools/plot_traffic_matrix.py
```

Outputs saved to `tools/figures/`:

| File | Description |
|---|---|
| `traffic_matrix_heatmap.pdf/.png` | 65×65 empirical demand heatmap |
| `traffic_flow_stats.pdf/.png` | Flow count, packet distribution, priority pie |
| `traffic_source_load.pdf/.png` | Per-node outgoing flow load |
