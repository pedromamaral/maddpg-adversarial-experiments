# MADDPG Adversarial Robustness Framework

Hop-by-hop MADDPG routing with adversarial attack evaluation.
All 65 nodes act as independent agents; packets are forwarded one hop at a time.

| Paper | What it produces |
|---|---|
| **MADDPG architecture Comparisson** — Architecture comparison | Convergence curves, packet-loss under normal + dual-link failure, OSPF baseline |
| **Adversarial FGSM study** — Adversarial attack study | FGSM impact across ε sweep, core vs access attack surface, GNN embedding attack |

---

## Prerequisites

- Remote host with Docker ≥ 20 and NVIDIA Container Toolkit
- Git

---

## Quick-start

```bash
git clone https://github.com/pedromamaral/maddpg-adversarial-experiments.git
cd maddpg-adversarial-experiments
docker build -t maddpg-exp:latest .
```

---

## Workflow
Step 0 Smoke-test ← always run first
Step 1 Training ← trains all 6 variants, saves weights (~8–12 h)
Step 2 Paper-1 eval ← architecture comparison + OSPF baseline (~1–2 h)
Step 3 Paper-2 eval ← adversarial attack study (~4–6 h)
Step 4 Pull & plot ← local machine, no Docker needed

---

## Step 0 — Smoke-test

```bash
./test_quick.sh
./check_progress_quick.sh
```

Completes in ~2 minutes. Fix any errors before proceeding.

---

## Step 1 — Training

```bash
./run_training.sh
./check_progress_training.sh
```

Artefacts saved under `host_data/`:
host_data/
results/phase1_training_results.json
models/CC-Simple/
models/CC-Duelling/
models/LC-Duelling/
models/CC-Simple-GNN/
models/CC-Duelling-GNN/
models/LC-Duelling-GNN/

---

## Step 2 — MADDPG evaluation

Requires Step 1 to be complete.

```bash
./run_maddpg_eval.sh
./check_progress_maddpg.sh
```

Evaluates:
- All 6 MADDPG variants under **normal traffic** and **dual-link failure**
- **OSPF baseline** (shortest-path, no learning) under both scenarios

Output: `host_data/results/phase2_maddpg_results.json`

---

## Step 3 — FGSM attack evaluation

Requires Step 1 to be complete.

```bash
./run_fgsm_eval.sh
./check_progress_fgsm.sh
```

Evaluates:
- FGSM sweep: `packet_loss`, `reward_minimize`, `confusion` × ε ∈ {0.01, 0.05, 0.10, 0.15, 0.20}
- Attack surface: core-only vs dist-only vs access-only node targeting at ε = 0.10
- GNN embedding attack (GNN variants only)

Output: `host_data/results/phase3_fgsm_results.json`

---

## Step 4 — Pull results and plot (local machine)

```bash
pip install matplotlib numpy networkx
scp -r user@server:/path/to/repo/host_data ./host_data

python tools/plot_results.py  --results host_data/results/main_run/phase1_training_results.json
python tools/plot_maddpg.py   --results host_data/results/main_run/phase2_paper1_results.json
python tools/plot_fgsm.py   --results host_data/results/main_run/phase3_paper2_results.json
```

---

## Re-running individual phases

```bash
git fetch origin && git reset --hard origin/main
docker build -t maddpg-exp:latest .

./reset_results.sh   # wipe results only, keep trained models
./run_paper1_eval.sh --results-dir host_data/results/<existing-dir>
./run_paper2_eval.sh --results-dir host_data/results/<existing-dir>
```

---

## Scripts reference

| Script | Phase | What it does |
|---|---|---|
| `test_quick.sh` | 0 | Smoke-test (1 epoch, detached) |
| `check_progress_quick.sh` | 0 | Tail smoke-test logs |
| `run_training.sh` | 1 | Full training, all 6 variants |
| `check_progress_training.sh` | 1 | Tail training logs |
| `run_maddpg_eval.sh` | 2 | maddpg evaluation |
| `check_progress_paper1.sh` | 2 | Tail maddpg logs |
| `run_paper2_eval.sh` | 3 | fgsm attack evaluation |
| `check_progress_paper2.sh` | 3 | Tail fgsm logs |
| `reset_results.sh` | — | Wipe results, keep models |
| `reset.sh` | — | Full wipe including models |

---

## Output layout
host_data/
models/<variant-name>/ PyTorch checkpoints
results/
phase1_training_results.json
phase2_paper1_results.json
phase3_paper2_results.json
host_logs/
training.log
paper1_eval.log
paper2_eval.log