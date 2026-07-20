# MADDPG Adversarial Routing Experiments

MADDPG routing policies on a real 86-node service-provider (SP) topology, and their
robustness to observation-space adversarial attacks. The repository backs two papers
and two ongoing MSc projects:

- **Paper 1** (`paper/paper1_revised.tex`) — architecture comparison: centralised vs.
  local critic, GNN encoding, against shortest-path / random-spreading / greedy
  baselines, with a failure-severity sweep and a 3-seed variance check.
- **Paper 2** (`paper/paper2_fgsm.tex`, and the broader `paper/paper2_robustness.tex`)
  — do observation attacks (FGSM/PGD, and a learned adversary) actually degrade the
  policy?
- **Student 1 — Gonçalo Martins** (`students/goncalo-martins-fgsm-thesis/`) — MSc
  thesis on the FGSM results that already exist.
- **Student 2 — Miguel Chen** (`students/miguel-chen-learned-adversary/`) — continues
  the code with a learned worst-case adversary.

---

## Repository layout

```text
src/
  standalone_experiment_runner.py   # the pipeline: train / evaluate / attack
  maddpg_clean/                     # MADDPG, networks, environment, topology
  attack_framework/
    improved_fgsm_attack.py         # FGSM/PGD + the random control
    learned_adversary.py            # SA-MDP learned adversary (student 2 scaffold)
tools/
  plot_paper1.py  plot_seed_variance.py   # Paper 1 figures (F1..F12)
  plot_thesis.py                          # Paper 2 / FGSM-thesis figures (T1..T7)
  analyze_fgsm.py  topo_invariants.py     # Paper 2 analysis, Paper 1 topology table
  train_adversary.py                      # learned-adversary trainer (student 2)
  plot_topology.py  plot_traffic_matrix.py  plot_results.py
paper/                                # LaTeX + committed figures (in writing)
students/                             # per-student handoff dirs (see below)
experiment_config.json                # base config (training, baselines, sweeps)
reward_fix_full_config.json           # canonical stress-trained config + attack grid
run_phase.sh check_progress.sh clean_outputs.sh save_weights.sh load_weights.sh
Dockerfile requirements.txt pyproject.toml
```

**Not in the repo (gitignored, obtained out-of-band):** trained weights and raw
results under `host_data/`, and container logs under `host_logs/`. See
[Weights & results](#weights--results).

---

## Setup

Everything runs inside the Docker image (local pip has been unreliable for this
project — prefer Docker):

```bash
docker build -t maddpg-exp:latest .      # or docker pull, if you have the image
```

The image provides torch, numpy, networkx, matplotlib, scipy. The shell scripts wrap
`docker run` with the right volume mounts; the raw runner is
`src/standalone_experiment_runner.py`.

---

## Weights & results

`host_data/` is **gitignored** — no weights or results are in git.

```text
host_data/results/
  reward_fix/          models/<variant>/...   # CANONICAL stress-trained victims
  reward_fix/          <phase JSONs>, figures/
  fgsm_tighten/        <variant>/fgsm_probe_results.json   # Paper 2 attack results
  seeds/               s1042/ s2042/ ...       # Paper 1 multi-seed
```

The canonical trained victims live in `host_data/results/reward_fix/models/<variant>/`.
Move weights between machines with:

```bash
./save_weights.sh user@server      # push local host_data/ weights to a server
./load_weights.sh user@server      # pull weights from a server into local host_data/
```

After pulling weights you can run evaluation and attacks **without retraining**.

---

## Paper 1 — training & evaluation

`run_phase.sh <phase> [comma,separated,variants]` wraps the runner; monitor with
`check_progress.sh <container> follow`. Phases: `train`, `paper1` (clean eval),
`paper2` (FGSM eval), `hotspot`, `failure`, `all`.

**Train** (writes checkpoints to `host_data/results/<run>/models/<variant>/`):
```bash
./run_phase.sh train                       # all variants
./run_phase.sh train CC-Simple,LC-Duelling # a subset
```
Training is resumable (skips variants whose `phase1_training_results.json` + checkpoints
exist). Best-validation checkpoints are used for all downstream evaluation.

**Clean evaluation** (load sweep, baselines, ceiling — consumes existing checkpoints):
```bash
./run_phase.sh paper1
```

**Figures** (from the result JSON, rendered in the Docker image):
```bash
python tools/plot_paper1.py            # F1..F11
python tools/plot_seed_variance.py     # F12 (3-seed variance)
python tools/topo_invariants.py        # SP-class representativeness table
```
Paper 1 figures are written to `host_data/results/reward_fix/figures/`.

---

## Attacks

### FGSM / PGD — with weights already present (Paper 2, Student 1)
No retraining. Point the runner at a config whose variants have trained checkpoints
(the canonical `reward_fix_full_config.json`), then:
```bash
# damage-ceiling + random-control + action-flip probe across the load/failure grid
python src/standalone_experiment_runner.py --config reward_fix_full_config.json \
    --phase fgsm_probe --results-dir data/results/fgsm_tighten/CC-Simple
```
The runner resolves victim weights from `<results-dir>/models/<variant>/`, so symlink
the canonical models in first (the FGSM run scripts do this):
`ln -sfn ../../reward_fix/models <results-dir>/models`.

Analyse and plot:
```bash
python tools/analyze_fgsm.py           # stdlib summary of the probe JSON
python tools/plot_thesis.py            # T1..T7 -> students/goncalo-martins-fgsm-thesis/figures/
```

### Learned adversary — with weights already present (Student 2)
The scaffold trains a SA-MDP adversary against a **frozen** victim and scores it with
the same metrics as FGSM. See `students/miguel-chen-learned-adversary/README.md`.
```bash
# train
python tools/train_adversary.py --config reward_fix_full_config.json \
    --variant CC-Simple --episodes 300 --load 2.0 --epsilon 0.30 \
    --victim-models host_data/results/reward_fix/models \
    --out host_data/results/learned_adv/CC-Simple
# score the trained adversary (paired clean/attacked PDR, drop, flip-rate)
python tools/train_adversary.py --config reward_fix_full_config.json \
    --variant CC-Simple --eval-only \
    --adv-ckpt host_data/results/learned_adv/CC-Simple/adversary.pt
```
`--victim-models` is symlinked in so the loader finds the weights; the driver **aborts
with a clear message** if the weights are missing (rather than silently training against
a random-init victim).

---

## Configuration

Two configs, both driving `src/standalone_experiment_runner.py`:
- `experiment_config.json` — base training/evaluation (epochs, traffic, reward,
  variants, sweeps).
- `reward_fix_full_config.json` — the **canonical** stress-trained setup (2× hotspot,
  corrected reward weights) plus the `attack_eval` grid used for the FGSM/attack work.

Select a subset of variants with the second arg to `run_phase.sh`, or `--variants` on
the runner directly.

---

## Students (ongoing work)

- **`students/goncalo-martins-fgsm-thesis/`** — MSc thesis on the FGSM results.
  Contains the narrative guide and the finished T1–T7 figures (committed). No
  experiments needed to write.
- **`students/miguel-chen-learned-adversary/`** — the learned-adversary continuation:
  the full project brief, the two research extensions (coordinated + timed), verified
  run commands, and milestones.

---

## Cleanup

```bash
./clean_outputs.sh                         # clear the default results dir
./clean_outputs.sh host_data/results/foo   # or a specific one
```
