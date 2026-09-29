# MADDPG Adversarial Routing Experiments

MADDPG routing policies on a real 86-node service-provider (SP) topology, and their
robustness to observation-space adversarial attacks. The repository holds the code,
configs and result-generation tooling behind two papers and two MSc theses:

- **Paper 1** — architecture comparison: centralised vs. local critic, GNN encoding,
  against shortest-path / random-spreading / greedy baselines, with a failure-severity
  sweep and a 3-seed variance check. The manuscript lives **outside this repo**; the
  code, configs and figure scripts that produce its results are here.
- **Paper 2** (`paper/paper2_robustness.tex`, target IEEE TNSM) — how robust are these
  policies to observation attacks, from a myopic gradient (FGSM, iterated, logit-margin)
  to a learned worst-case adversary? Work still to do before submission is tracked in
  [`paper/PAPER2_ROADMAP.md`](paper/PAPER2_ROADMAP.md).
- **MSc theses** (handoffs complete): Gonçalo Martins on the FGSM results
  (`students/goncalo-martins-fgsm-thesis/`) and Miguel Chen on the learned adversary
  (`students/miguel-chen-learned-adversary/`).

---

## Repository layout

```text
src/
  standalone_experiment_runner.py   # the pipeline: train / evaluate / attack
  maddpg_clean/                     # MADDPG, networks, environment, topology
  attack_framework/
    improved_fgsm_attack.py         # FGSM/PGD, logit objectives, random control
    learned_adversary.py            # SA-MDP learned adversary
tools/                              # analysis and figure scripts (see below)
configs/                            # one-off experiment configs (seeds, logit, partial compromise)
paper/                              # Paper 2 LaTeX + roadmap; figures are generated
students/                           # per-student handoff dirs
experiment_config.json              # base config (training, baselines, sweeps)
reward_fix_full_config.json         # canonical stress-trained config + attack grid
run_phase.sh check_progress.sh clean_outputs.sh save_weights.sh load_weights.sh run_smoke.sh
Dockerfile requirements.txt pyproject.toml
```

**Not in the repo (gitignored, obtained out-of-band):** trained weights and raw
results under `host_data/`, container logs under `host_logs/`, and generated figures
(`*.png`, `*.pdf`, except the committed thesis figures). See
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
`src/standalone_experiment_runner.py` (phases: `train`, `paper1`, `paper2`, `hotspot`,
`failure`, `ceiling`, `fgsm_probe`, `all`).

---

## Weights & results

`host_data/` is **gitignored** — no weights or results are in git.

```text
host_data/results/
  reward_fix/                <phase JSONs>, models/<variant>/   # CANONICAL stress-trained victims
  ceil2x/<variant>/          damage_ceiling.json                # per-victim damage ceilings, 2x hotspot
  seeds/s1042 s2042/         models/, ceil_<variant>/           # extra training seeds (non-GNN)
  fgsm_tighten/<variant>/    fgsm_probe_results.json            # 15-episode paired FGSM probe
  fgsm_full/<variant>/       fgsm_probe_results.json            # earlier epsilon/load sweep
  logit_attack/<variant>/    fgsm_probe_results.json            # GNN-faithful FGSM + logit objectives
  seed_probe/s1042 s2042/    <variant>/...                      # FGSM probe on the seed victims
  partial_compromise_multi/  <variant>/...                      # 1/4/7/14 agents, 4 draws each
  pgd_diagnostic/            <variant>.json                     # iterated-attack tuning (T8)
  mchen/                     learned_adv_parity/, ...           # learned-adversary evaluations
```

Two traps, both of which have produced wrong numbers before:
- **GNN victims.** `fgsm_tighten` took the attack gradient through the actor alone,
  bypassing the encoder the victim decides through. Use `logit_attack` (run with
  `faithful_gnn: true`) for any GNN FGSM number.
- **Empty model folders.** A run whose `models/` holds no checkpoints evaluates an
  untrained network. The rule rows of `reward_fix/sweep_baselines/` are valid, its
  `policy` rows are not; take per-load policy PDR from
  `reward_fix/phase2_hotspot_sweep_results.json`.

Move weights between machines with:

```bash
./save_weights.sh user@server      # push local host_data/ weights to a server
./load_weights.sh user@server      # pull weights from a server into local host_data/
```

After pulling weights you can run evaluation and attacks **without retraining**.

---

## Training & clean evaluation (Paper 1)

`run_phase.sh <phase> [comma,separated,variants]` wraps the runner; monitor with
`check_progress.sh <container> follow`. Select a config with `CONFIG_PATH` (repo-relative,
e.g. `CONFIG_PATH=configs/gnn_seed_1042_config.json`) and an output dir with `RESULTS_DIR`.

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

**Figures:**
```bash
python tools/plot_paper1.py            # F1..F11
python tools/plot_seed_variance.py     # F12 (3-seed variance)
python tools/topo_invariants.py        # SP-class representativeness table
```

---

## Attacks (Paper 2)

### Gradient attacks: FGSM, iterated, logit objectives
No retraining. Point the runner at a config whose variants have trained checkpoints:
```bash
python src/standalone_experiment_runner.py --config reward_fix_full_config.json \
    --phase fgsm_probe --results-dir data/results/fgsm_tighten/CC-Simple
```
The runner resolves victim weights from `<results-dir>/models/<variant>/`, so symlink
the canonical models in first: `ln -sfn ../../reward_fix/models <results-dir>/models`.
The probe refuses to run if it finds no trained weights.

**Threat model.** Every attacker perturbs only the per-path bottleneck utilisations
the agent reads (`NetworkEngine.path_util_slots`, 61 of 94 observation slots), within
an L∞ ball of radius ε, clamped to [0,1]; queue state, timestep and destination flags
are local to the switch and never move. Results produced before 29 Sep 2026
(`fgsm_tighten`, `logit_attack`, `seed_probe`, `partial_compromise*`, `mchen`) used an
earlier attacker that perturbed all 94 slots and are superseded.
`tools/threat_model_check.py` verifies the invariants on real observations.

The `attack_eval` block of the config selects the grid. Keys that matter:
- `faithful_gnn: true` — differentiate GNN victims through their real decision path
  (encoder included, other agents' clean observations as context). Always set it.
- attack types `packet_loss` (FGSM objective), `logit_congestion` and `logit_margin`
  (single step on the pre-sigmoid logits; the FGSM objective is gradient-masked),
  `random` (budget-matched control).
- probe entries are `[type, epsilon, n_steps(, fraction_of_agents)]`; for n_steps > 1,
  `pgd_momentum` (MI-FGSM) and `pgd_alpha_frac` (step = fraction of ε) set the iteration.
- `configs/probe_*.json` hold `attack_eval` overrides for `tools/run_probe_grid.sh`,
  which runs the probe for several victims in parallel on the server:
  `tools/run_probe_grid.sh threat_util configs/probe_threat_util_sweep.json`.

Analysis (stdlib unless noted):
```bash
python tools/analyze_fgsm.py fgsm_tighten        # probe summary
python tools/analyze_logit_attack.py             # FGSM vs logit, GNN faithful vs original
python tools/analyze_seed_variance.py            # adversarial gap across training seeds
python tools/analyze_partial_compromise.py       # damage vs number of compromised agents
python tools/path_diversity_analysis.py          # vacuous flips from padded K-paths (torch)
python tools/gradient_signal_analysis.py         # why flips don't reach worse paths (torch)
python tools/pgd_diagnostic.py                   # iterated-attack budget spend (torch)
python tools/recompute_learned_vs_fgsm.py        # learned-adversary vs FGSM tables
```

Figures (T1–T9). The thesis copies go to the student dir; the paper copies drop the
in-figure titles:
```bash
python tools/plot_thesis.py                                  # -> students/goncalo-martins-fgsm-thesis/figures/
PAPER_MODE=1 FIG_DIR=paper/figures python tools/plot_thesis.py   # -> paper/figures/
```

### Learned adversary
Trains an SA-MDP adversary against a **frozen** victim and scores it with the same
paired protocol as FGSM. See `students/miguel-chen-learned-adversary/README.md`.
```bash
python tools/train_adversary.py --config reward_fix_full_config.json \
    --variant CC-Simple --episodes 300 --load 2.0 --epsilon 0.30 \
    --victim-models host_data/results/reward_fix/models \
    --out host_data/results/learned_adv/CC-Simple
python tools/train_adversary.py --config reward_fix_full_config.json \
    --variant CC-Simple --eval-only \
    --adv-ckpt host_data/results/learned_adv/CC-Simple/adversary.pt
```
The driver aborts if the victim weights are missing rather than training against a
random-init victim.

---

## Configuration

- `experiment_config.json` — base training/evaluation (epochs, traffic, reward,
  variants, sweeps).
- `reward_fix_full_config.json` — the **canonical** stress-trained setup (2× hotspot,
  `mean_util_weight=0.1`) plus the `attack_eval` grid.
- `configs/*.json` — full copies of the canonical config with the overrides for one
  reported run each (extra seeds, logit attack, partial compromise).

---

## Cleanup

```bash
./clean_outputs.sh                         # clear the default results dir
./clean_outputs.sh host_data/results/foo   # or a specific one
```
