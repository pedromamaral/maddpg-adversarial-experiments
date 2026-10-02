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

> **Status (2 Oct 2026): the learning is being fixed and every variant retrained.** The
> policies trained so far (v1) never learned to route. They behave as near-static
> routing tables that ignore their utilisation telemetry, and a greedy least-utilised
> rule beats all of them. `tools/learnability_probe.py` reproduces the failure in a
> minute and shows its two causes: a critic that cannot credit a single destination's
> choice, and independent sigmoid actor outputs that saturate into a fixed table.
> Paper 1's comparison and Paper 2's attack study, both built on v1 policies, are on hold.
> The v1 results are archived in `host_data/archive_v1/`. The results that stay valid
> (policy-independent baselines) are listed under [Weights & results](#weights--results).

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
host_data/
  results/                                   # what stays valid after retraining
    reward_fix/       models/<variant>/        v1 canonical victims and training curves (reference)
                      phase2_hotspot_sweep_results.json   shortest-path rows per load (policy rows are v1)
                      sweep_baselines/load_*/  greedy / random / sp / worst rules per load
    unisweep/         uniform-traffic sweep    shortest-path rows (policy rows are v1)
    ceil2x/<variant>/ damage_ceiling.json      rules at 2x hotspot (policy rows are v1)
    p1_ceiling_1x/    damage_ceiling.json      rules at 1x uniform
    failsev_fixed/n_<k>/ damage_ceiling.json   rules under k = 0..8 random link failures
    telemetry_reliance/, policy_attribution.json   evidence that v1 ignores its telemetry
  archive_v1/                                # superseded, kept for the record and the theses
    results/          every attack run on v1 victims, extra seeds, old models (main_run,
                      stress_run, seeds, seeds_gnn), the pre-fix failure sweeps
    server_home/, repo_untracked/            old server scripts, logs and stray configs
  topo_check/         zoo/*.graphml, our_topo.txt   input of tools/topo_invariants.py
```

The rule rows (greedy, random, sp, worst) and the shortest-path rows do not depend on any
trained policy, so they serve as baselines for the retrained policies unchanged. The
thesis figures were drawn from what is now `archive_v1/results` (plus `reward_fix` and
`ceil2x`); point `RESULTS_ROOT` there to regenerate them.

Three traps, all of which have produced wrong numbers before:
- **Accumulating link failures** (fixed in `8b4a4f9`). Failures were never undone between
  chained evaluations, so each rule or attack arm after the first inherited the previous
  ones' failed links. Every failure result before the fix is superseded.
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

**Does a policy actually route?** Check before trusting any comparison:
```bash
python tools/learnability_probe.py --critic factored --head softmax  # can the learner learn the easy case?
python tools/policy_attribution.py --failures 0,2,4   # static share, greedy agreement, saturation
tools/run_probe_grid.sh telemetry_reliance configs/probe_telemetry_reliance.json
python tools/analyze_telemetry_reliance.py            # does delivery depend on the utilisation signal?
tools/run_failsev.sh failsev_fixed CC-Simple "1 2 4 6 8"   # rules vs policy under link failures
```

**Figures:**
```bash
python tools/plot_paper1.py            # F1..F11
python tools/plot_seed_variance.py     # F12 (3-seed variance)
ZOO_DIR=host_data/topo_check/zoo OUR_TOPO=host_data/topo_check/our_topo.txt \
    python tools/topo_invariants.py    # SP-class representativeness table
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

Analysis (stdlib):
```bash
python tools/analyze_threat_util.py     # budget sweep + iterated attack vs random control, % of ceiling
python tools/analyze_stage_b.py         # extra seeds, failure sweep, partial compromise
python tools/threat_model_check.py      # attack invariants on real observations (torch)
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
- `configs/*_config.json` — full copies of the canonical config with the overrides of one
  v1 run each (GNN seeds, logit attack, partial compromise, seed probe).
- `configs/probe_*.json` — `attack_eval` overrides for `tools/run_probe_grid.sh`.

---

## Cleanup

```bash
./clean_outputs.sh                         # clear the default results dir
./clean_outputs.sh host_data/results/foo   # or a specific one
```
