# Student 2 — Miguel Chen — Learned Worst-Case Observation Adversary

**Task.** Pick up the code as it is and build the *learned* adversary that continues
the FGSM robustness study. The scaffold is written, verified end-to-end on the server,
and waiting for your two research extensions (coordinated + timed). This README is your
complete brief.

This is the follow-up to the FGSM robustness study. The FGSM result was a *negative*:
a myopic, per-agent gradient attack flips ~26% of routing decisions but extracts almost
none of the ~21pp damage ceiling, because the service-provider topology's K-path
redundancy absorbs the flips. **Your job is to find out whether that robustness is real
or whether FGSM was just too weak** — by building a *learned* adversary that optimises
whole trajectories and can coordinate across agents and time.

Everything you need to start is already in the repo. Clone, read this, run the smoke
commands, then pick up extension (A) or (B).

## The one question
> Can a trained, coordinated, well-timed observation attacker — within the same
> L∞ ε-budget as FGSM — approach the damage ceiling that FGSM cannot?
>
> - **If no (H1):** redundancy is a genuine structural adversarial defence. Strong result.
> - **If yes (H2):** the myopic result hid a real vulnerability. Also a strong result.

Either answer is a paper. The manuscript is drafted at `paper/paper2_robustness.tex`
(Section VI is yours to fill; the `[TO COMPLETE]` box marks exactly what).

## Threat model (unchanged from FGSM — keep it identical for comparability)
- Observation-space perturbation only; never touch traffic, topology, or reward.
- L∞-bounded by ε (default 0.30), utilisation/bandwidth features re-clamped to [0,1].
- Victim is a **frozen** trained MADDPG variant. Never update its weights.

## Files
| file | what it is |
|---|---|
| `src/attack_framework/learned_adversary.py` | the scaffold: adversary actor/critic, replay, DDPG trainer, and an **FGSM-compatible eval wrapper** so a trained adversary drops into the existing scoring loop |
| `tools/train_adversary.py` | driver: loads a frozen victim + attack env (reusing the runner), trains, saves, and `--eval-only` scores against the damage ceiling / random control |
| `src/attack_framework/improved_fgsm_attack.py` | the FGSM/PGD attacker + the `random` control you are compared against |
| `paper/paper2_robustness.tex` | the paper; your results fill Section VI |

## What you need BESIDES cloning the repo
Cloning gives you the code, not the inputs. You also need, all obtained from the
server / shared drive (they are **gitignored, not in the repo**):
1. **The trained victim weights** — `reward_fix/models/<variant>/...` (the seven
   trained MADDPG policies). Place them under `host_data/results/reward_fix/models/`.
2. **The config** — `reward_fix_full_config.json` (now committed to the repo root).
3. **The runtime** — the `maddpg-exp:latest` docker image (or build from the tracked
   `Dockerfile` + `requirements.txt`; local pip has been unreliable).

Note: `MADDPG.load_checkpoint()` fails **silently** if a weight file is missing, so the
driver has a hard guard — if you point `--victim-models` at the wrong place it aborts
with a clear message instead of training against a random-init victim. (You can tell a
correctly-loaded victim by its clean PDR ~87% at 2× hotspot; a mis-loaded one sits
~55%.)

## Run it (inside the `maddpg-exp` docker image)
Train an adversary against CC-Simple at 2× hotspot (`--victim-models` defaults to
`host_data/results/reward_fix/models`, shown explicitly here):
```bash
python tools/train_adversary.py --config reward_fix_full_config.json \
    --variant CC-Simple --episodes 300 --load 2.0 --epsilon 0.30 \
    --victim-models host_data/results/reward_fix/models \
    --out host_data/results/learned_adv/CC-Simple
```
Score the trained adversary with the SAME metrics as FGSM (paired clean/attacked PDR,
drop, action-flip rate):
```bash
python tools/train_adversary.py --config reward_fix_full_config.json \
    --variant CC-Simple --eval-only \
    --adv-ckpt host_data/results/learned_adv/CC-Simple/adversary.pt \
    --load 2.0 --epsilon 0.30
```
Both commands are verified end-to-end on the server: the driver loads the trained
victim ("loaded best checkpoint"), trains, saves, and `--eval-only` emits the identical
metric shape as `fgsm_probe`. A real run needs a few hundred episodes to converge.

## Your two research extensions (the actual contribution)
Both are marked `TODO(student ...)` in `learned_adversary.py`:

- **(A) Coordinated multi-agent perturbation** — replace the independent per-agent loop
  in `_attack_states` with a *joint* adversary: concatenate the compromised agents'
  observations, emit a joint δ, and give the critic the global link state, so the
  attacker can push several flows onto **one shared surviving link**. This is the
  mechanism the per-agent gradient literally cannot express, and the one most likely to
  reach the ceiling. See `--coordinate`.
- **(B) Strategically-timed / critical-state attack** — given an L0 budget (fraction of
  steps allowed), gate perturbation on a critical-state score (max link utilisation, or
  a Q-saliency `|Q(clean) − Q(perturbed)|`) so the attacker spends budget only at
  high-leverage moments — congestion onset, or right after a failure when redundancy is
  thin (exactly the window the FGSM study found the network is most sensitive). See
  `--timing-budget`.

## Suggested milestones
1. **Reproduce the FGSM baseline** you're improving on (`--phase fgsm_probe`, existing).
2. **Per-agent learned adversary converges** — training reward (victim step-loss) rises;
   eval drop ≥ FGSM's on at least the exploitable variants (CC-Duelling, LC-Simple).
3. **Coordinated variant (A)** — compare achieved-vs-ceiling to the per-agent version.
4. **Timed variant (B)** — same damage at a fraction of the perturbation budget?
5. **Failure sweep** — does the learned adversary beat the *random* control under
   failures (where FGSM did not)? That is the sharpest H1-vs-H2 test.
6. **Fill Section VI** of the paper and the comparison figure.

## Pitfalls (learned from the FGSM study)
- **Report the adversarial-specific gap (learned − random), not the raw drop.** Under
  failures the raw drop is huge but mostly noise; only the gap over the random control
  is evidence of a real attack.
- **Keep episodes paired** (same traffic + failure seed for clean/random/learned) or the
  CIs are invalid.
- **Flip rate ≠ success.** A high flip rate with flat PDR is the network absorbing the
  attack, not the attack winning. Track decisions and outcomes separately.
- Watch for the degenerate high-failure cells (n≈6) where the network self-collapses;
  they are failure-dominated, not attack-informative.
