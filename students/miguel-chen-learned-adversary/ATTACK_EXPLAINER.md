# Understanding Your Attack — from DRL/MADDPG to the Learned Adversary

A conceptual guide to the attack you will build and study: what it does, how it works,
and how it relates to FGSM/PGD and to the reinforcement-learning machinery of the victim.
Read this once before touching the code; it is the "why" behind `learned_adversary.py`.

---

## 1. The system you are attacking (MADDPG routing in one page)

**Reinforcement learning (DRL).** An *agent* observes a *state*, takes an *action*, and
receives a *reward*. Its *policy* π maps observations to actions. Through trial and error
it adjusts π to maximise long-run reward. A *deep* RL agent represents π (and a helper
*value function* / *critic*) with neural networks.

**MADDPG (the victim).** Routing is framed as a *multi-agent* problem: each provider-edge
(PE) switch is an agent. It uses **Centralised Training, Decentralised Execution (CTDE)**:
- Each agent has an **actor** πᵢ that, from its local observation, outputs *logits* over
  `K=3` candidate paths **per destination**; the executed action is an `argmax` — pick the
  highest-scoring path for each destination.
- A **critic** estimates how good an action is; it is used **only during training** to
  teach the actors, then discarded at deployment.

**The observation is the attack surface.** The critical part of an agent's observation is
`U_i` — the per-path **bottleneck (max-over-hops) utilisation** of each candidate path. It
is exactly what lets a smart agent avoid a congested path. It is also exactly what your
attacker perturbs.

**The decision is discrete.** Because the action is an `argmax` over path logits, the
attacker's job is to move the continuous observation *just enough* to flip that `argmax`
onto a worse path. Two quantities matter throughout:
- **Action-flip rate** — did the attack change the *decision*?
- **Delivery drop (PDR)** — did the attack change the *outcome*?
These are not the same, and telling them apart is the whole story (Section 5).

---

## 2. The threat model (the rules of the game)

Your attacker may **only** add a perturbation δ to an agent's observation, bounded so no
single feature moves by more than ε (an **L∞ ball** of radius ε), and utilisation/bandwidth
features stay in `[0,1]`. It may **not** touch the network, the traffic, or the reward.
This models a **compromised telemetry channel** — the monitoring feed the routing agents
read has been tampered with. The attacker's goal is to **maximise the victim's packet
loss** (minimise delivery). ε is the attacker's *budget*: bigger ε = stronger but more
detectable tampering.

---

## 3. FGSM — the one-shot gradient attack (the baseline)

**Origin.** FGSM (Fast Gradient Sign Method) comes from image classification: to fool a
network, nudge the input in the direction that most increases the network's loss. The
gradient ∇ₓL points that way; take one step along its sign.

**Here.** We define a differentiable **packet-loss objective** that is large when the
policy puts probability mass on *congested* paths:
```
L(δ) = Σ_paths  π(path) · σ((U − 0.5)·10)
```
`σ((U−0.5)·10)` is a sharpened weight ≈1 for congested paths (U>0.5), ≈0 for free ones.
Maximising L means "push the policy toward the busy paths" — i.e. toward worse routing.

**The FGSM step.** Perturb the observation by
```
δ = ε · sign(∇_obs L)
```
This jumps to the corner of the ε-ball that most increases the congestion objective — one
gradient evaluation, one step. If that step moves a path's logit across the `argmax`
boundary, the agent's chosen path **flips** to a more congested one.

**Its defining trait: it is myopic.** FGSM optimises the *objective at the current step*,
for *each agent independently*. It does not reason about the future, and it does not
coordinate agents.

---

## 4. PGD — iterated FGSM

PGD (Projected Gradient Descent) is FGSM done in several smaller steps: take a step, clip
(*project*) back into the ε-ball, repeat. It searches the ε-ball more thoroughly and is the
standard "we tried harder" upper bound on gradient-attack strength. On this routing problem
it flips only marginally more decisions than FGSM, because the `argmax` margins are small —
one good step already crosses most of them. **PGD is still myopic and still per-agent.**

---

## 5. Why FGSM/PGD "work but don't" here — and why you exist

The completed FGSM study (Paper 2) found something sharp:

> FGSM flips **~26%** of routing decisions — three times a random perturbation of the same
> size — yet delivery drops by **≤2.8pp** of a **~21pp** "damage ceiling." The network's
> `K`-shortest-path **redundancy absorbs the flips**: a flipped flow lands on a *different*
> good path and still gets delivered.

So FGSM is a *competent decision-level attack* that is *not an effective outcome-level
attack* — on a well-provisioned service-provider topology. But that negative result has an
uncomfortable ambiguity, and resolving it is your project:

- Maybe **robustness is real** (the topology genuinely defends), **or**
- Maybe **FGSM was just too weak** to find the attack that matters.

Two concrete weaknesses of FGSM point at what a stronger attack would do:
1. **It is myopic.** Congestion in a network *builds over time*. A one-step gradient can't
   choose to sacrifice an immediate flip in order to engineer a downstream collapse.
2. **It is per-agent and uncoordinated.** The worst-case damage (the "ceiling") is an
   inherently *coordinated* outcome — many flows herded onto one shared bottleneck. FGSM
   perturbs each agent alone and cannot express that.

Your attack removes both limitations.

---

## 6. The learned adversary — your attack (SA-MDP)

**The core idea: make the attacker itself a reinforcement-learning agent.** Instead of a
hand-crafted one-step gradient, *train a policy that outputs perturbations*, rewarded for
how much victim delivery it destroys over a whole episode. This is the **State-Adversarial
MDP (SA-MDP)** of Zhang et al. (NeurIPS 2020): the optimal observation attacker is itself
the solution to an RL problem.

Formally, fix the (frozen) victim π. The **adversary** is a policy ν that maps the true
observation `o` to a perturbation δ inside the ε-ball. The victim then acts on the tampered
observation `õ = Π(o + δ)` (Π projects back into the ε-ball and the valid range). The
adversary's **reward is the victim's per-step packet loss**:
```
r_adv(t) = L_t          (victim's fraction of packets lost this step)
```
Maximising the adversary's *return* (discounted sum of r_adv) = minimising the victim's
delivery **over the trajectory**.

**How it is trained: DDPG.** DDPG (Deep Deterministic Policy Gradient) is the single-agent
cousin of MADDPG — an actor-critic method for continuous actions (here the "action" is the
continuous perturbation δ). The loop:
1. The **adversary actor** ν emits δ; project into the ε-ball; the victim acts; the env
   steps; record the victim's loss as the reward.
2. Store the transition in a replay buffer.
3. The **adversary critic** Q learns to predict the return from `(observation, δ)` (a
   temporal-difference update).
4. The actor is nudged to output perturbations the critic rates highly — **gradient
   *ascent* on the victim's loss**.
5. Soft-update target networks; repeat.

**What it can do that FGSM cannot:**
- **Non-myopic.** Because it optimises the *discounted trajectory* return, it can learn to
  set up congestion now that pays off in a bigger collapse later.
- **(Extension A) Coordinated.** A single adversary emitting a *joint* perturbation over
  several agents, with a critic that sees the global link state, can push multiple flows
  onto one shared surviving link — the ceiling-reaching mechanism.
- **(Extension B) Strategically timed.** Given a budget on *how often* it may act, it learns
  to spend perturbations only at high-leverage moments (congestion onset; right after a
  link failure, when redundancy is thin — exactly where the FGSM study saw the network
  become sensitive).

It is, in short, the **worst-case** observation attacker within the same ε-budget as FGSM.

---

## 7. The symmetry worth internalising

Attacker and victim are **the same kind of object with opposite goals**:

| | Victim (MADDPG) | Adversary (yours) |
|---|---|---|
| what it is | actor–critic policy | actor–critic policy |
| observes | telemetry `o` | telemetry `o` |
| acts | choose paths (argmax) | emit perturbation δ (ε-ball) |
| reward | **minimise** loss | **maximise** the victim's loss |
| trained by | MADDPG (CTDE) | DDPG (SA-MDP) |

This is a **two-player adversarial game**: one network learns to route well, another learns
to make it route badly by lying to it — within a bounded lie. Your study measures who wins.

There is also a neat parallel to **Paper 1**: there, a hand-crafted *greedy* heuristic was a
one-step approximation to what the *learned* MADDPG policy does over a trajectory. Here,
**FGSM is the greedy heuristic of attacks, and your learned adversary is the trained policy**
— the same "myopic rule vs. learned strategy" contrast, now on the attacker's side.

---

## 8. The scientific question you are answering

Frame every result against the **damage ceiling** (what a perfect decision-steerer could
destroy, ~21pp) and the **random control** (a budget-matched *non-adversarial* perturbation —
the "value of intelligence" is `learned_drop − random_drop`).

- **H1 — robustness is fundamental.** If even your trained/coordinated/timed adversary
  extracts only a small fraction of the ceiling at nominal load, redundancy is a genuine
  structural defence, and the FGSM negative was *not* an artefact of a weak attack. Strong,
  clean result.
- **H2 — myopia hid a vulnerability.** If your adversary extracts substantially more than
  FGSM — especially the coordinated variant nearing the ceiling — then observation attacks
  *can* degrade the policy once they optimise trajectories and coordinate, and the defensive
  recommendation shifts from "redundancy suffices" to "telemetry integrity is essential."

Either outcome is a paper (you fill Section VI of `paper/paper2_robustness.tex`).

---

## 9. How the concepts map to the code

`src/attack_framework/learned_adversary.py`:
| concept | code |
|---|---|
| adversary policy ν | `AdversaryActor` (obs → δ direction, scaled by ε) |
| adversary value Q | `AdversaryCritic` |
| ε-ball + domain projection Π | `LearnedObservationAdversary._project` |
| the SA-MDP DDPG loop | `AdversaryTrainer.train` (reward = `info["packet_loss_rate"]`) |
| drop-in eval (same metrics as FGSM) | `LearnedObservationAdversary.generate_adversarial_state` |
| Extension A (coordinate) | `TODO(student A)` in `_attack_states` |
| Extension B (timed) | `TODO(student B)` in `_attack_states` |

The victim is the **frozen** MADDPG loaded by `tools/train_adversary.py`; the FGSM/PGD
baseline and the random control live in `improved_fgsm_attack.py`. Your adversary is scored
by the *same* paired protocol, so its numbers sit directly beside FGSM's.

---

## 10. Mental models to keep

- **Decisions ≠ outcomes.** A high flip rate with flat delivery is the network *absorbing*
  the attack, not the attack winning. Always report both.
- **The ε-ball is the budget.** Everything is "the most damage achievable within ε."
- **Gradient sign = fastest local increase** of the objective — a *greedy* move. A learned
  policy can be *strategic* instead of greedy: that is your entire edge over FGSM.
- **Always subtract the random control.** Under failures the raw drop is huge but mostly
  noise; only `learned − random` is evidence of a *real, intelligent* attack.
