#!/usr/bin/env python
"""
Entry point for the learned worst-case observation adversary (MSc follow-up).

It reuses the existing experiment runner to build the environment and load a
FROZEN trained victim, then trains a SA-MDP DDPG adversary against it and (optionally)
evaluates the trained adversary through the SAME attack loop used for FGSM, so the
numbers are directly comparable to the FGSM/PGD results in the paper.

Usage (inside the maddpg-exp docker image, GPU optional):
    python tools/train_adversary.py \
        --config reward_fix_full_config.json \
        --variant CC-Simple \
        --episodes 300 --load 2.0 --failures 0 \
        --epsilon 0.30 \
        --out host_data/results/learned_adv/CC-Simple

Then evaluate the trained adversary against the damage ceiling / random control:
    python tools/train_adversary.py ... --eval-only --adv-ckpt <path>/adversary.pt

This is a SCAFFOLD. The two research extensions (coordinated, timed) are flagged
with TODO(student) in src/attack_framework/learned_adversary.py; enabling them is
the point of the project.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "src/attack_framework")
sys.path.insert(0, "src/maddpg_clean")

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402
from learned_adversary import (  # noqa: E402
    AdversaryConfig, AdversaryTrainer, LearnedObservationAdversary,
)


def build_victim_and_env(runner: "StandaloneExperimentRunner", variant_name: str):
    """Load the named trained victim + an attack env, mirroring fgsm_probe()."""
    vcfg = next(v for v in runner.config["variants"] if v["name"] == variant_name)
    maddpg, _, _ = runner._make_variant(vcfg)
    runner._load_variant_checkpoint(maddpg, variant_name)
    # freeze the victim
    for ag in maddpg.agents:
        ag.actor.eval()
        for p in ag.actor.parameters():
            p.requires_grad_(False)
    hotspot = (runner.config.get("attack_eval", {}) or {}).get("hotspot")
    env = runner._make_attack_env(hotspot)
    trainable = env.engine.trainable_host_indices
    obs_dim = len(env.engine.get_state(env.engine.get_all_hosts()[trainable[0]]))
    return maddpg, env, trainable, obs_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="CC-Simple")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--load", type=float, default=2.0)
    ap.add_argument("--failures", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=0.30)
    ap.add_argument("--out", default="host_data/results/learned_adv")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--adv-ckpt", default=None)
    ap.add_argument("--coordinate", action="store_true",
                    help="TODO(student A): joint multi-agent perturbation")
    ap.add_argument("--timing-budget", type=float, default=None,
                    help="TODO(student B): fraction of steps the attacker may act")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runner = StandaloneExperimentRunner(config_path=args.config,
                                        results_dir=args.out)
    maddpg, env, trainable, obs_dim = build_victim_and_env(runner, args.variant)
    cfg = AdversaryConfig(epsilon=args.epsilon,
                          coordinate=args.coordinate,
                          timing_budget=args.timing_budget)

    if args.eval_only:
        assert args.adv_ckpt, "--eval-only needs --adv-ckpt"
        adv = LearnedObservationAdversary(obs_dim, cfg).load(args.adv_ckpt)
        # Drop the learned adversary into the runner's attack loop and reuse the
        # existing paired clean/attacked scoring. The runner sets .epsilon per case;
        # attack_type='learned' routes to LearnedObservationAdversary.perturb().
        runner.attack_framework = adv
        clean = runner._attack_episodes(maddpg, env, 15, args.steps, attack=False,
                                        offered_load_factor=args.load,
                                        n_link_failures=args.failures)
        attacked = runner._attack_episodes(maddpg, env, 15, args.steps, attack=True,
                                           attack_type="learned", epsilon=args.epsilon,
                                           offered_load_factor=args.load,
                                           n_link_failures=args.failures,
                                           measure_flips=True)
        result = {
            "variant": args.variant, "epsilon": args.epsilon,
            "clean_pdr": clean["mean_end_to_end_pdr"],
            "attacked_pdr": attacked["mean_end_to_end_pdr"],
            "drop_pp": clean["mean_end_to_end_pdr"] - attacked["mean_end_to_end_pdr"],
            "action_flip_rate": attacked.get("action_flip_rate"),
        }
        print(json.dumps(result, indent=2))
        json.dump(result, open(os.path.join(args.out, "learned_adv_eval.json"), "w"),
                  indent=2)
        return

    trainer = AdversaryTrainer(
        victim=maddpg, env=env, trainable_indices=trainable, obs_dim=obs_dim,
        cfg=cfg, build_full_actions=runner._build_full_actions,
    )
    history = trainer.train(n_episodes=args.episodes, t_per_ep=args.steps,
                            offered_load_factor=args.load,
                            n_link_failures=args.failures)
    ckpt = os.path.join(args.out, "adversary.pt")
    trainer.save(ckpt)
    json.dump(history, open(os.path.join(args.out, "train_history.json"), "w"))
    print(f"saved adversary -> {ckpt}")
    print("next: re-run with --eval-only --adv-ckpt", ckpt,
          "to score it against the damage ceiling / random control.")


if __name__ == "__main__":
    main()
