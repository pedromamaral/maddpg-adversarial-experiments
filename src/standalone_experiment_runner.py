"""
Standalone Experiment Runner
Three sequential phases:
  Phase 1  — training  (all 6 variants, hop-by-hop mode)
  Phase 2  — Paper-1 evaluation (architecture comparison + OSPF baseline)
  Phase 3  — Paper-2 evaluation (FGSM adversarial attack study)
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, 'src/maddpg_clean')
sys.path.insert(0, 'src/attack_framework')

from maddpg_implementation import MADDPG
from network_environment import NetworkEngine, NetworkEnv
from improved_fgsm_attack import FGSMAttackFramework, ThesisVisualizationSuite

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class StandaloneExperimentRunner:

    def __init__(self, config_path: str, gpu_id: int = 0,
                 results_dir: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.gpu_id = gpu_id
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = results_dir or f"data/results/exp_{ts}"
        os.makedirs(self.results_dir, exist_ok=True)
        self.attack_framework = FGSMAttackFramework()
        if gpu_id >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"Results → {self.results_dir}")

    # ── config ────────────────────────────────────────────────────────────────

    def _load_config(self, path: str) -> Dict:
        with open(path) as f:
            cfg = json.load(f)
        cfg.setdefault('training', {'epochs': 200, 'episodes_per_epoch': 5,
                                    'timesteps_per_episode': 256})
        return cfg

    # ── factory ───────────────────────────────────────────────────────────────

    def _make_variant(self, vcfg: Dict) -> Tuple[MADDPG, NetworkEngine, NetworkEnv]:
        engine = NetworkEngine(
            topology_type=self.config.get('topology', {}).get('type', 'service_provider'),
            n_nodes=vcfg['n_agents'],
        )
        env = NetworkEnv(engine)
        maddpg = MADDPG(
            actor_dims=vcfg['actor_dims'],
            critic_dims=vcfg['critic_dims'],
            n_agents=vcfg['n_agents'],
            n_actions=vcfg['n_actions'],
            chkpt_dir=f"{self.results_dir}/models/{vcfg['name']}",
            critic_type=vcfg['critic_domain'],
            network_type=vcfg['neural_network'],
            fc1=vcfg.get('fc1', 256),
            fc2=vcfg.get('fc2', 128),
            alpha=vcfg.get('alpha', 0.001),
            beta=vcfg.get('beta', 0.001),
            use_gnn=vcfg.get('use_gnn', False),
        )
        return maddpg, engine, env

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Training
    # ═══════════════════════════════════════════════════════════════════════════

    def train_variant(self, vcfg: Dict) -> Dict:
        name = vcfg['name']
        logger.info(f"[TRAIN] {name} — start")
        maddpg, engine, env = self._make_variant(vcfg)

        cfg_t      = self.config['training']
        epochs     = cfg_t['epochs']
        eps_per_ep = cfg_t['episodes_per_epoch']
        t_per_ep   = cfg_t['timesteps_per_episode']

        all_rewards, all_losses = [], []

        for epoch in range(epochs):
            ep_rewards, ep_losses = [], []
            for _ in range(eps_per_ep):
                states = env.reset()
                ep_r = ep_sent = ep_dropped = 0
                for t in range(t_per_ep):
                    actions = maddpg.choose_action(states)
                    next_states, rewards, info = env.step(actions)
                    done = [t == t_per_ep - 1] * len(states)
                    maddpg.store_transition(states, actions, rewards, next_states, done)
                    freq = 10 if epoch < 20 else 25
                    if t % freq == 0:
                        maddpg.learn()
                    states = next_states
                    ep_r      += sum(rewards)
                    ep_sent   += info.get('packets_sent', 0)
                    ep_dropped += info.get('packets_dropped', 0)
                ep_rewards.append(ep_r / len(maddpg.agents))
                ep_losses.append(ep_dropped / max(1, ep_sent) * 100)

            all_rewards.extend(ep_rewards)
            all_losses.extend(ep_losses)

            if epoch % 20 == 0 or epoch == epochs - 1:
                logger.info(
                    f"[TRAIN] {name}  epoch {epoch:3d}  "
                    f"reward={np.mean(ep_rewards):7.2f}  "
                    f"pkt_loss={np.mean(ep_losses):5.2f}%"
                )

        maddpg.save_checkpoint()
        logger.info(f"[TRAIN] {name} — done")
        return {
            'name':           name,
            'model_dir':      f"{self.results_dir}/models/{name}",
            'rewards':        all_rewards,
            'pkt_losses':     all_losses,
            'final_reward':   float(np.mean(all_rewards[-50:])),
            'final_pkt_loss': float(np.mean(all_losses[-50:])),
        }

    def run_training(self) -> Dict:
        logger.info("══════ PHASE 1 — TRAINING ══════")
        results = {}
        for vcfg in self.config['variants']:
            try:
                results[vcfg['name']] = self.train_variant(vcfg)
            except Exception as exc:
                logger.error(f"[TRAIN] {vcfg['name']} FAILED: {exc}")
        self._save(results, 'phase1_training_results.json')
        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — MADDPG evaluation
    # ═══════════════════════════════════════════════════════════════════════════

    def evaluate_maddpg(self, training_results: Optional[Dict] = None) -> Dict:
        logger.info("══════ PHASE 2 — MADDPG EVALUATION ══════")
        if training_results is None:
            training_results = self._load('phase1_training_results.json')

        results  = {}
        eval_eps = self.config.get('paper1_eval', {}).get('evaluation_episodes', 30)
        t_per_ep = self.config['training']['timesteps_per_episode']

        # MADDPG variants
        for vcfg in self.config['variants']:
            name = vcfg['name']
            if name not in training_results:
                logger.warning(f"[P1] {name} — no trained model, skipping")
                continue
            logger.info(f"[P1] Evaluating {name}")
            maddpg, engine, env = self._make_variant(vcfg)
            maddpg.load_checkpoint()
            normal   = self._run_eval_episodes(maddpg, env, eval_eps, t_per_ep)
            failures = self._run_eval_episodes(maddpg, env, eval_eps, t_per_ep,
                                               n_link_failures=2)
            results[name] = {'normal': normal, 'dual_link_failure': failures}

        # OSPF baseline
        logger.info("[P1] OSPF baseline")
        engine = NetworkEngine(
            topology_type=self.config.get('topology', {}).get('type', 'service_provider'),
            n_nodes=65,
        )
        results['OSPF'] = {
            'normal':           self._run_ospf_episodes(engine, eval_eps, t_per_ep),
            'dual_link_failure': self._run_ospf_episodes(engine, eval_eps, t_per_ep,
                                                         n_link_failures=2),
        }

        self._save(results, 'phase2_maddpg_results.json')
        logger.info("[P1] evaluation complete")
        return results

    def _run_eval_episodes(self, maddpg: MADDPG, env: NetworkEnv,
                           n_eps: int, t_per_ep: int,
                           n_link_failures: int = 0) -> Dict:
        ep_rewards, ep_losses, ep_utils = [], [], []
        for _ in range(n_eps):
            states = env.reset()
            if n_link_failures:
                self._inject_failures(env.engine, n_link_failures)
            ep_r = ep_sent = ep_dropped = 0
            for t in range(t_per_ep):
                actions = maddpg.choose_action(states)
                next_states, rewards, info = env.step(actions)
                states = next_states
                ep_r      += sum(rewards)
                ep_sent   += info['packets_sent']
                ep_dropped += info['packets_dropped']
            ep_rewards.append(ep_r / len(maddpg.agents))
            ep_losses.append(ep_dropped / max(1, ep_sent) * 100)
            ep_utils.append(float(np.mean(env.engine.get_link_utilization_distribution())))
        return {
            'mean_reward':   float(np.mean(ep_rewards)),
            'std_reward':    float(np.std(ep_rewards)),
            'mean_pkt_loss': float(np.mean(ep_losses)),
            'std_pkt_loss':  float(np.std(ep_losses)),
            'mean_util':     float(np.mean(ep_utils)),
        }

    def _run_ospf_episodes(self, engine: NetworkEngine,
                           n_eps: int, t_per_ep: int,
                           n_link_failures: int = 0) -> Dict:
        ep_losses, ep_utils = [], []
        for _ in range(n_eps):
            engine.reset()
            if n_link_failures:
                self._inject_failures(engine, n_link_failures)
            ep_sent = ep_dropped = 0
            for _ in range(t_per_ep):
                info = engine.ospf_step()
                ep_sent    += info['packets_sent']
                ep_dropped += info['packets_dropped']
            ep_losses.append(ep_dropped / max(1, ep_sent) * 100)
            ep_utils.append(float(np.mean(engine.get_link_utilization_distribution())))
        return {
            'mean_pkt_loss': float(np.mean(ep_losses)),
            'std_pkt_loss':  float(np.std(ep_losses)),
            'mean_util':     float(np.mean(ep_utils)),
        }

    @staticmethod
    def _inject_failures(engine: NetworkEngine, n: int):
        edges = list(engine.topology.graph.edges())
        for idx in np.random.choice(len(edges), size=min(n, len(edges)), replace=False):
            u, v = edges[idx]
            if engine.topology.graph.has_edge(u, v):
                engine.topology.graph.remove_edge(u, v)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — FGSM atttack evaluation
    # ═══════════════════════════════════════════════════════════════════════════

    def evaluate_fgsm(self, training_results: Optional[Dict] = None) -> Dict:
        logger.info("══════ PHASE 3 — FGSM ATTACK EVALUATION ══════")
        if training_results is None:
            training_results = self._load('phase1_training_results.json')

        all_results = {}

        for vcfg in self.config['variants']:
            name = vcfg['name']
            if name not in training_results:
                logger.warning(f"[P2] {name} — no trained model, skipping")
                continue
            logger.info(f"[P2] Attacking {name}")
            maddpg, engine, env = self._make_variant(vcfg)
            maddpg.load_checkpoint()
            variant_results = {}

            # Standard FGSM sweep
            for acfg in self.config['attack_configs']:
                atype    = acfg['attack_type']
                eps      = acfg['epsilon']
                n_eps    = acfg['evaluation_episodes']
                t_per_ep = self.config['training']['timesteps_per_episode']
                key      = f"{atype}_eps{eps}"
                logger.info(f"[P2]   {name}  {key}")
                clean    = self._attack_episodes(maddpg, env, n_eps, t_per_ep, attack=False)
                attacked = self._attack_episodes(maddpg, env, n_eps, t_per_ep,
                                                 attack=True, attack_type=atype,
                                                 epsilon=eps)
                variant_results[key] = {
                    'clean': clean, 'attacked': attacked,
                    'metrics': self._compare(clean, attacked),
                }

            # Attack-surface analysis (core vs dist vs access)
            logger.info(f"[P2]   {name}  attack surface analysis")
            n_eps    = self.config['attack_configs'][-1]['evaluation_episodes']
            t_per_ep = self.config['training']['timesteps_per_episode']
            variant_results['surface'] = self._attack_surface_analysis(
                maddpg, env, n_eps=n_eps, t_per_ep=t_per_ep, epsilon=0.10)

            # GNN embedding attack (GNN variants only)
            if vcfg.get('use_gnn', False):
                logger.info(f"[P2]   {name}  GNN embedding attack")
                variant_results['gnn_embedding_attack'] = self._attack_episodes(
                    maddpg, env, n_eps=n_eps, t_per_ep=t_per_ep,
                    attack=True, attack_type='packet_loss', epsilon=0.10)

            all_results[name] = variant_results

        self._save(all_results, 'phase3_fgsm_results.json')
        logger.info("[P2] evaluation complete")
        return all_results

    def _attack_episodes(self, maddpg: MADDPG, env: NetworkEnv,
                         n_eps: int, t_per_ep: int,
                         attack: bool = False,
                         attack_type: str = 'packet_loss',
                         epsilon: float = 0.05,
                         targeted_tiers: Optional[List[str]] = None) -> Dict:
        ep_rewards, ep_losses = [], []
        hosts = env.engine.get_all_hosts()

        for _ in range(n_eps):
            states = env.reset()
            ep_r = ep_sent = ep_dropped = 0
            for _ in range(t_per_ep):
                if attack:
                    adv = []
                    for idx, (host, s) in enumerate(zip(hosts, states)):
                        if targeted_tiers and env.engine.get_tier(host) not in targeted_tiers:
                            adv.append(s)
                        else:
                            adv.append(self.attack_framework.generate_adversarial_state(
                                state=s,
                                agent_network=maddpg.agents[idx],
                                network_engine=env.engine,
                                agent_index=idx,
                            ))
                    states = adv
                actions = maddpg.choose_action(states)
                next_states, rewards, info = env.step(actions)
                states = next_states
                ep_r      += sum(rewards)
                ep_sent   += info['packets_sent']
                ep_dropped += info['packets_dropped']
            ep_rewards.append(ep_r / len(maddpg.agents))
            ep_losses.append(ep_dropped / max(1, ep_sent) * 100)

        return {
            'mean_reward':   float(np.mean(ep_rewards)),
            'std_reward':    float(np.std(ep_rewards)),
            'mean_pkt_loss': float(np.mean(ep_losses)),
            'std_pkt_loss':  float(np.std(ep_losses)),
        }

    def _attack_surface_analysis(self, maddpg: MADDPG, env: NetworkEnv,
                                  n_eps: int, t_per_ep: int,
                                  epsilon: float = 0.10) -> Dict:
        clean  = self._attack_episodes(maddpg, env, n_eps, t_per_ep, attack=False)
        core_r = self._attack_episodes(maddpg, env, n_eps, t_per_ep, attack=True,
                                       attack_type='packet_loss', epsilon=epsilon,
                                       targeted_tiers=['core'])
        dist_r = self._attack_episodes(maddpg, env, n_eps, t_per_ep, attack=True,
                                       attack_type='packet_loss', epsilon=epsilon,
                                       targeted_tiers=['dist'])
        acc_r  = self._attack_episodes(maddpg, env, n_eps, t_per_ep, attack=True,
                                       attack_type='packet_loss', epsilon=epsilon,
                                       targeted_tiers=['access'])
        return {
            'clean':           clean,
            'core_attacked':   {'results': core_r, 'metrics': self._compare(clean, core_r)},
            'dist_attacked':   {'results': dist_r, 'metrics': self._compare(clean, dist_r)},
            'access_attacked': {'results': acc_r,  'metrics': self._compare(clean, acc_r)},
            'n_core_agents':   len(env.engine.topology.core_nodes),
            'n_dist_agents':   len(env.engine.topology.dist_nodes),
            'n_access_agents': len(env.engine.topology.access_nodes),
        }

    @staticmethod
    def _compare(clean: Dict, attacked: Dict) -> Dict:
        cr, ar = clean['mean_reward'], attacked['mean_reward']
        return {
            'reward_degradation_pct':  float((cr - ar) / abs(cr) * 100 if cr != 0 else 0),
            'pkt_loss_increase_pct':   float(attacked['mean_pkt_loss'] - clean['mean_pkt_loss']),
        }

    # ── full pipeline ─────────────────────────────────────────────────────────

    def run_all(self):
        t0 = time.time()
        tr = self.run_training()
        self.evaluate_maddpg(tr)
        self.evaluate_fgsm(tr)
        logger.info(f"All phases done in {(time.time()-t0)/3600:.2f} h  →  {self.results_dir}")

    # ── persistence ───────────────────────────────────────────────────────────

    def _save(self, obj, filename: str):
        path = os.path.join(self.results_dir, filename)
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2, default=str)
        logger.info(f"Saved {path}")

    def _load(self, filename: str) -> Dict:
        path = os.path.join(self.results_dir, filename)
        with open(path) as f:
            return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',      default='experiment_config.json')
    ap.add_argument('--gpu',         type=int, default=0)
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--phase', choices=['all', 'train', 'paper1', 'paper2'], default='all')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    runner = StandaloneExperimentRunner(args.config, args.gpu, args.results_dir)

    if args.quick:
        runner.config['training'].update(
            epochs=1, episodes_per_epoch=1, timesteps_per_episode=10)
        for a in runner.config['attack_configs']:
            a['evaluation_episodes'] = 2
        runner.config.setdefault('paper1_eval', {})['evaluation_episodes'] = 2
        logger.info("Quick mode active")

    if   args.phase == 'all':    runner.run_all()
    elif args.phase == 'train':  runner.run_training()
    elif args.phase == 'paper1': runner.evaluate_paper1()
    elif args.phase == 'paper2': runner.evaluate_paper2()


if __name__ == '__main__':
    main()