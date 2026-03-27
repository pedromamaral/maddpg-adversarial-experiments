"""
Complete Standalone Experiment Runner
Everything needed to run MADDPG adversarial robustness experiments
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add our clean implementations to path
sys.path.insert(0, 'src/maddpg_clean')
sys.path.insert(0, 'src/attack_framework')

from maddpg_implementation import MADDPG
from network_environment import NetworkEngine, NetworkEnv
from improved_fgsm_attack import FGSMAttackFramework, ThesisVisualizationSuite

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandaloneExperimentRunner:
    """
    Complete standalone experiment runner
    No external code dependencies - everything is self-contained
    """
    
    def __init__(self, config_path: str, gpu_id: int = 0):
        self.config = self.load_config(config_path)
        self.gpu_id = gpu_id
        self.results_dir = f"data/results/standalone_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize attack framework
        self.attack_framework = FGSMAttackFramework()
        
        logger.info("✅ Standalone experiment runner initialized")
        logger.info(f"📁 Results directory: {self.results_dir}")
        
        # Set device
        if gpu_id >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    def load_config(self, config_path: str) -> Dict:
        """Load experiment configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add default values if missing
        if 'training' not in config:
            config['training'] = {
                'epochs': 200,
                'episodes_per_epoch': 100,
                'timesteps_per_episode': 256
            }
        
        return config
    
    def create_maddpg_variant(self, variant_config: Dict) -> Tuple[MADDPG, NetworkEngine, NetworkEnv]:
        """Create MADDPG variant with clean implementation"""
        
        logger.info(f"🏗️  Creating {variant_config['name']} variant")
        
        # Create network environment
        network_engine = NetworkEngine(
            topology_type=self.config.get('topology', {}).get('type', 'service_provider'),
            n_nodes=variant_config['n_agents']
        )
        network_env = NetworkEnv(network_engine)
        
        # Create MADDPG with variant configuration
        maddpg = MADDPG(
            actor_dims=variant_config['actor_dims'],
            critic_dims=variant_config['critic_dims'],
            n_agents=variant_config['n_agents'],
            n_actions=variant_config['n_actions'],
            chkpt_dir=f"{self.results_dir}/models/{variant_config['name']}",
            critic_type=variant_config['critic_domain'],
            network_type=variant_config['neural_network'],
            use_gnn=variant_config.get('use_gnn', False)
        )
        
        logger.info(f"✅ {variant_config['name']} variant created successfully")
        return maddpg, network_engine, network_env
    
    def train_baseline_variant(self, variant_config: Dict) -> Tuple[str, List[float], List[float]]:
        """Train baseline MADDPG variant without attacks"""
        
        variant_name = variant_config['name']
        logger.info(f"🚀 Training baseline {variant_name}")
        
        # Create variant
        maddpg, network_engine, network_env = self.create_maddpg_variant(variant_config)
        
        # Training metrics
        episode_rewards = []
        episode_packet_losses = []
        
        # Training configuration
        epochs = self.config['training']['epochs']
        episodes_per_epoch = self.config['training']['episodes_per_epoch'] 
        timesteps_per_episode = self.config['training']['timesteps_per_episode']
        
        logger.info(f"📖 Training configuration: {epochs} epochs, {episodes_per_epoch} episodes/epoch")
        
        for epoch in range(epochs):
            epoch_rewards = []
            epoch_packet_losses = []
            
            for episode in range(episodes_per_epoch):
                # Reset environment
                states = network_env.reset()
                episode_reward = 0
                episode_packets_sent = 0
                episode_packets_dropped = 0
                
                for timestep in range(timesteps_per_episode):
                    # Choose actions (no attacks during training)
                    actions = maddpg.choose_action(states)
                    
                    # Environment step
                    next_states, rewards, info = network_env.step(actions)
                    
                    # Store transition and learn
                    done = [False] * len(states)  # Continuous task
                    maddpg.store_transition(states, actions, rewards, next_states, done)
                                        if timestep % 10 == 0:
                        maddpg.learn()
                    # Update states and metrics
                    states = next_states
                    episode_reward += sum(rewards)
                    episode_packets_sent += info.get('packets_sent', 0)
                    episode_packets_dropped += info.get('packets_dropped', 0)
                
                # Calculate episode metrics
                total_reward = episode_reward / len(maddpg.agents)  # Average per agent
                packet_loss_rate = (episode_packets_dropped / max(1, episode_packets_sent)) * 100
                
                epoch_rewards.append(total_reward)
                epoch_packet_losses.append(packet_loss_rate)
            
            episode_rewards.extend(epoch_rewards)
            episode_packet_losses.extend(epoch_packet_losses)
            
            # Log progress
            if epoch % 20 == 0 or epoch == epochs - 1:
                avg_reward = np.mean(epoch_rewards)
                avg_packet_loss = np.mean(epoch_packet_losses)
                logger.info(f"  Epoch {epoch:3d}: Reward={avg_reward:7.2f}, Packet Loss={avg_packet_loss:5.2f}%")
        
        # Save trained model
        model_path = f"{self.results_dir}/models/{variant_name}_baseline.pth"
        maddpg.save_checkpoint()
        
        logger.info(f"✅ {variant_name} training complete. Model saved to {model_path}")
        
        return model_path, episode_rewards, episode_packet_losses
    
    def evaluate_variant_robustness(self, variant_config: Dict, model_path: str) -> Dict:
        """Evaluate variant robustness under adversarial attacks"""
        
        variant_name = variant_config['name']
        logger.info(f"🎯 Evaluating {variant_name} robustness")
        
        # Load trained model
        maddpg, network_engine, network_env = self.create_maddpg_variant(variant_config)
        maddpg.load_checkpoint()
        
        attack_results = {}
        
        # Test each attack configuration
        for attack_config in self.config['attack_configs']:
            attack_type = attack_config['attack_type']
            epsilon = attack_config['epsilon']
            n_episodes = attack_config['evaluation_episodes']
            
            logger.info(f"  🔥 Testing {attack_type} attack with ε={epsilon}")
            
            # Configure attack framework
            self.attack_framework.epsilon = epsilon
            self.attack_framework.attack_type = attack_type
            
            clean_results = []
            attacked_results = []
            
            for episode in range(n_episodes):
                # Run clean episode
                clean_reward, clean_packet_loss = self.run_evaluation_episode(
                    maddpg, network_env, use_attack=False
                )
                clean_results.append({
                    'reward': clean_reward,
                    'packet_loss': clean_packet_loss
                })
                
                # Run attacked episode
                attacked_reward, attacked_packet_loss = self.run_evaluation_episode(
                    maddpg, network_env, use_attack=True
                )
                attacked_results.append({
                    'reward': attacked_reward,
                    'packet_loss': attacked_packet_loss
                })
            
            # Compute comparison metrics
            comparison_metrics = self.compute_attack_metrics(clean_results, attacked_results)
            
            # Store results
            attack_key = f"{attack_type}_eps{epsilon}"
            attack_results[attack_key] = {
                'attack_config': attack_config,
                'clean_results': clean_results,
                'attacked_results': attacked_results,
                'comparison_metrics': comparison_metrics
            }
            
            # Log results
            degradation = comparison_metrics['reward_degradation_percent']
            success_rate = comparison_metrics['attack_success_rate_percent']
            logger.info(f"    💥 Reward degradation: {degradation:.1f}%, Success rate: {success_rate:.1f}%")
        
        logger.info(f"✅ {variant_name} robustness evaluation complete")
        return attack_results
    
    def run_evaluation_episode(self, maddpg: MADDPG, network_env: NetworkEnv, 
                              use_attack: bool = False) -> Tuple[float, float]:
        """Run single evaluation episode with or without attack"""
        
        states = network_env.reset()
        episode_reward = 0
        episode_packets_sent = 0
        episode_packets_dropped = 0
        
        for timestep in range(256):  # Standard episode length
            if use_attack:
                # Apply adversarial attack to states
                attacked_states = []
                for agent_idx, state in enumerate(states):
                    attacked_state = self.attack_framework.generate_adversarial_state(
                        state=state,
                        agent_network=maddpg.agents[agent_idx],
                        network_engine=network_env.engine,  # Pass network engine
                        agent_index=agent_idx
                    )
                    attacked_states.append(attacked_state)
                states = attacked_states
            
            # Choose actions
            actions = maddpg.choose_action(states)
            
            # Environment step
            next_states, rewards, info = network_env.step(actions)
            
            # Update metrics
            states = next_states
            episode_reward += sum(rewards)
            episode_packets_sent += info.get('packets_sent', 0)
            episode_packets_dropped += info.get('packets_dropped', 0)
        
        # Calculate final metrics
        avg_reward = episode_reward / len(maddpg.agents)
        packet_loss_rate = (episode_packets_dropped / max(1, episode_packets_sent)) * 100
        
        return avg_reward, packet_loss_rate
    
    def compute_attack_metrics(self, clean_results: List[Dict], 
                              attacked_results: List[Dict]) -> Dict:
        """Compute comparative attack metrics"""
        
        clean_rewards = [r['reward'] for r in clean_results]
        attacked_rewards = [r['reward'] for r in attacked_results]
        clean_packet_losses = [r['packet_loss'] for r in clean_results]
        attacked_packet_losses = [r['packet_loss'] for r in attacked_results]
        
        # Reward degradation
        mean_clean_reward = np.mean(clean_rewards)
        mean_attacked_reward = np.mean(attacked_rewards)
        reward_degradation = ((mean_clean_reward - mean_attacked_reward) / 
                            abs(mean_clean_reward) * 100)
        
        # Packet loss increase
        packet_loss_increase = np.mean(attacked_packet_losses) - np.mean(clean_packet_losses)
        
        # Attack success rate (episodes with >10% reward degradation)
        successful_attacks = sum(
            1 for clean_r, attacked_r in zip(clean_rewards, attacked_rewards)
            if attacked_r < clean_r * 0.9
        )
        attack_success_rate = (successful_attacks / len(clean_results)) * 100
        
        # Performance variance change
        clean_std = np.std(clean_rewards)
        attacked_std = np.std(attacked_rewards)
        variance_change = ((attacked_std - clean_std) / abs(clean_std) * 100) if clean_std > 0 else 0
        
        # Robustness score (higher is better)
        robustness_score = max(0, 100 - abs(reward_degradation) - packet_loss_increase)
        
        return {
            'reward_degradation_percent': reward_degradation,
            'packet_loss_increase_percent': packet_loss_increase,
            'attack_success_rate_percent': attack_success_rate,
            'variance_change_percent': variance_change,
            'robustness_score': robustness_score
        }
    
    def run_complete_experiment(self) -> Dict:
        """Run complete standalone experiment"""
        
        logger.info("🚀 Starting Complete MADDPG Adversarial Robustness Experiment")
        logger.info("=" * 70)
        
        experiment_start = time.time()
        all_results = {}
        
        # Phase 1: Baseline Training
        logger.info("📚 PHASE 1: Baseline Training")
        logger.info("-" * 30)
        
        training_results = {}
        for variant_config in self.config['variants']:
            variant_name = variant_config['name']
            
            try:
                model_path, rewards, packet_losses = self.train_baseline_variant(variant_config)
                training_results[variant_name] = {
                    'model_path': model_path,
                    'training_rewards': rewards,
                    'training_packet_losses': packet_losses,
                    'final_avg_reward': np.mean(rewards[-50:]),  # Last 50 episodes
                    'final_avg_packet_loss': np.mean(packet_losses[-50:])
                }
                logger.info(f"✅ {variant_name} training successful")
                
            except Exception as e:
                logger.error(f"❌ {variant_name} training failed: {e}")
                continue
        
        # Phase 2: Robustness Evaluation
        logger.info("\n🛡️  PHASE 2: Robustness Evaluation")
        logger.info("-" * 35)
        
        for variant_config in self.config['variants']:
            variant_name = variant_config['name']
            
            if variant_name not in training_results:
                logger.warning(f"⏭️  Skipping {variant_name} (training failed)")
                continue
            
            try:
                model_path = training_results[variant_name]['model_path']
                attack_results = self.evaluate_variant_robustness(variant_config, model_path)
                
                all_results[variant_name] = {
                    'training_results': training_results[variant_name],
                    'attack_results': attack_results
                }
                
                # Log summary for this variant
                avg_robustness = np.mean([
                    data['comparison_metrics']['robustness_score']
                    for data in attack_results.values()
                ])
                logger.info(f"✅ {variant_name} evaluation complete (Avg robustness: {avg_robustness:.1f})")
                
            except Exception as e:
                logger.error(f"❌ {variant_name} evaluation failed: {e}")
                continue
        
        # Phase 3: Analysis and Visualization
        logger.info("\n📊 PHASE 3: Analysis and Visualization")
        logger.info("-" * 40)
        
        try:
            self.generate_complete_analysis(all_results)
            logger.info("✅ Analysis and visualization complete")
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
        
        # Save complete results
        results_file = f"{self.results_dir}/complete_experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        experiment_time = time.time() - experiment_start
        logger.info(f"\n🎉 EXPERIMENT COMPLETE!")
        logger.info(f"⏱️  Total time: {experiment_time/3600:.2f} hours")
        logger.info(f"📁 Results saved to: {self.results_dir}")
        
        return all_results
    
    def generate_complete_analysis(self, results: Dict):
        """Generate complete analysis including plots and summary"""
        
        # Transform results for visualization
        viz_data = self.transform_results_for_visualization(results)
        
        # Generate thesis plots
        viz_suite = ThesisVisualizationSuite(
            viz_data,
            save_path=f"{self.results_dir}/thesis_graphs"
        )
        viz_suite.generate_all_thesis_plots()
        
        # Generate statistical summary
        self.generate_comprehensive_summary(results)
        
        logger.info(f"📊 Thesis graphs saved to: {self.results_dir}/thesis_graphs/")
    
    def transform_results_for_visualization(self, results: Dict) -> Dict:
        """Transform results for visualization suite"""
        
        viz_data = {}
        
        for variant_name, variant_results in results.items():
            if 'attack_results' not in variant_results:
                continue
            
            viz_data[variant_name] = {}
            
            for attack_key, attack_data in variant_results['attack_results'].items():
                # Extract epsilon from attack key
                try:
                    epsilon = float(attack_key.split('_eps')[-1])
                    epsilon_key = f"epsilon_{epsilon}"
                    
                    clean_results = attack_data['clean_results']
                    attacked_results = attack_data['attacked_results']
                    
                    viz_data[variant_name][epsilon_key] = {
                        'clean': {
                            'mean_reward': np.mean([r['reward'] for r in clean_results]),
                            'mean_packet_loss': np.mean([r['packet_loss'] for r in clean_results])
                        },
                        'attacked': {
                            'mean_reward': np.mean([r['reward'] for r in attacked_results]),
                            'mean_packet_loss': np.mean([r['packet_loss'] for r in attacked_results])
                        },
                        'comparison': attack_data['comparison_metrics']
                    }
                except (ValueError, IndexError):
                    continue
        
        return viz_data
    
    def generate_comprehensive_summary(self, results: Dict):
        """Generate comprehensive experimental summary"""
        
        summary = {
            "experiment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "variants_tested": list(results.keys()),
                "total_variants": len(results),
                "framework": "standalone_clean_implementation"
            },
            "key_findings": {},
            "robustness_rankings": {},
            "architecture_analysis": {},
            "recommendations": {}
        }
        
        # Analyze robustness across variants
        variant_robustness = {}
        variant_details = {}
        
        for variant_name, variant_results in results.items():
            if 'attack_results' in variant_results:
                # Calculate average robustness
                robustness_scores = []
                reward_degradations = []
                attack_success_rates = []
                
                for attack_data in variant_results['attack_results'].values():
                    metrics = attack_data['comparison_metrics']
                    robustness_scores.append(metrics['robustness_score'])
                    reward_degradations.append(metrics['reward_degradation_percent'])
                    attack_success_rates.append(metrics['attack_success_rate_percent'])
                
                avg_robustness = np.mean(robustness_scores)
                avg_degradation = np.mean(reward_degradations)
                avg_success_rate = np.mean(attack_success_rates)
                
                variant_robustness[variant_name] = avg_robustness
                variant_details[variant_name] = {
                    'avg_robustness_score': avg_robustness,
                    'avg_reward_degradation': avg_degradation,
                    'avg_attack_success_rate': avg_success_rate,
                    'training_final_reward': variant_results['training_results'].get('final_avg_reward', 0),
                    'training_final_packet_loss': variant_results['training_results'].get('final_avg_packet_loss', 0)
                }
        
        # Robustness rankings
        if variant_robustness:
            ranked_variants = sorted(variant_robustness.items(), key=lambda x: x[1], reverse=True)
            
            summary['key_findings'] = {
                "most_robust_variant": {
                    "name": ranked_variants[0][0],
                    "robustness_score": ranked_variants[0][1]
                },
                "least_robust_variant": {
                    "name": ranked_variants[-1][0],
                    "robustness_score": ranked_variants[-1][1]
                },
                "robustness_range": ranked_variants[0][1] - ranked_variants[-1][1]
            }
            
            summary['robustness_rankings'] = {name: score for name, score in ranked_variants}
            summary['variant_details'] = variant_details
        
        # Architecture analysis
        summary['architecture_analysis'] = self.analyze_architectural_impact(variant_robustness)
        
        # Generate recommendations
        summary['recommendations'] = self.generate_recommendations(variant_details)
        
        # Save summary
        summary_file = f"{self.results_dir}/comprehensive_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print key findings to console
        logger.info("🔍 KEY EXPERIMENTAL FINDINGS:")
        if 'most_robust_variant' in summary['key_findings']:
            most_robust = summary['key_findings']['most_robust_variant']
            least_robust = summary['key_findings']['least_robust_variant']
            
            logger.info(f"  🥇 Most Robust: {most_robust['name']} (Score: {most_robust['robustness_score']:.1f})")
            logger.info(f"  🥉 Least Robust: {least_robust['name']} (Score: {least_robust['robustness_score']:.1f})")
            
            arch_analysis = summary['architecture_analysis']
            if 'gnn_impact' in arch_analysis:
                gnn_impact = arch_analysis['gnn_impact']
                logger.info(f"  🤖 GNN Impact: {gnn_impact['improvement_percentage']:.1f}% improvement")
        
        logger.info(f"📄 Comprehensive summary saved to: {summary_file}")
    
    def analyze_architectural_impact(self, variant_robustness: Dict) -> Dict:
        """Analyze impact of different architectural choices"""
        
        analysis = {}
        
        # GNN impact analysis
        gnn_variants = {k: v for k, v in variant_robustness.items() if 'GNN' in k}
        non_gnn_variants = {k: v for k, v in variant_robustness.items() if 'GNN' not in k}
        
        if gnn_variants and non_gnn_variants:
            gnn_avg = np.mean(list(gnn_variants.values()))
            non_gnn_avg = np.mean(list(non_gnn_variants.values()))
            improvement = gnn_avg - non_gnn_avg
            
            analysis['gnn_impact'] = {
                "gnn_average_robustness": gnn_avg,
                "non_gnn_average_robustness": non_gnn_avg,
                "improvement_percentage": improvement,
                "conclusion": "GNN improves robustness" if improvement > 0 else "GNN reduces robustness"
            }
        
        # Critic type analysis
        central_critics = {k: v for k, v in variant_robustness.items() if k.startswith('CC')}
        local_critics = {k: v for k, v in variant_robustness.items() if k.startswith('LC')}
        
        if central_critics and local_critics:
            cc_avg = np.mean(list(central_critics.values()))
            lc_avg = np.mean(list(local_critics.values()))
            
            analysis['critic_comparison'] = {
                "central_critic_avg": cc_avg,
                "local_critic_avg": lc_avg,
                "conclusion": "Local critics more robust" if lc_avg > cc_avg else "Central critics more robust"
            }
        
        # Network type analysis
        simple_networks = {k: v for k, v in variant_robustness.items() if 'Simple' in k}
        duelling_networks = {k: v for k, v in variant_robustness.items() if 'Duelling' in k}
        
        if simple_networks and duelling_networks:
            simple_avg = np.mean(list(simple_networks.values()))
            duelling_avg = np.mean(list(duelling_networks.values()))
            
            analysis['network_comparison'] = {
                "simple_q_avg": simple_avg,
                "duelling_q_avg": duelling_avg,
                "conclusion": "Duelling networks more robust" if duelling_avg > simple_avg else "Simple networks more robust"
            }
        
        return analysis
    
    def generate_recommendations(self, variant_details: Dict) -> Dict:
        """Generate practical recommendations based on results"""
        
        if not variant_details:
            return {"error": "No variant details available for recommendations"}
        
        recommendations = {}
        
        # Find best overall variant
        best_variant = max(variant_details.items(), 
                          key=lambda x: x[1]['avg_robustness_score'])
        
        recommendations['best_overall_choice'] = {
            "variant": best_variant[0],
            "reasoning": f"Highest robustness score ({best_variant[1]['avg_robustness_score']:.1f}) with good training performance"
        }
        
        # Performance vs robustness trade-off
        balanced_variants = []
        for name, details in variant_details.items():
            training_reward = details.get('training_final_reward', 0)
            robustness_score = details['avg_robustness_score']
            
            # Simple scoring: balance training performance and robustness
            balanced_score = (training_reward / 1000) + robustness_score  # Normalize training reward
            balanced_variants.append((name, balanced_score, details))
        
        balanced_variants.sort(key=lambda x: x[1], reverse=True)
        
        recommendations['balanced_choice'] = {
            "variant": balanced_variants[0][0],
            "reasoning": "Best balance between training performance and adversarial robustness"
        }
        
        # Security-critical recommendation
        most_robust_name = max(variant_details.items(), 
                              key=lambda x: x[1]['avg_robustness_score'])[0]
        
        recommendations['security_critical_choice'] = {
            "variant": most_robust_name,
            "reasoning": "Highest adversarial robustness for security-critical applications"
        }
        
        return recommendations


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Standalone MADDPG Adversarial Robustness Experiments')
    parser.add_argument('--config', type=str, default='experiment_config.json',
                       help='Experiment configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (reduced epochs and episodes)')
    
    args = parser.parse_args()
    
    print("🔬 STANDALONE MADDPG ADVERSARIAL ROBUSTNESS EXPERIMENT")
    print("=" * 60)
    print("✅ Self-contained implementation - no external code dependencies")
    print("✅ Clean MADDPG training implementation")
    print("✅ Corrected FGSM attack framework")
    print("✅ Thesis-quality analysis and visualization")
    print("")
    
    # Quick mode adjustments
    if args.quick:
        print("⚡ Quick test mode enabled")
        # Will modify config to reduce training time
    
    # Initialize and run experiment
    try:
        runner = StandaloneExperimentRunner(args.config, args.gpu)
        
        # Modify config for quick mode
        if args.quick:
            runner.config['training']['epochs'] = 10
            runner.config['training']['episodes_per_epoch'] = 10
            runner.config['training']['timesteps_per_episode'] = 50          
            for attack_config in runner.config['attack_configs']:
                attack_config['evaluation_episodes'] = 10
            logger.info("⚡ Config modified for quick testing")
        
        # Run complete experiment
        results = runner.run_complete_experiment()
        
        print("\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("📁 Check the results directory for:")
        print("   • Trained model weights")
        print("   • Experimental data (JSON)")
        print("   • Thesis-quality plots (PNG)")
        print("   • Comprehensive analysis summary")
        
    except Exception as e:
        logger.error(f"💥 Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
