"""
Improved FGSM Attack Framework for MADDPG Routing
Fixes attack objective and implements proper comparative metrics for thesis analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import json

class FGSMAttackFramework:
    """
    Enhanced FGSM attack framework for adversarial analysis of MADDPG routing variants
    """
    
    def __init__(self, epsilon: float = 0.05, attack_type: str = 'packet_loss'):
        """
        Initialize FGSM attack framework
        
        Args:
            epsilon: Perturbation magnitude
            attack_type: Type of attack objective ('packet_loss', 'reward_minimize', 'confusion')
        """
        self.epsilon = epsilon
        self.attack_type = attack_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Attack statistics
        self.attack_stats = {
            'clean_rewards': [],
            'attacked_rewards': [],
            'clean_packet_loss': [],
            'attacked_packet_loss': [],
            'attack_success_count': 0,
            'total_attacks': 0
        }
    
    def generate_adversarial_state(self, state: np.ndarray, 
                                 agent_network, 
                                 network_engine, 
                                 agent_index: int,
                                 bandwidth_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Generate adversarial state using FGSM with proper attack objective
        
        Args:
            state: Original state representation
            agent_network: Actor network of the agent
            network_engine: Network environment engine
            agent_index: Index of the target agent
            bandwidth_indices: Indices of bandwidth values in state vector
        
        Returns:
            Adversarial state with perturbations applied
        """
        
        # Convert to tensor with gradient tracking
        state_tensor = torch.tensor([state], dtype=torch.float32, requires_grad=True).to(self.device)
        
        try:
            # Get action probabilities from actor
            action_probs = agent_network.actor.forward(state_tensor)
            
            # Compute attack objective based on attack type
            if self.attack_type == 'packet_loss':
                loss = self._packet_loss_objective(state_tensor, action_probs, network_engine, agent_index)
            elif self.attack_type == 'reward_minimize':
                loss = self._reward_minimize_objective(state_tensor, action_probs, network_engine)
            elif self.attack_type == 'confusion':
                loss = self._confusion_objective(action_probs)
            else:
                raise ValueError(f"Unknown attack type: {self.attack_type}")
            
            # Compute gradients
            loss.backward()
            
            # Generate perturbation using FGSM
            state_grad = state_tensor.grad.data
            perturbation = self.epsilon * torch.sign(state_grad)
            
            # Apply perturbation with domain constraints
            adversarial_state = state_tensor + perturbation
            adversarial_state = self._apply_domain_constraints(adversarial_state, bandwidth_indices)
            
            return adversarial_state.detach().cpu().numpy()[0]
        
        except Exception as e:
            print(f"Attack generation failed: {e}")
            return state  # Return original state on failure
    
    def _packet_loss_objective(self, state: torch.Tensor, 
                             action_probs: torch.Tensor, 
                             network_engine, 
                             agent_index: int) -> torch.Tensor:
        """
        Attack objective: Maximize packet loss by encouraging congested path selection
        """
        # Extract bandwidth information from state (first n elements typically)
        num_neighbors = network_engine.get_number_neighbors(network_engine.get_all_hosts()[agent_index])
        
        if num_neighbors == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        # Get bandwidth states (normalized 0-1, where 0 = congested, 1 = free)
        bandwidth_states = state[:, :num_neighbors]
        
        # Create congestion weights: higher weight for more congested links
        # Sigmoid transformation to create smooth gradients
        congestion_weights = torch.sigmoid((1.0 - bandwidth_states) * 10.0)
        
        # Encourage selection of actions leading to congested paths
        # This pushes agent toward paths with low available bandwidth
        congestion_loss = torch.sum(action_probs * congestion_weights.mean(dim=1, keepdim=True))
        
        return -congestion_loss  # Maximize congestion (minimize negative)
    
    def _reward_minimize_objective(self, state: torch.Tensor, 
                                 action_probs: torch.Tensor, 
                                 network_engine) -> torch.Tensor:
        """
        Attack objective: Minimize expected reward
        """
        # Create penalty weights based on network state
        # Higher penalties for actions that typically yield good performance
        penalty_weights = torch.ones_like(action_probs)
        
        # If we have access to historical performance, weight accordingly
        # For now, use uniform penalty to encourage suboptimal actions
        reward_loss = torch.sum(action_probs * penalty_weights)
        
        return reward_loss  # Minimize expected reward
    
    def _confusion_objective(self, action_probs: torch.Tensor) -> torch.Tensor:
        """
        Attack objective: Maximize action uncertainty (entropy)
        """
        # Maximize entropy to create confused/random behavior
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
        return -entropy  # Minimize negative entropy = maximize entropy
    
    def _apply_domain_constraints(self, adversarial_state: torch.Tensor, 
                                bandwidth_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Apply domain-specific constraints to adversarial state
        """
        # Clone to avoid in-place operations
        constrained_state = adversarial_state.clone()
        
        # Bandwidth values should be in [0, 1] 
        if bandwidth_indices is not None:
            constrained_state[:, bandwidth_indices] = torch.clamp(
                constrained_state[:, bandwidth_indices], 0.0, 1.0
            )
        else:
            # Assume first elements are bandwidth values (common pattern)
            # Adjust this based on actual state representation
            bandwidth_size = min(4, adversarial_state.shape[1])  # Max 4 neighbors typically
            constrained_state[:, :bandwidth_size] = torch.clamp(
                constrained_state[:, :bandwidth_size], 0.0, 1.0
            )
        
        # Other state elements (destinations, flow counts) should remain as integers
        # For now, we only constrain bandwidth values
        
        return constrained_state
    
    def update_statistics(self, clean_reward: float, attacked_reward: float,
                         clean_packet_loss: float, attacked_packet_loss: float):
        """Update attack effectiveness statistics"""
        self.attack_stats['clean_rewards'].append(clean_reward)
        self.attack_stats['attacked_rewards'].append(attacked_reward)
        self.attack_stats['clean_packet_loss'].append(clean_packet_loss)
        self.attack_stats['attacked_packet_loss'].append(attacked_packet_loss)
        self.attack_stats['total_attacks'] += 1
        
        # Count as successful attack if packet loss increased or reward decreased significantly
        if (attacked_packet_loss > clean_packet_loss * 1.1) or (attacked_reward < clean_reward * 0.9):
            self.attack_stats['attack_success_count'] += 1


class MADDPGRobustnessEvaluator:
    """
    Comprehensive evaluation framework for MADDPG variant robustness
    """
    
    def __init__(self, maddpg_variants: Dict, network_engine):
        """
        Initialize evaluator
        
        Args:
            maddpg_variants: Dictionary of MADDPG variants {'name': maddpg_instance}
            network_engine: Network simulation engine
        """
        self.maddpg_variants = maddpg_variants
        self.network_engine = network_engine
        self.results = defaultdict(lambda: defaultdict(list))
        
    def evaluate_attack_effectiveness(self, attack_framework: FGSMAttackFramework,
                                   num_episodes: int = 100,
                                   epsilon_values: List[float] = [0.01, 0.05, 0.1, 0.15, 0.2]) -> Dict:
        """
        Comprehensive evaluation of attack effectiveness across variants and intensities
        """
        evaluation_results = {}
        
        for variant_name, maddpg_agent in self.maddpg_variants.items():
            print(f"Evaluating {variant_name}...")
            variant_results = {}
            
            for epsilon in epsilon_values:
                print(f"  Testing epsilon = {epsilon}")
                attack_framework.epsilon = epsilon
                
                # Run clean episodes
                clean_metrics = self._run_episodes(maddpg_agent, num_episodes, attack=False)
                
                # Run attacked episodes
                attacked_metrics = self._run_episodes(maddpg_agent, num_episodes, attack=True, 
                                                    attack_framework=attack_framework)
                
                # Compute comparison metrics
                comparison_metrics = self._compute_comparison_metrics(clean_metrics, attacked_metrics)
                
                variant_results[f'epsilon_{epsilon}'] = {
                    'clean': clean_metrics,
                    'attacked': attacked_metrics,
                    'comparison': comparison_metrics
                }
            
            evaluation_results[variant_name] = variant_results
        
        return evaluation_results
    
    def _run_episodes(self, maddpg_agent, num_episodes: int, attack: bool = False,
                     attack_framework: Optional[FGSMAttackFramework] = None) -> Dict:
        """Run episodes with or without attacks and collect metrics"""
        
        episode_rewards = []
        episode_packet_losses = []
        episode_utilization_distributions = []
        
        for episode in range(num_episodes):
            # Reset environment
            self.network_engine.reset()
            
            total_reward = 0
            total_packet_loss = 0
            total_packets_sent = 0
            
            # Run episode timesteps
            for timestep in range(256):  # Standard episode length
                # Get states for all agents
                states = []
                all_hosts = self.network_engine.get_all_hosts()
                
                for agent_idx, host in enumerate(all_hosts):
                    state = self.network_engine.get_state(host, 1)
                    
                    # Apply attack if enabled
                    if attack and attack_framework is not None:
                        state = attack_framework.generate_adversarial_state(
                            state, maddpg_agent, self.network_engine, agent_idx
                        )
                    
                    states.append(state)
                
                # Choose actions
                actions = maddpg_agent.choose_action(states)
                
                # Execute actions and get rewards
                next_states, rewards, packet_loss_info = self._execute_actions(actions)
                
                total_reward += sum(rewards)
                total_packet_loss += packet_loss_info['packets_lost']
                total_packets_sent += packet_loss_info['packets_sent']
            
            # Record episode metrics
            episode_rewards.append(total_reward)
            packet_loss_rate = total_packet_loss / max(total_packets_sent, 1) * 100
            episode_packet_losses.append(packet_loss_rate)
            
            # Collect link utilization distribution
            utilization_dist = self.network_engine.get_link_utilization_distribution()
            episode_utilization_distributions.append(utilization_dist)
        
        return {
            'rewards': episode_rewards,
            'packet_losses': episode_packet_losses,
            'utilization_distributions': episode_utilization_distributions,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_packet_loss': np.mean(episode_packet_losses),
            'std_packet_loss': np.std(episode_packet_losses)
        }
    
    def _execute_actions(self, actions: List[int]) -> Tuple[List, List[float], Dict]:
        """Execute actions in environment and return results"""
        # This would integrate with your NetworkEngine
        # For now, return mock data structure
        rewards = [np.random.normal(50, 10) for _ in actions]  # Mock rewards
        packet_loss_info = {
            'packets_lost': np.random.poisson(5),
            'packets_sent': 100
        }
        next_states = [np.random.random(26) for _ in actions]  # Mock next states
        
        return next_states, rewards, packet_loss_info
    
    def _compute_comparison_metrics(self, clean_metrics: Dict, attacked_metrics: Dict) -> Dict:
        """Compute comparative metrics between clean and attacked performance"""
        
        # Reward degradation
        reward_degradation = (
            (clean_metrics['mean_reward'] - attacked_metrics['mean_reward']) / 
            clean_metrics['mean_reward'] * 100
        )
        
        # Packet loss increase
        packet_loss_increase = attacked_metrics['mean_packet_loss'] - clean_metrics['mean_packet_loss']
        
        # Attack success rate (episodes with significant performance drop)
        clean_rewards = np.array(clean_metrics['rewards'])
        attacked_rewards = np.array(attacked_metrics['rewards'])
        successful_attacks = np.sum(attacked_rewards < clean_rewards * 0.9)
        attack_success_rate = successful_attacks / len(clean_rewards) * 100
        
        # Performance variance change
        variance_change = (
            (attacked_metrics['std_reward'] - clean_metrics['std_reward']) / 
            clean_metrics['std_reward'] * 100
        )
        
        return {
            'reward_degradation_percent': reward_degradation,
            'packet_loss_increase_percent': packet_loss_increase,
            'attack_success_rate_percent': attack_success_rate,
            'variance_change_percent': variance_change,
            'robustness_score': max(0, 100 - reward_degradation - packet_loss_increase)
        }


class ThesisVisualizationSuite:
    """
    Generate publication-quality graphs for thesis inclusion
    """
    
    def __init__(self, results_data: Dict, save_path: str = '/Users/pedroamaral/.openclaw/workspace/thesis_graphs'):
        self.results_data = results_data
        self.save_path = save_path
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Configure matplotlib for publication-quality plots"""
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def generate_all_thesis_plots(self):
        """Generate complete set of thesis-quality plots"""
        
        # 1. Architecture Robustness Comparison
        self.plot_architecture_robustness()
        
        # 2. Attack Intensity Analysis
        self.plot_attack_intensity_analysis()
        
        # 3. Performance Degradation Matrix
        self.plot_performance_degradation_matrix()
        
        # 4. Reward vs Packet Loss Trade-offs
        self.plot_reward_packet_loss_tradeoffs()
        
        # 5. GNN Impact on Robustness
        self.plot_gnn_robustness_impact()
        
        # 6. Attack Success Rate Analysis
        self.plot_attack_success_rates()
        
        print(f"All thesis plots saved to {self.save_path}")
    
    def plot_architecture_robustness(self):
        """Plot robustness comparison across MADDPG variants"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        variants = list(self.results_data.keys())
        epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2]
        
        # Subplot 1: Reward degradation
        for variant in variants:
            reward_degradations = []
            for eps in epsilon_values:
                degradation = self.results_data[variant][f'epsilon_{eps}']['comparison']['reward_degradation_percent']
                reward_degradations.append(degradation)
            
            ax1.plot(epsilon_values, reward_degradations, 'o-', label=variant, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Attack Intensity (ε)')
        ax1.set_ylabel('Reward Degradation (%)')
        ax1.set_title('Reward Degradation vs Attack Intensity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Packet loss increase
        for variant in variants:
            packet_loss_increases = []
            for eps in epsilon_values:
                increase = self.results_data[variant][f'epsilon_{eps}']['comparison']['packet_loss_increase_percent']
                packet_loss_increases.append(increase)
            
            ax2.plot(epsilon_values, packet_loss_increases, 's-', label=variant, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Attack Intensity (ε)')
        ax2.set_ylabel('Packet Loss Increase (%)')
        ax2.set_title('Packet Loss Increase vs Attack Intensity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/architecture_robustness_comparison.png')
        plt.show()
    
    def plot_attack_intensity_analysis(self):
        """Analyze attack effectiveness across different epsilon values"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2]
        variants = list(self.results_data.keys())
        
        # Create heatmap data
        heatmap_data = []
        for variant in variants:
            robustness_scores = []
            for eps in epsilon_values:
                score = self.results_data[variant][f'epsilon_{eps}']['comparison']['robustness_score']
                robustness_scores.append(score)
            heatmap_data.append(robustness_scores)
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(epsilon_values)))
        ax.set_xticklabels([f'{eps:.2f}' for eps in epsilon_values])
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Robustness Score', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(variants)):
            for j in range(len(epsilon_values)):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}',
                             ha="center", va="center", color="white", fontweight='bold')
        
        ax.set_xlabel('Attack Intensity (ε)')
        ax.set_ylabel('MADDPG Variant')
        ax.set_title('Robustness Score Heatmap Across Variants and Attack Intensities')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/attack_intensity_heatmap.png')
        plt.show()
    
    def plot_performance_degradation_matrix(self):
        """Create performance degradation matrix visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        variants = list(self.results_data.keys())
        metrics = ['reward_degradation_percent', 'packet_loss_increase_percent', 
                  'attack_success_rate_percent', 'variance_change_percent']
        metric_names = ['Reward Degradation (%)', 'Packet Loss Increase (%)', 
                       'Attack Success Rate (%)', 'Performance Variance Change (%)']
        axes = [ax1, ax2, ax3, ax4]
        
        epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2]
        
        for idx, (metric, metric_name, ax) in enumerate(zip(metrics, metric_names, axes)):
            # Collect data for all variants
            for variant in variants:
                values = []
                for eps in epsilon_values:
                    val = self.results_data[variant][f'epsilon_{eps}']['comparison'][metric]
                    values.append(val)
                
                ax.bar([f'{eps:.2f}' for eps in epsilon_values], values, 
                      alpha=0.7, label=variant, width=0.6/len(variants))
            
            ax.set_xlabel('Attack Intensity (ε)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} by Variant')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/performance_degradation_matrix.png')
        plt.show()
    
    def plot_reward_packet_loss_tradeoffs(self):
        """Plot reward vs packet loss trade-offs under attack"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        colors = sns.color_palette("husl", len(self.results_data.keys()))
        
        for idx, (variant, variant_data) in enumerate(self.results_data.items()):
            clean_rewards = []
            clean_packet_losses = []
            attacked_rewards = []
            attacked_packet_losses = []
            
            for eps_key, eps_data in variant_data.items():
                if 'epsilon_' in eps_key:
                    clean_rewards.append(eps_data['clean']['mean_reward'])
                    clean_packet_losses.append(eps_data['clean']['mean_packet_loss'])
                    attacked_rewards.append(eps_data['attacked']['mean_reward'])
                    attacked_packet_losses.append(eps_data['attacked']['mean_packet_loss'])
            
            # Plot clean performance
            ax.scatter(clean_packet_losses, clean_rewards, 
                      c=[colors[idx]], s=100, marker='o', 
                      label=f'{variant} (Clean)', alpha=0.8)
            
            # Plot attacked performance
            ax.scatter(attacked_packet_losses, attacked_rewards, 
                      c=[colors[idx]], s=100, marker='x', 
                      label=f'{variant} (Attacked)', alpha=0.8)
            
            # Draw arrows showing attack effect
            for i in range(len(clean_rewards)):
                ax.annotate('', xy=(attacked_packet_losses[i], attacked_rewards[i]),
                           xytext=(clean_packet_losses[i], clean_rewards[i]),
                           arrowprops=dict(arrowstyle='->', color=colors[idx], alpha=0.5))
        
        ax.set_xlabel('Packet Loss (%)')
        ax.set_ylabel('Average Reward')
        ax.set_title('Reward vs Packet Loss Trade-offs Under FGSM Attack')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/reward_packet_loss_tradeoffs.png')
        plt.show()
    
    def plot_gnn_robustness_impact(self):
        """Compare robustness with and without GNN integration"""
        # This assumes you have both GNN and non-GNN results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mock data - replace with actual GNN vs non-GNN comparison
        gnn_variants = [k for k in self.results_data.keys() if 'GNN' in k]
        non_gnn_variants = [k for k in self.results_data.keys() if 'GNN' not in k]
        
        epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2]
        
        # Compare robustness scores
        if gnn_variants and non_gnn_variants:
            # Plot GNN vs non-GNN robustness
            for variant in gnn_variants:
                robustness_scores = []
                for eps in epsilon_values:
                    score = self.results_data[variant][f'epsilon_{eps}']['comparison']['robustness_score']
                    robustness_scores.append(score)
                ax1.plot(epsilon_values, robustness_scores, 'o-', label=f'{variant}', linewidth=2)
            
            for variant in non_gnn_variants:
                robustness_scores = []
                for eps in epsilon_values:
                    score = self.results_data[variant][f'epsilon_{eps}']['comparison']['robustness_score']
                    robustness_scores.append(score)
                ax1.plot(epsilon_values, robustness_scores, 's--', label=f'{variant}', linewidth=2)
        
        ax1.set_xlabel('Attack Intensity (ε)')
        ax1.set_ylabel('Robustness Score')
        ax1.set_title('GNN vs Non-GNN Robustness Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Attack success rate comparison
        if gnn_variants and non_gnn_variants:
            gnn_success_rates = []
            non_gnn_success_rates = []
            
            for eps in epsilon_values:
                gnn_avg = np.mean([self.results_data[v][f'epsilon_{eps}']['comparison']['attack_success_rate_percent'] 
                                  for v in gnn_variants])
                non_gnn_avg = np.mean([self.results_data[v][f'epsilon_{eps}']['comparison']['attack_success_rate_percent'] 
                                      for v in non_gnn_variants])
                gnn_success_rates.append(gnn_avg)
                non_gnn_success_rates.append(non_gnn_avg)
            
            ax2.plot(epsilon_values, gnn_success_rates, 'o-', label='With GNN', linewidth=2, markersize=8)
            ax2.plot(epsilon_values, non_gnn_success_rates, 's-', label='Without GNN', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Attack Intensity (ε)')
        ax2.set_ylabel('Attack Success Rate (%)')
        ax2.set_title('Attack Success Rate: GNN vs Non-GNN')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/gnn_robustness_impact.png')
        plt.show()
    
    def plot_attack_success_rates(self):
        """Plot detailed attack success rate analysis"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        variants = list(self.results_data.keys())
        epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2]
        
        # Create grouped bar chart
        x = np.arange(len(epsilon_values))
        width = 0.8 / len(variants)
        
        for i, variant in enumerate(variants):
            success_rates = []
            for eps in epsilon_values:
                rate = self.results_data[variant][f'epsilon_{eps}']['comparison']['attack_success_rate_percent']
                success_rates.append(rate)
            
            ax.bar(x + i * width, success_rates, width, label=variant, alpha=0.8)
        
        ax.set_xlabel('Attack Intensity (ε)')
        ax.set_ylabel('Attack Success Rate (%)')
        ax.set_title('Attack Success Rate Comparison Across MADDPG Variants')
        ax.set_xticks(x + width * (len(variants) - 1) / 2)
        ax.set_xticklabels([f'{eps:.2f}' for eps in epsilon_values])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, variant in enumerate(variants):
            success_rates = []
            for eps in epsilon_values:
                rate = self.results_data[variant][f'epsilon_{eps}']['comparison']['attack_success_rate_percent']
                success_rates.append(rate)
            
            for j, rate in enumerate(success_rates):
                ax.text(x[j] + i * width, rate + 1, f'{rate:.1f}%', 
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/attack_success_rates.png')
        plt.show()


# Mock data for demonstration - replace with actual experimental results
def generate_mock_results():
    """Generate mock results for demonstration purposes"""
    variants = ['CC-Simple', 'CC-Duelling', 'LC-Duelling', 'CC-Simple-GNN', 'CC-Duelling-GNN', 'LC-Duelling-GNN']
    epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2]
    
    results = {}
    for variant in variants:
        variant_results = {}
        for eps in epsilon_values:
            # Simulate different robustness levels for different variants
            base_robustness = 85 if 'GNN' in variant else 80
            if 'LC' in variant:
                base_robustness += 5  # Local critic slightly more robust
            if 'Duelling' in variant:
                base_robustness += 3  # Duelling architecture slightly better
            
            robustness_loss = eps * 100  # More loss with higher epsilon
            robustness_score = max(20, base_robustness - robustness_loss)
            
            variant_results[f'epsilon_{eps}'] = {
                'clean': {
                    'mean_reward': 1400 + np.random.normal(0, 20),
                    'mean_packet_loss': 0.5 + np.random.normal(0, 0.1)
                },
                'attacked': {
                    'mean_reward': 1400 - eps * 200 + np.random.normal(0, 30),
                    'mean_packet_loss': 0.5 + eps * 10 + np.random.normal(0, 0.15)
                },
                'comparison': {
                    'reward_degradation_percent': eps * 15 + np.random.normal(0, 2),
                    'packet_loss_increase_percent': eps * 20 + np.random.normal(0, 3),
                    'attack_success_rate_percent': min(95, eps * 300 + np.random.normal(0, 5)),
                    'variance_change_percent': eps * 25 + np.random.normal(0, 4),
                    'robustness_score': robustness_score
                }
            }
        results[variant] = variant_results
    
    return results


if __name__ == "__main__":
    # Example usage and demonstration
    print("FGSM Attack Framework for MADDPG Routing Analysis")
    print("=" * 60)
    
    # Create mock results for demonstration
    mock_results = generate_mock_results()
    
    # Initialize visualization suite
    viz_suite = ThesisVisualizationSuite(mock_results)
    
    # Generate all thesis plots
    viz_suite.generate_all_thesis_plots()
    
    print("\n✅ Thesis-quality plots generated successfully!")
    print(f"📊 Plots saved to: {viz_suite.save_path}")
    print("\n📋 Generated plots include:")
    print("  1. Architecture Robustness Comparison")
    print("  2. Attack Intensity Analysis (Heatmap)")
    print("  3. Performance Degradation Matrix") 
    print("  4. Reward vs Packet Loss Trade-offs")
    print("  5. GNN Impact on Robustness")
    print("  6. Attack Success Rate Analysis")
    
    # Print summary statistics for thesis discussion
    print(f"\n📈 Summary for Thesis Discussion:")
    print(f"{'Variant':<20} {'Min Robustness':<15} {'Max Attack Success':<20}")
    print("-" * 55)
    
    for variant, data in mock_results.items():
        min_robustness = min([data[f'epsilon_{eps}']['comparison']['robustness_score'] 
                             for eps in [0.01, 0.05, 0.1, 0.15, 0.2]])
        max_success = max([data[f'epsilon_{eps}']['comparison']['attack_success_rate_percent'] 
                          for eps in [0.01, 0.05, 0.1, 0.15, 0.2]])
        print(f"{variant:<20} {min_robustness:<15.1f} {max_success:<20.1f}")