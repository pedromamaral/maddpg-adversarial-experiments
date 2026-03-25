#!/usr/bin/env python3
"""
Analyze and visualize the traffic matrix used in our MADDPG framework
"""

import sys
import os
sys.path.insert(0, 'src/maddpg_clean')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import seaborn as sns
from network_environment import NetworkEngine, FlowManager, NetworkTopology

def analyze_traffic_patterns():
    """Analyze the traffic patterns and flow generation"""
    
    # Set seed for reproducible analysis
    random.seed(42)
    np.random.seed(42)
    
    # Create network engine
    engine = NetworkEngine("service_provider", 65)
    topology = engine.topology
    flow_manager = engine.flow_manager
    
    print("🚦 TRAFFIC MATRIX ANALYSIS")
    print("=" * 60)
    
    # Generate multiple flow sets to analyze patterns
    all_flows = []
    for i in range(10):  # Generate 10 different flow sets
        flows = flow_manager.generate_random_flows(100)
        all_flows.extend(flows.values())
    
    # Analyze source-destination pairs
    hosts = topology.hosts
    traffic_matrix = np.zeros((len(hosts), len(hosts)))
    host_to_idx = {host: i for i, host in enumerate(hosts)}
    
    # Build traffic matrix
    for flow in all_flows:
        src_idx = host_to_idx[flow['src']]
        dst_idx = host_to_idx[flow['dst']]
        traffic_matrix[src_idx, dst_idx] += flow['packets']
    
    # Analyze patterns
    print(f"📊 Total flows generated: {len(all_flows)}")
    print(f"📊 Unique source-destination pairs: {np.count_nonzero(traffic_matrix)}")
    print(f"📊 Traffic matrix density: {np.count_nonzero(traffic_matrix)/(65*64)*100:.1f}%")
    
    # Flow characteristics
    packet_counts = [flow['packets'] for flow in all_flows]
    priorities = [flow['priority'] for flow in all_flows]
    
    print(f"\n📦 PACKET CHARACTERISTICS:")
    print(f"   Min packets per flow: {min(packet_counts)}")
    print(f"   Max packets per flow: {max(packet_counts)}")
    print(f"   Avg packets per flow: {np.mean(packet_counts):.1f}")
    print(f"   Total packets: {sum(packet_counts)}")
    
    print(f"\n⚡ PRIORITY DISTRIBUTION:")
    priority_counts = {p: priorities.count(p) for p in set(priorities)}
    for priority, count in priority_counts.items():
        print(f"   {priority}: {count} flows ({count/len(priorities)*100:.1f}%)")
    
    # Traffic distribution by tier
    core_hosts = hosts[:6]      # First 6 are core
    dist_hosts = hosts[6:25]    # Next 19 are distribution  
    access_hosts = hosts[25:]   # Remaining are access
    
    print(f"\n🏗️  TRAFFIC BY NETWORK TIER:")
    
    # Count flows by source tier
    core_src_flows = [f for f in all_flows if f['src'] in core_hosts]
    dist_src_flows = [f for f in all_flows if f['src'] in dist_hosts]
    access_src_flows = [f for f in all_flows if f['src'] in access_hosts]
    
    print(f"   Core sources: {len(core_src_flows)} flows ({len(core_src_flows)/len(all_flows)*100:.1f}%)")
    print(f"   Distribution sources: {len(dist_src_flows)} flows ({len(dist_src_flows)/len(all_flows)*100:.1f}%)")
    print(f"   Access sources: {len(access_src_flows)} flows ({len(access_src_flows)/len(all_flows)*100:.1f}%)")
    
    # Count flows by destination tier
    core_dst_flows = [f for f in all_flows if f['dst'] in core_hosts]
    dist_dst_flows = [f for f in all_flows if f['dst'] in dist_hosts]
    access_dst_flows = [f for f in all_flows if f['dst'] in access_hosts]
    
    print(f"\n   Core destinations: {len(core_dst_flows)} flows ({len(core_dst_flows)/len(all_flows)*100:.1f}%)")
    print(f"   Distribution destinations: {len(dist_dst_flows)} flows ({len(dist_dst_flows)/len(all_flows)*100:.1f}%)")
    print(f"   Access destinations: {len(access_dst_flows)} flows ({len(access_dst_flows)/len(all_flows)*100:.1f}%)")
    
    return traffic_matrix, all_flows, hosts

def visualize_traffic_matrix(traffic_matrix, hosts):
    """Create traffic matrix visualizations"""
    
    # Create traffic matrix heatmap
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Full traffic matrix (65x65 is too dense, so show aggregated view)
    # Aggregate by tiers
    core_size = 6
    dist_size = 19
    
    # Aggregate traffic by tiers
    tier_matrix = np.zeros((3, 3))
    
    # Core to Core
    tier_matrix[0, 0] = traffic_matrix[:core_size, :core_size].sum()
    # Core to Dist  
    tier_matrix[0, 1] = traffic_matrix[:core_size, core_size:core_size+dist_size].sum()
    # Core to Access
    tier_matrix[0, 2] = traffic_matrix[:core_size, core_size+dist_size:].sum()
    
    # Dist to Core
    tier_matrix[1, 0] = traffic_matrix[core_size:core_size+dist_size, :core_size].sum()
    # Dist to Dist
    tier_matrix[1, 1] = traffic_matrix[core_size:core_size+dist_size, core_size:core_size+dist_size].sum()
    # Dist to Access
    tier_matrix[1, 2] = traffic_matrix[core_size:core_size+dist_size, core_size+dist_size:].sum()
    
    # Access to Core  
    tier_matrix[2, 0] = traffic_matrix[core_size+dist_size:, :core_size].sum()
    # Access to Dist
    tier_matrix[2, 1] = traffic_matrix[core_size+dist_size:, core_size:core_size+dist_size].sum()
    # Access to Access
    tier_matrix[2, 2] = traffic_matrix[core_size+dist_size:, core_size+dist_size:].sum()
    
    # Plot tier-aggregated traffic matrix
    tier_labels = ['Core', 'Distribution', 'Access']
    sns.heatmap(tier_matrix, annot=True, fmt='.0f', 
                xticklabels=tier_labels, yticklabels=tier_labels,
                cmap='YlOrRd', ax=ax1)
    ax1.set_title('Traffic Matrix by Network Tier\n(Packet Counts)', fontweight='bold')
    ax1.set_xlabel('Destination Tier')
    ax1.set_ylabel('Source Tier')
    
    # Traffic distribution pie chart
    tier_totals = [tier_matrix[i, :].sum() for i in range(3)]
    ax2.pie(tier_totals, labels=tier_labels, autopct='%1.1f%%', colors=['red', 'orange', 'lightblue'])
    ax2.set_title('Traffic Distribution by Source Tier', fontweight='bold')
    
    # Heat map of top traffic pairs (sample of full matrix)
    sample_size = 20
    sample_indices = np.random.choice(len(hosts), sample_size, replace=False)
    sample_matrix = traffic_matrix[np.ix_(sample_indices, sample_indices)]
    sample_hosts = [hosts[i] for i in sample_indices]
    
    sns.heatmap(sample_matrix, annot=False, cmap='YlOrRd', ax=ax3,
                xticklabels=sample_hosts, yticklabels=sample_hosts)
    ax3.set_title(f'Sample Traffic Matrix\n({sample_size}x{sample_size} hosts)', fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=6)
    
    # Traffic volume distribution
    non_zero_traffic = traffic_matrix[traffic_matrix > 0]
    ax4.hist(non_zero_traffic, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('Traffic Volume Distribution\n(Non-zero entries)', fontweight='bold')
    ax4.set_xlabel('Packets per Source-Destination Pair')
    ax4.set_ylabel('Frequency')
    ax4.set_yscale('log')
    
    plt.suptitle('MADDPG Framework - Traffic Matrix Analysis', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('traffic_matrix_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_with_original_traffic():
    """Compare our traffic patterns with typical networking scenarios"""
    
    print("\n" + "="*60)
    print("TRAFFIC MATRIX COMPARISON")
    print("="*60)
    
    print("🎯 OUR IMPLEMENTATION:")
    print("   • Random flow generation (uniform distribution)")
    print("   • 50 initial flows, 10 new flows every 20 steps")
    print("   • Flow cleanup every 50 steps (remove 50% of flows)")
    print("   • Packet counts: 5-20 packets per flow")
    print("   • Priority levels: high/medium/low (equal probability)")
    print("   • Source-destination pairs: uniformly random")
    
    print("\n📚 TYPICAL NETWORKING PATTERNS:")
    print("   • Client-server traffic (80% access-to-core/distribution)")
    print("   • Peer-to-peer traffic (15% access-to-access)")
    print("   • Management traffic (5% core-to-all)")
    print("   • Heavy-tail distributions (few heavy flows, many light)")
    print("   • Time-of-day variations (business hours vs. off-hours)")
    
    print("\n⚖️  EVALUATION:")
    print("   ✅ GOOD: Uniform coverage ensures comprehensive testing")
    print("   ✅ GOOD: Dynamic flow generation simulates realistic churn")
    print("   ✅ GOOD: Multiple priority levels test QoS handling")
    print("   ⚠️  SIMPLIFIED: Real networks have more predictable patterns")
    print("   ⚠️  SIMPLIFIED: No geographic or application-specific clustering")
    
    print("\n🎓 FOR THESIS PURPOSES:")
    print("   ✅ Random patterns are BETTER for adversarial evaluation")
    print("   ✅ Uniform coverage tests all network regions equally")
    print("   ✅ No bias toward specific traffic patterns")
    print("   ✅ More challenging for both agents and attacks")
    print("   ✅ Results are more generalizable")

def analyze_dynamic_behavior():
    """Analyze how traffic patterns change during a simulation run"""
    
    print("\n" + "="*60) 
    print("DYNAMIC TRAFFIC ANALYSIS")
    print("="*60)
    
    # Set seed
    random.seed(42)
    np.random.seed(42)
    
    # Create engine and run simulation
    engine = NetworkEngine("service_provider", 65)
    
    # Track traffic over time
    timesteps = []
    flow_counts = []
    packet_counts = []
    
    print("🔄 Simulating 100 timesteps...")
    
    for t in range(100):
        # Get current state (this triggers flow management)
        states = []
        for host in engine.get_all_hosts()[:5]:  # Sample 5 hosts
            state = engine.get_state(host)
            states.append(state)
        
        # Random actions (for simulation)
        actions = [np.random.dirichlet([1, 1, 1]) for _ in range(5)]
        
        # Step environment 
        next_states, rewards, info = engine.step(states[:5], actions)
        
        # Track metrics
        timesteps.append(t)
        flow_counts.append(len(engine.current_flows))
        packet_counts.append(info['packets_sent'])
        
        if t % 20 == 0:
            print(f"   Step {t:3d}: {len(engine.current_flows):3d} flows, {info['packets_sent']:4d} packets")
    
    print(f"\n📊 FLOW DYNAMICS:")
    print(f"   Initial flows: {flow_counts[0]}")
    print(f"   Final flows: {flow_counts[-1]}")
    print(f"   Min flows: {min(flow_counts)}")
    print(f"   Max flows: {max(flow_counts)}")
    print(f"   Avg flows: {np.mean(flow_counts):.1f}")
    
    print(f"\n📦 PACKET DYNAMICS:")
    print(f"   Min packets/step: {min(packet_counts)}")
    print(f"   Max packets/step: {max(packet_counts)}")
    print(f"   Avg packets/step: {np.mean(packet_counts):.1f}")
    print(f"   Total packets: {sum(packet_counts)}")
    
    # Create time series plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(timesteps, flow_counts, 'b-', linewidth=2, label='Active Flows')
    ax1.set_ylabel('Number of Flows')
    ax1.set_title('Dynamic Flow Management Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(timesteps, packet_counts, 'r-', linewidth=2, label='Packets per Step')
    ax2.set_ylabel('Packets Sent')
    ax2.set_xlabel('Simulation Step')
    ax2.set_title('Packet Generation Over Time', fontweight='bold') 
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('traffic_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return flow_counts, packet_counts

if __name__ == "__main__":
    print("🚦 MADDPG Framework - Traffic Matrix Analysis")
    print("=" * 60)
    
    # Analyze traffic patterns
    traffic_matrix, flows, hosts = analyze_traffic_patterns()
    
    # Create visualizations
    print("\n📊 Creating traffic visualizations...")
    visualize_traffic_matrix(traffic_matrix, hosts)
    
    # Compare with original
    compare_with_original_traffic()
    
    # Analyze dynamics
    flow_counts, packet_counts = analyze_dynamic_behavior()
    
    print(f"\n📊 Visualizations saved:")
    print("   • traffic_matrix_analysis.png")
    print("   • traffic_dynamics.png")
    print("✅ Traffic analysis complete!")