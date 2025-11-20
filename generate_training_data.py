"""
Generate Training Data for GNN Digital Twin
===========================================

This script generates training datasets using the macroscopic traffic 
simulation for training Graph Neural Networks to predict traffic patterns.

The "Trick": We simulate traffic flow mathematically (pressure model) 
instead of tracking individual vehicles (agents).

Training Strategy:
1. Randomly close roads
2. Propagate congestion upstream using graph connectivity
3. Record edge weights (travel times) as training samples
4. Simulate recovery and random events for realism
"""

import networkx as nx
import random
import pickle
from macroscopic_traffic_simulation import (
    MacroscopicTrafficSimulator,
    SimulationConfig
)


def generate_training_scenarios(
    graph: nx.Graph,
    num_scenarios: int = 100,
    duration_per_scenario: int = 60,
    output_file: str = 'gnn_training_data.pkl'
):
    """
    Generate diverse traffic scenarios for GNN training.
    
    Args:
        graph: City road network
        num_scenarios: Number of different scenarios to simulate
        duration_per_scenario: Minutes to simulate each scenario
        output_file: Output pickle file
    """
    print(f"🎯 Generating {num_scenarios} training scenarios...")
    print(f"   Duration per scenario: {duration_per_scenario} minutes")
    print(f"   Total simulation time: {num_scenarios * duration_per_scenario} minutes")
    
    all_training_data = []
    
    # Configuration for varied scenarios
    configs = [
        # Normal traffic
        SimulationConfig(
            base_congestion_multiplier=2.0,
            ripple_decay=0.8,
            ripple_depth=2,
            random_event_probability=0.01
        ),
        # Heavy congestion
        SimulationConfig(
            base_congestion_multiplier=4.0,
            ripple_decay=0.6,
            ripple_depth=4,
            random_event_probability=0.05
        ),
        # Moderate traffic
        SimulationConfig(
            base_congestion_multiplier=3.0,
            ripple_decay=0.7,
            ripple_depth=3,
            random_event_probability=0.03
        ),
        # Light traffic with quick recovery
        SimulationConfig(
            base_congestion_multiplier=2.5,
            ripple_decay=0.9,
            ripple_depth=2,
            random_event_probability=0.02,
            recovery_rate=0.9
        )
    ]
    
    for scenario_idx in range(num_scenarios):
        print(f"\n{'='*60}")
        print(f"📋 Scenario {scenario_idx + 1}/{num_scenarios}")
        print(f"{'='*60}")
        
        # Choose configuration
        config = configs[scenario_idx % len(configs)]
        
        # Initialize simulator
        sim = MacroscopicTrafficSimulator(graph, config)
        
        # Determine number of roads to close (1-5)
        num_closures = random.randint(1, 5)
        
        # Get list of edges
        edges = list(sim.G.edges(keys=True))
        
        # Select random edges to close at different times
        closure_schedule = []
        for i in range(num_closures):
            edge = random.choice(edges)
            closure_time = random.randint(5, 30)  # Close between minute 5-30
            reopening_time = closure_time + random.randint(15, 40)  # Reopen 15-40 min later
            
            closure_schedule.append({
                'edge': edge,
                'close_at': closure_time,
                'reopen_at': reopening_time
            })
        
        print(f"📍 Scheduled closures: {num_closures}")
        for idx, event in enumerate(closure_schedule, 1):
            u, v, key = event['edge']
            print(f"   {idx}. Edge {u}->{v} (close: {event['close_at']}m, reopen: {event['reopen_at']}m)")
        
        # Run simulation
        closed_edges = set()
        
        for minute in range(duration_per_scenario):
            # Check for scheduled closures
            for event in closure_schedule:
                if minute == event['close_at']:
                    u, v, key = event['edge']
                    sim.close_road(u, v, key)
                    closed_edges.add((u, v, key))
                    print(f"\n⏰ Minute {minute}: Closing {u}->{v}")
                
                elif minute == event['reopen_at'] and event['edge'] in closed_edges:
                    u, v, key = event['edge']
                    sim.reopen_road(u, v, key)
                    closed_edges.discard((u, v, key))
                    print(f"\n⏰ Minute {minute}: Reopening {u}->{v}")
            
            # Advance simulation
            sim.step(delta_time=1.0)
            
            # Print progress every 15 minutes
            if (minute + 1) % 15 == 0:
                print(f"\n   Progress: {minute + 1}/{duration_per_scenario} minutes")
                stats = sim.get_statistics()
                print(f"   Network delay: {stats['total_network_delay']:.1f}m")
                print(f"   Congested edges: {stats['congested_edges']}/{stats['total_edges']}")
        
        # Store scenario data
        scenario_data = {
            'scenario_id': scenario_idx,
            'config': config,
            'closure_schedule': closure_schedule,
            'snapshots': sim.snapshots,
            'final_stats': sim.get_statistics()
        }
        
        all_training_data.append(scenario_data)
        
        print(f"\n✅ Scenario {scenario_idx + 1} complete")
        sim.print_statistics()
    
    # Compile final training dataset
    training_dataset = {
        'scenarios': all_training_data,
        'graph_info': {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges()
        },
        'num_scenarios': num_scenarios,
        'total_snapshots': sum(len(s['snapshots']) for s in all_training_data)
    }
    
    # Save to file
    with open(output_file, 'wb') as f:
        pickle.dump(training_dataset, f)
    
    print(f"\n{'='*60}")
    print(f"🎉 TRAINING DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"📦 Output file: {output_file}")
    print(f"📊 Scenarios: {num_scenarios}")
    print(f"📸 Total snapshots: {training_dataset['total_snapshots']}")
    print(f"💾 File size: {get_file_size(output_file)}")
    print(f"{'='*60}\n")


def get_file_size(filename: str) -> str:
    """Get human-readable file size"""
    try:
        import os
        size_bytes = os.path.getsize(filename)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.2f} TB"
    except:
        return "Unknown"


def main():
    """Main entry point"""
    print("🌆 Loading city graph...")
    
    try:
        G = nx.read_graphml('city_graph.graphml')
        print(f"✅ Loaded city graph")
        print(f"   Nodes: {G.number_of_nodes()}")
        print(f"   Edges: {G.number_of_edges()}")
    except FileNotFoundError:
        print("❌ city_graph.graphml not found!")
        print("   Please run generate_complex_city.py first")
        return
    
    # Generate training data
    generate_training_scenarios(
        graph=G,
        num_scenarios=100,  # Adjust based on your needs
        duration_per_scenario=60,
        output_file='gnn_training_data.pkl'
    )
    
    print("\n🚀 Next Steps:")
    print("   1. Use gnn_training_data.pkl to train your GNN model")
    print("   2. The model learns: closed_edge → congestion_pattern")
    print("   3. Deploy in dashboard for real-time predictions")


if __name__ == "__main__":
    main()
