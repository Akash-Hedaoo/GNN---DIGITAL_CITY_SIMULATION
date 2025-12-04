import networkx as nx
import random
import pickle
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm  # Progress bar

# Configuration
INPUT_GRAPH = 'real_city_processed.graphml'
OUTPUT_DATA = 'gnn_training_data.pkl'
NUM_SNAPSHOTS = 2000  # Number of training examples

# Physics Constants for "Density-Aware" Simulation
# High decay (1.0) = Traffic persists (Urban center)
# Low decay (0.6) = Traffic dissipates (Suburbs)
BASE_DECAY = 0.6        
POPULATION_WEIGHT = 0.4 

def get_upstream_edges(G, u, v, key):
    """Find edges feeding into the target edge (u, v)"""
    upstream = []
    # In a directed graph, edges feeding (u,v) are (predecessor, u)
    if u in G:
        for predecessor in G.predecessors(u):
            if G.has_edge(predecessor, u):
                for k in G[predecessor][u]:
                    upstream.append((predecessor, u, k))
    return upstream

def simulate_congestion_event(G, base_graph_data):
    """
    Simulates a traffic event with Population-Aware Physics.
    """
    # 1. Clone base data
    current_travel_times = base_graph_data['travel_times'].copy()
    current_congestion = {k: 1.0 for k in base_graph_data['travel_times'].keys()}
    
    # 2. Randomly close 1 to 4 road edges (Simulating accidents/construction)
    road_edges = base_graph_data['road_edges']
    num_closures = random.randint(1, 4)
    
    if len(road_edges) < num_closures:
        closed_edges = road_edges
    else:
        closed_edges = random.sample(road_edges, num_closures)
    
    # 3. Apply Closure & Propagate
    # Queue: (u, v, key, congestion_factor)
    queue = [] 
    visited = set()
    
    # Initialize closures (Factor 50.0 = Blocked/Severe Jam)
    for u, v, k in closed_edges:
        current_congestion[(u, v, k)] = 50.0
        current_travel_times[(u, v, k)] *= 50.0 
        queue.append((u, v, k, 50.0))
        visited.add((u, v, k))

    # BFS Propagation
    depth = 0
    max_depth = 5 
    
    while queue:
        next_queue = []
        for u, v, k, factor in queue:
            # Stop if ripple is too weak
            if factor < 1.1: continue
                
            # Find upstream edges (traffic flowing INTO this jammed edge)
            upstream = get_upstream_edges(G, u, v, k)
            
            for up_u, up_v, up_k in upstream:
                if (up_u, up_v, up_k) in visited: continue
                
                # METRO CHECK: Metro edges are immune to traffic
                edge_data = G[up_u][up_v][up_k]
                # Robust check for string 'True' or boolean True
                is_metro_val = edge_data.get('is_metro', 'False')
                is_metro = str(is_metro_val) == 'True'
                
                if is_metro: continue
                    
                # DENSITY PHYSICS:
                # Calculate decay based on the POPULATION of the intersection (up_u)
                # Denser area = Traffic jam holds longer (higher decay factor)
                # Emptier area = Traffic jam dissipates faster (lower decay factor)
                node_pop = base_graph_data['populations'].get(up_u, 0)
                
                # Normalize pop (0 to ~5000) to a 0.0-1.0 factor
                pop_factor = min(node_pop / 5000.0, 1.0)
                
                # Dynamic Decay: Base (0.6) + PopBoost (up to 0.4)
                # Range: 0.6 (Empty) to 1.0 (Dense Center)
                current_node_decay = BASE_DECAY + (POPULATION_WEIGHT * pop_factor)
                
                # Calculate new factor
                new_factor = 1.0 + (factor - 1.0) * current_node_decay
                
                # Update State
                current_congestion[(up_u, up_v, up_k)] = new_factor
                current_travel_times[(up_u, up_v, up_k)] *= new_factor
                
                visited.add((up_u, up_v, up_k))
                next_queue.append((up_u, up_v, up_k, new_factor))
        
        queue = next_queue
        depth += 1
        if depth >= max_depth:
            break

    # 4. Pack Snapshot
    snapshot = SimpleNamespace(
        edge_travel_times=current_travel_times,
        edge_congestion=current_congestion,
        closed_edges=closed_edges,
        node_populations=base_graph_data['populations']
    )
    return snapshot

def generate_dataset():
    print(f"🔄 Loading graph: {INPUT_GRAPH}...")
    
    # Robust Loading: Use standard NetworkX to avoid OSMnx type strictness
    G = nx.read_graphml(INPUT_GRAPH)
    
    # Ensure MultiDiGraph for handling parallel edges
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)
    
    print("   Caching static graph data...")
    road_edges = []
    base_travel_times = {}
    node_populations = {}
    
    # Cache Edge Data
    for u, v, k, data in G.edges(keys=True, data=True):
        # Robust is_metro check
        is_metro = str(data.get('is_metro', 'False')) == 'True'
        
        # Robust float parsing
        try:
            val = data.get('base_travel_time', 1.0)
            if isinstance(val, list): val = val[0]
            tt = float(val)
        except: tt = 1.0
        
        base_travel_times[(u, v, k)] = tt
        
        if not is_metro:
            road_edges.append((u, v, k))
            
    # Cache Node Data (Populations)
    for n, data in G.nodes(data=True):
        try:
            val = data.get('population', 0)
            if isinstance(val, list): val = val[0]
            pop = float(val)
        except: pop = 0.0
        node_populations[n] = pop

    base_data = {
        'road_edges': road_edges,
        'travel_times': base_travel_times,
        'populations': node_populations
    }

    print(f"   Generating {NUM_SNAPSHOTS} density-aware traffic scenarios...")
    snapshots = []
    
    # Run Simulation
    for _ in tqdm(range(NUM_SNAPSHOTS)):
        snap = simulate_congestion_event(G, base_data)
        snapshots.append(snap)
        
    # Save
    print(f"💾 Saving dataset to {OUTPUT_DATA}...")
    with open(OUTPUT_DATA, 'wb') as f:
        pickle.dump(snapshots, f)
        
    print("✅ Done! Dataset ready for training.")

if __name__ == "__main__":
    generate_dataset()