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

# ==========================================
# CTM PHYSICS CONSTANTS (Traffic Engineering Standards)
# ==========================================
JAM_DENSITY = 150.0  # vehicles per km per lane (standard urban value)
SATURATION_FLOW = 1800.0  # vehicles per hour per lane (max capacity)
WAVE_SPEED = 15.0  # km/h (speed at which shockwave travels backward)
SIMULATION_TIME_MINUTES = 45  # Total simulation duration (reduced for efficiency)
TIME_STEP_SECONDS = 10  # Resolution of simulation (10 seconds - balanced speed/accuracy)
FREE_FLOW_SPEED = 50.0  # km/h (typical urban speed limit)

# Derived constants
SIMULATION_STEPS = int((SIMULATION_TIME_MINUTES * 60) / TIME_STEP_SECONDS)
TIME_STEP_HOURS = TIME_STEP_SECONDS / 3600.0

# BPR Formula constants (Bureau of Public Roads)
BPR_ALPHA = 0.15  # Time multiplier coefficient
BPR_BETA = 4.0  # Exponential factor


# ==========================================
# CTM LINK CLASS (Cell Transmission Model)
# ==========================================
class CTMLink:
    """
    Represents a road segment (edge) in the Cell Transmission Model.
    Tracks vehicle density and calculates sending/receiving flows.
    """
    
    def __init__(self, edge_key, length_km, num_lanes=1, base_travel_time=1.0, is_metro=False):
        self.edge_key = edge_key  # (u, v, k)
        self.length_km = length_km
        self.num_lanes = max(1, num_lanes)  # At least 1 lane
        self.base_travel_time = base_travel_time
        self.is_metro = is_metro
        
        # Capacity calculations
        self.capacity_veh = self.length_km * self.num_lanes * JAM_DENSITY  # Max vehicles that fit
        self.q_max = self.num_lanes * SATURATION_FLOW  # Max flow rate (veh/hr)
        
        # State variables (change during simulation)
        self.n_vehicles = 0.0  # Current number of vehicles on link
        
        # Critical density (where flow is maximum)
        # Using triangular fundamental diagram: critical density = q_max / free_flow_speed
        self.critical_density = self.q_max / FREE_FLOW_SPEED if FREE_FLOW_SPEED > 0 else self.capacity_veh * 0.3
    
    def get_current_density(self):
        """Get current density (vehicles per km per lane)"""
        if self.length_km <= 0:
            return 0.0
        return self.n_vehicles / (self.length_km * self.num_lanes)
    
    def get_sending_flow(self):
        """
        Calculate sending flow: How many vehicles want to leave this link?
        Based on current density and fundamental diagram.
        """
        if self.is_metro:
            # Metro has constant high flow, not affected by road traffic
            return self.q_max
        
        density = self.get_current_density()
        
        if density <= self.critical_density:
            # Free flow: flow increases linearly with density
            # Flow = density * free_flow_speed
            flow = density * FREE_FLOW_SPEED * self.num_lanes
        else:
            # Congested: flow decreases as density increases
            # Using triangular fundamental diagram
            # Flow = q_max * (jam_density - density) / (jam_density - critical_density)
            jam_density_per_lane = JAM_DENSITY
            critical_density_per_lane = self.critical_density / self.num_lanes if self.num_lanes > 0 else 0
            
            if jam_density_per_lane <= critical_density_per_lane:
                flow = 0.0
            else:
                density_per_lane = density / self.num_lanes if self.num_lanes > 0 else 0
                flow = self.q_max * (jam_density_per_lane - density_per_lane) / (jam_density_per_lane - critical_density_per_lane)
        
        # Ensure non-negative and within capacity
        flow = max(0.0, min(flow, self.q_max))
        return flow
    
    def get_receiving_flow(self):
        """
        Calculate receiving flow: How many vehicles can enter this link?
        Limited by available space (jam density - current density).
        """
        if self.is_metro:
            # Metro has high capacity, not affected by road traffic
            return self.q_max
        
        density = self.get_current_density()
        jam_density_per_lane = JAM_DENSITY
        
        if density >= jam_density_per_lane:
            # Link is full (jam density reached)
            return 0.0
        
        # Available space determines receiving capacity
        # Receiving flow = q_max * (jam_density - current_density) / jam_density
        available_space_ratio = (jam_density_per_lane - density) / jam_density_per_lane
        receiving_flow = self.q_max * available_space_ratio
        
        return max(0.0, receiving_flow)
    
    def add_vehicles(self, num_vehicles):
        """Safely add vehicles to the link (respecting capacity)"""
        self.n_vehicles = min(self.n_vehicles + num_vehicles, self.capacity_veh)
    
    def remove_vehicles(self, num_vehicles):
        """Safely remove vehicles from the link"""
        self.n_vehicles = max(0.0, self.n_vehicles - num_vehicles)
    
    def set_capacity_zero(self):
        """Set capacity to zero (for road closures)"""
        self.q_max = 0.0
        self.capacity_veh = 0.0
    
    def reset(self):
        """Reset link to initial state"""
        self.n_vehicles = 0.0


# ==========================================
# BPR FORMULA: Density to Congestion Factor
# ==========================================
def density_to_congestion_factor(link, base_travel_time):
    """
    Convert vehicle density to congestion factor using BPR formula.
    
    BPR Formula: t = t0 * (1 + alpha * (V/C)^beta)
    Where:
    - t0 = base travel time
    - V = current volume (density)
    - C = capacity (max density)
    - alpha, beta = calibration parameters
    
    Returns: Congestion factor (multiplier for base travel time)
    """
    if link.is_metro:
        # Metro maintains constant speed
        return 1.0
    
    current_density = link.get_current_density()
    max_density = JAM_DENSITY
    
    if max_density <= 0 or current_density <= 0:
        return 1.0
    
    # Volume/Capacity ratio
    v_over_c = current_density / max_density
    
    # BPR Formula
    congestion_factor = 1.0 + BPR_ALPHA * (v_over_c ** BPR_BETA)
    
    return congestion_factor


# ==========================================
# CTM SIMULATION ENGINE
# ==========================================
def simulate_congestion_event_ctm(G, base_graph_data):
    """
    Simulates a traffic event using Cell Transmission Model (CTM).
    This replaces the old BFS ripple heuristic with physics-based flow propagation.
    """
    # 1. Initialize CTM Links from graph edges
    links = {}
    edge_to_link = {}
    
    for u, v, k, data in G.edges(keys=True, data=True):
        edge_key = (u, v, k)
        
        # Get edge properties
        try:
            length_km = float(data.get('length', 50)) / 1000.0  # Convert m to km
        except:
            length_km = 0.05  # Default 50m
        
        # Estimate lanes (default 1, can be enhanced with OSM data)
        num_lanes = 1
        if 'lanes' in data:
            try:
                lanes_val = data['lanes']
                if isinstance(lanes_val, list):
                    num_lanes = int(lanes_val[0]) if len(lanes_val) > 0 else 1
                else:
                    num_lanes = int(lanes_val)
            except:
                num_lanes = 1
        
        # Get base travel time
        try:
            base_tt = float(data.get('base_travel_time', 1.0))
        except:
            base_tt = 1.0
        
        # Check if metro
        is_metro = str(data.get('is_metro', 'False')) == 'True'
        
        # Create CTM Link
        link = CTMLink(edge_key, length_km, num_lanes, base_tt, is_metro)
        links[edge_key] = link
        edge_to_link[edge_key] = link
    
    # 2. Randomly select edges to close (simulating accidents/construction)
    road_edges = base_graph_data['road_edges']
    num_closures = random.randint(1, min(4, len(road_edges)))
    
    if len(road_edges) < num_closures:
        closed_edges = road_edges
    else:
        closed_edges = random.sample(road_edges, num_closures)
    
    # 3. Apply closures by setting capacity to zero
    for edge_key in closed_edges:
        if edge_key in links:
            links[edge_key].set_capacity_zero()
    
    # 4. Initialize vehicle distribution (start with some baseline traffic)
    # Distribute vehicles based on population density at nodes
    for edge_key, link in links.items():
        if link.is_metro:
            continue  # Metro starts empty
        
        u, v, k = edge_key
        pop_u = base_graph_data['populations'].get(u, 0)
        pop_v = base_graph_data['populations'].get(v, 0)
        avg_pop = (pop_u + pop_v) / 2.0
        
        # Initial density proportional to population (normalized)
        # Higher population = more initial traffic
        initial_density_factor = min(avg_pop / 5000.0, 1.0)  # Normalize to 0-1
        initial_density = initial_density_factor * link.critical_density * 0.5  # Start at 50% of critical
        
        link.n_vehicles = initial_density * link.length_km * link.num_lanes
    
    # 5. TIME-STEPPED SIMULATION LOOP
    for step in range(SIMULATION_STEPS):
        # Phase A: Calculate flows for all node intersections
        # Process each node independently to minimize memory usage
        
        # Build node-to-edges mapping (more efficient)
        # For each node, track edges that END at it (incoming) and START from it (outgoing)
        node_incoming_edges = {}  # {node: [list of (u,v,k) edges ending at node]}
        node_outgoing_edges = {}  # {node: [list of (u,v,k) edges starting from node]}
        
        for edge_key, link in links.items():
            u, v, k = edge_key
            if link.q_max <= 0:  # Skip closed links
                continue
            
            # Edge (u,v,k) ends at node v (incoming to v)
            if v not in node_incoming_edges:
                node_incoming_edges[v] = []
            node_incoming_edges[v].append(edge_key)
            
            # Edge (u,v,k) starts from node u (outgoing from u)
            if u not in node_outgoing_edges:
                node_outgoing_edges[u] = []
            node_outgoing_edges[u].append(edge_key)
        
        # Phase B: Process each node and calculate flows
        flow_updates = {}  # {(u, v, k): net_flow_change}
        
        # Process nodes that have both incoming and outgoing edges (intersections)
        all_nodes = set(node_incoming_edges.keys()) | set(node_outgoing_edges.keys())
        
        for node in all_nodes:
            # Get incoming links (edges ending at this node - vehicles arrive here)
            incoming = node_incoming_edges.get(node, [])
            # Get outgoing links (edges starting from this node - vehicles leave here)
            outgoing = node_outgoing_edges.get(node, [])
            
            if not incoming or not outgoing:
                continue
            
            # Calculate total sending flow from incoming links
            total_sending = 0.0
            sending_flows = {}
            for edge_key in incoming:
                link = links[edge_key]
                sending = link.get_sending_flow()
                sending_flows[edge_key] = sending
                total_sending += sending
            
            # Calculate total receiving capacity from outgoing links
            total_receiving = 0.0
            receiving_caps = {}
            for edge_key in outgoing:
                link = links[edge_key]
                receiving = link.get_receiving_flow()
                receiving_caps[edge_key] = receiving
                total_receiving += receiving
            
            # Actual flow = min(sending, receiving) (CTM rule)
            actual_flow = min(total_sending, total_receiving)
            
            if actual_flow <= 0:
                continue
            
            # Convert to vehicles per time step
            flow_per_step = actual_flow * TIME_STEP_HOURS
            
            # Distribute flow proportionally to receiving capacity
            if total_receiving > 0:
                for edge_key in outgoing:
                    receiving_cap = receiving_caps[edge_key]
                    if receiving_cap > 0:
                        flow_fraction = receiving_cap / total_receiving
                        flow_amount = flow_per_step * flow_fraction
                        
                        if edge_key not in flow_updates:
                            flow_updates[edge_key] = 0.0
                        flow_updates[edge_key] += flow_amount
            
            # Remove flow from incoming links proportionally
            if total_sending > 0:
                for edge_key in incoming:
                    sending = sending_flows[edge_key]
                    sending_fraction = sending / total_sending
                    remove_amount = flow_per_step * sending_fraction
                    
                    if edge_key not in flow_updates:
                        flow_updates[edge_key] = 0.0
                    flow_updates[edge_key] -= remove_amount
        
        # Phase C: Apply flow updates (update vehicle counts)
        for edge_key, flow_change in flow_updates.items():
            if edge_key in links:
                if flow_change > 0:
                    links[edge_key].add_vehicles(flow_change)
                else:
                    links[edge_key].remove_vehicles(abs(flow_change))
    
    # 6. Convert final density to congestion factors (BPR Formula)
    edge_travel_times = {}
    edge_congestion = {}
    
    for edge_key, link in links.items():
        u, v, k = edge_key
        base_tt = base_graph_data['travel_times'].get(edge_key, 1.0)
        
        # Calculate congestion factor from density
        congestion_factor = density_to_congestion_factor(link, base_tt)
        
        # Calculate final travel time
        final_travel_time = base_tt * congestion_factor
        
        edge_travel_times[edge_key] = final_travel_time
        edge_congestion[edge_key] = congestion_factor
    
    # 7. Pack snapshot (same format as before)
    snapshot = SimpleNamespace(
        edge_travel_times=edge_travel_times,
        edge_congestion=edge_congestion,
        closed_edges=closed_edges,
        node_populations=base_graph_data['populations']
    )
    
    return snapshot


# ==========================================
# DATASET GENERATION
# ==========================================
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

    print(f"   Generating {NUM_SNAPSHOTS} CTM-based traffic scenarios...")
    print(f"   Simulation: {SIMULATION_TIME_MINUTES} min, {TIME_STEP_SECONDS}s steps, {SIMULATION_STEPS} iterations per scenario")
    print(f"   Memory optimization: Clearing cache every 250 scenarios")
    snapshots = []
    
    # Run Simulation with memory optimization
    for i in tqdm(range(NUM_SNAPSHOTS), desc="Generating scenarios"):
        snap = simulate_congestion_event_ctm(G, base_data)
        snapshots.append(snap)
        
        # Memory optimization: Clear intermediate data periodically
        # More frequent clearing for 6GB GPU systems
        if (i + 1) % 250 == 0:
            import gc
            gc.collect()
            print(f"   [Memory cleanup at {i+1}/{NUM_SNAPSHOTS}]")
        
    # Save
    print(f"💾 Saving dataset to {OUTPUT_DATA}...")
    with open(OUTPUT_DATA, 'wb') as f:
        pickle.dump(snapshots, f)
        
    print("✅ Done! CTM dataset ready for training.")
    print(f"   Generated {len(snapshots)} scenarios with physics-based traffic flow.")


if __name__ == "__main__":
    generate_dataset()
