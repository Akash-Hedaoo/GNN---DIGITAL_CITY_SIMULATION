"""
Macroscopic Traffic Simulation using Fluid Dynamics Approach
============================================================

This module implements a pressure-based traffic flow model that simulates 
traffic congestion without tracking individual vehicles (agents).

Key Concepts:
1. Traffic as Fluid: Roads are pipes, traffic is water pressure
2. Backlog Propagation: Closed roads cause ripple effects upstream
3. Graph Connectivity: Uses predecessors/successors to simulate flow
4. Mathematical Validity: Probability-based congestion multipliers

Author: Digital Twin City Simulation
Date: November 2025
"""

import networkx as nx
import numpy as np
import random
import pickle
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import time


@dataclass
class TrafficSnapshot:
    """Represents traffic state at a specific time"""
    timestamp: float
    edge_travel_times: Dict[Tuple[int, int, int], float]  # (u, v, key) -> time
    edge_congestion: Dict[Tuple[int, int, int], float]  # (u, v, key) -> congestion factor
    closed_edges: Set[Tuple[int, int, int]]
    total_network_delay: float
    node_populations: Optional[Dict[int, int]] = None  # Population at each node
    node_daily_trips: Optional[Dict[int, int]] = None  # Daily trips from each node


@dataclass
class SimulationConfig:
    """Configuration for macroscopic simulation"""
    base_congestion_multiplier: float = 3.0  # Initial congestion impact
    ripple_decay: float = 0.7  # How much congestion decreases per hop
    ripple_depth: int = 3  # How many hops to propagate
    time_quantum: float = 1.0  # Time step in minutes
    recovery_rate: float = 0.85  # How fast congestion clears
    random_event_probability: float = 0.05  # Chance of random slowdown
    closed_road_penalty: float = 999.0  # Virtually infinite travel time


class MacroscopicTrafficSimulator:
    """
    Simulates city-wide traffic using fluid dynamics principles.
    
    Instead of tracking individual vehicles, we model traffic flow as 
    pressure waves through a network of pipes (roads).
    """
    
    def __init__(self, graph: nx.MultiDiGraph, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulator with a city graph.
        
        Args:
            graph: NetworkX MultiDiGraph representing the road network
            config: Simulation configuration parameters
        """
        # Convert to MultiDiGraph if needed
        if isinstance(graph, nx.MultiDiGraph):
            self.G = graph.copy()
        else:
            self.G = nx.MultiDiGraph(graph)
        
        self.config = config or SimulationConfig()
        self.is_multigraph = isinstance(graph, nx.MultiDiGraph)
        
        # Current state (must initialize before _initialize_base_times)
        self.current_travel_times = {}
        self.congestion_factors = {}
        self.closed_edges = set()
        self.population_demand = {}  # Track demand from population
        
        # History tracking
        self.snapshots: List[TrafficSnapshot] = []
        self.simulation_time = 0.0
        
        # Initialize base travel times
        self._initialize_base_times()
        self._calculate_population_demand()
        
        print(f"[TRAFFIC] Macroscopic Traffic Simulator initialized")
        print(f"   Nodes: {self.G.number_of_nodes()}")
        print(f"   Edges: {self.G.number_of_edges()}")
        
    def _initialize_base_times(self):
        """Calculate base travel times from edge attributes"""
        metro_count = 0
        road_count = 0
        
        for u, v, key, data in self.G.edges(keys=True, data=True):
            # Get length and speed
            length_km = data.get('length', 1.0)  # Default 1 km
            speed_kmh = data.get('speed_limit', 40.0)  # Default 40 km/h
            
            # Calculate base travel time in minutes
            base_time = (length_km / speed_kmh) * 60.0
            
            # Check if this is a metro edge
            is_metro = data.get('is_metro', False) or data.get('transport_mode') == 'metro'
            
            # Store in graph
            self.G[u][v][key]['base_travel_time'] = base_time
            self.G[u][v][key]['current_travel_time'] = base_time
            self.G[u][v][key]['is_metro'] = is_metro
            
            # Initialize our tracking
            self.current_travel_times[(u, v, key)] = base_time
            self.congestion_factors[(u, v, key)] = 1.0
            
            if is_metro:
                metro_count += 1
            else:
                road_count += 1
        
        if metro_count > 0:
            print(f"   [METRO] Metro edges: {metro_count}")
            print(f"   🚗 Road edges: {road_count}")
    
    def _calculate_population_demand(self):
        """
        Calculate traffic demand based on node populations.
        Higher population = more traffic demand on edges.
        
        This creates realistic baseline congestion from human activity,
        not just random events.
        """
        total_daily_trips = 0
        
        for node in self.G.nodes():
            population = self.G.nodes[node].get('population', 0)
            daily_trips = self.G.nodes[node].get('daily_trips', 0)
            
            if daily_trips > 0:
                total_daily_trips += daily_trips
                self.population_demand[node] = daily_trips
        
        # Apply population demand to edges by distributing across connected edges
        for u, v, key, data in self.G.edges(keys=True, data=True):
            if (u, v, key) not in self.population_demand:
                # Estimate demand on this edge from endpoint populations
                u_demand = self.population_demand.get(u, 0)
                v_demand = self.population_demand.get(v, 0)
                avg_demand = (u_demand + v_demand) / 2.0
                
                # Apply demand as baseline congestion factor
                # Assuming standard road capacity: 500 vehicles/hour
                # If demand > capacity, congestion increases
                is_metro = data.get('is_metro', False)
                
                if not is_metro and avg_demand > 0:
                    # Roads: higher demand = more congestion
                    demand_multiplier = 1.0 + (avg_demand / 1500.0) * 0.3  # Up to +30% baseline
                    self.G[u][v][key]['demand_congestion_factor'] = min(demand_multiplier, 1.5)
                    self.congestion_factors[(u, v, key)] *= demand_multiplier
                else:
                    # Metro: demand doesn't increase congestion (high capacity)
                    self.G[u][v][key]['demand_congestion_factor'] = 1.0
        
        if total_daily_trips > 0:
            print(f"   👥 Population-based Demand: {total_daily_trips:,} trips/day")
            print(f"   [CHART] Average demand per node: {total_daily_trips // max(1, len(self.G.nodes())):,} trips")
    
    def close_road(self, u: int, v: int, key: int = 0):
        """
        Close a road and propagate congestion upstream.
        
        This is the core of the "pressure model" - when a pipe bursts,
        water backs up into the pipes feeding it.
        
        Args:
            u, v, key: Edge identifier in the MultiDiGraph
        """
        edge_id = (u, v, key)
        
        if edge_id not in self.G.edges(keys=True):
            print(f"[WARNING]️  Edge {edge_id} not found in graph")
            return
        
        print(f"🚧 Closing road: {u} -> {v} (key={key})")
        
        # Mark as closed
        self.closed_edges.add(edge_id)
        self.current_travel_times[edge_id] = self.config.closed_road_penalty
        self.G[u][v][key]['current_travel_time'] = self.config.closed_road_penalty
        
        # Propagate congestion upstream
        self._propagate_congestion(u, v, key)
        
    def _propagate_congestion(self, u: int, v: int, key: int, 
                             depth: int = 0, multiplier: float = None):
        """
        Recursive ripple effect: congestion propagates upstream.
        
        This simulates what happens when 1,000 cars suddenly need to reroute.
        We don't track the cars - we track the mathematical consequence.
        
        Args:
            u, v, key: The blocked edge
            depth: Current recursion depth
            multiplier: Current congestion multiplier
        """
        if depth >= self.config.ripple_depth:
            return
        
        if multiplier is None:
            multiplier = self.config.base_congestion_multiplier
        
        # Find all edges feeding into node 'u'
        predecessors = list(self.G.predecessors(u))
        
        print(f"  {'  ' * depth}↑ Backlog at depth {depth}: {len(predecessors)} upstream edges")
        
        for pred in predecessors:
            # Get all edges from predecessor to u
            for pred_key in self.G[pred][u].keys():
                pred_edge = (pred, u, pred_key)
                
                # Skip if already closed
                if pred_edge in self.closed_edges:
                    continue
                
                # Apply congestion
                old_time = self.current_travel_times[pred_edge]
                new_time = old_time * multiplier
                
                self.current_travel_times[pred_edge] = new_time
                self.congestion_factors[pred_edge] = multiplier
                self.G[pred][u][pred_key]['current_travel_time'] = new_time
                
                print(f"  {'  ' * depth}   Edge {pred}->{u}: {old_time:.1f}m -> {new_time:.1f}m (×{multiplier:.1f})")
        
        # Decay and recurse
        new_multiplier = 1.0 + (multiplier - 1.0) * self.config.ripple_decay
        
        if new_multiplier > 1.1:  # Only propagate if significant
            for pred in predecessors:
                self._propagate_congestion(pred, u, 0, depth + 1, new_multiplier)
    
    def reopen_road(self, u: int, v: int, key: int = 0):
        """
        Reopen a closed road and allow traffic to recover.
        
        Args:
            u, v, key: Edge identifier
        """
        edge_id = (u, v, key)
        
        if edge_id not in self.closed_edges:
            print(f"[WARNING]️  Edge {edge_id} was not closed")
            return
        
        print(f"[OK] Reopening road: {u} -> {v} (key={key})")
        
        # Remove from closed set
        self.closed_edges.remove(edge_id)
        
        # Restore to base time (with some residual congestion)
        base_time = self.G[u][v][key]['base_travel_time']
        recovery_time = base_time * 1.2  # 20% residual congestion
        
        self.current_travel_times[edge_id] = recovery_time
        self.congestion_factors[edge_id] = 1.2
        self.G[u][v][key]['current_travel_time'] = recovery_time
    
    def simulate_random_events(self):
        """
        Add stochastic traffic events (accidents, construction, etc.)
        
        This adds realism to the training data.
        Metro lines are immune to road traffic events!
        """
        num_events = 0
        
        for edge_id in self.current_travel_times.keys():
            if edge_id in self.closed_edges:
                continue
            
            u, v, key = edge_id
            
            # Skip metro lines - they don't get stuck in traffic!
            if self.G[u][v][key].get('is_metro', False):
                continue
            
            if random.random() < self.config.random_event_probability:
                # Random slowdown (only for roads)
                slowdown = random.uniform(1.2, 2.0)
                old_time = self.current_travel_times[edge_id]
                new_time = old_time * slowdown
                
                self.current_travel_times[edge_id] = new_time
                self.G[u][v][key]['current_travel_time'] = new_time
                
                num_events += 1
        
        if num_events > 0:
            print(f"🎲 Random events: {num_events} edges affected")
    
    def apply_recovery(self):
        """
        Gradually reduce congestion over time (traffic clears).
        
        This simulates drivers adapting to the new conditions.
        Metro lines maintain constant speeds.
        """
        for edge_id in self.current_travel_times.keys():
            if edge_id in self.closed_edges:
                continue
            
            u, v, key = edge_id
            base_time = self.G[u][v][key]['base_travel_time']
            current_time = self.current_travel_times[edge_id]
            
            # Metro lines don't need recovery - they're always at base speed
            if self.G[u][v][key].get('is_metro', False):
                self.current_travel_times[edge_id] = base_time
                self.G[u][v][key]['current_travel_time'] = base_time
                self.congestion_factors[edge_id] = 1.0
                continue
            
            # Move towards base time for roads
            new_time = base_time + (current_time - base_time) * self.config.recovery_rate
            
            self.current_travel_times[edge_id] = new_time
            self.G[u][v][key]['current_travel_time'] = new_time
            
            # Update congestion factor
            self.congestion_factors[edge_id] = new_time / base_time
    
    def step(self, delta_time: Optional[float] = None):
        """
        Advance simulation by one time quantum.
        
        Args:
            delta_time: Time step in minutes (uses config default if None)
        """
        if delta_time is None:
            delta_time = self.config.time_quantum
        
        # Apply recovery (traffic disperses)
        self.apply_recovery()
        
        # Random events (accidents, etc.)
        self.simulate_random_events()
        
        # Update time
        self.simulation_time += delta_time
        
        # Take snapshot
        self._save_snapshot()
    
    def _save_snapshot(self):
        """Save current traffic state to history"""
        total_delay = sum(
            self.current_travel_times[(u, v, key)] - self.G[u][v][key]['base_travel_time']
            for u, v, key in self.current_travel_times.keys()
            if (u, v, key) not in self.closed_edges
        )
        
        # Capture population data
        node_populations = {node: self.G.nodes[node].get('population', 0) 
                          for node in self.G.nodes()}
        node_daily_trips = {node: self.G.nodes[node].get('daily_trips', 0) 
                          for node in self.G.nodes()}
        
        snapshot = TrafficSnapshot(
            timestamp=self.simulation_time,
            edge_travel_times=self.current_travel_times.copy(),
            edge_congestion=self.congestion_factors.copy(),
            closed_edges=self.closed_edges.copy(),
            total_network_delay=total_delay,
            node_populations=node_populations,
            node_daily_trips=node_daily_trips
        )
        
        self.snapshots.append(snapshot)
    
    def get_shortest_path(self, source: int, target: int) -> Optional[List[int]]:
        """
        Calculate shortest path using current traffic conditions.
        
        This uses Dijkstra's algorithm with current_travel_time as weights.
        The algorithm automatically avoids closed roads (infinite weight).
        
        Args:
            source: Start node
            target: End node
            
        Returns:
            List of nodes forming the path, or None if no path exists
        """
        try:
            path = nx.shortest_path(
                self.G, 
                source=source, 
                target=target, 
                weight='current_travel_time'
            )
            return path
        except nx.NetworkXNoPath:
            print(f"[WARNING]️  No path from {source} to {target}")
            return None
    
    def get_path_travel_time(self, path: List[int]) -> float:
        """
        Calculate total travel time for a given path.
        
        Args:
            path: List of nodes
            
        Returns:
            Total travel time in minutes
        """
        if not path or len(path) < 2:
            return 0.0
        
        total_time = 0.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Get minimum travel time across all edges between u and v
            min_time = float('inf')
            for key in self.G[u][v].keys():
                edge_time = self.current_travel_times.get((u, v, key), float('inf'))
                min_time = min(min_time, edge_time)
            
            total_time += min_time
        
        return total_time
    
    def export_training_data(self, filename: str = 'traffic_training_data.pkl'):
        """
        Export simulation history for GNN training.
        
        Args:
            filename: Output pickle file
        """
        training_data = {
            'snapshots': self.snapshots,
            'graph_info': {
                'num_nodes': self.G.number_of_nodes(),
                'num_edges': self.G.number_of_edges()
            },
            'config': self.config,
            'simulation_duration': self.simulation_time
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"[SAVE] Training data exported to {filename}")
        print(f"   Snapshots: {len(self.snapshots)}")
        print(f"   Duration: {self.simulation_time:.1f} minutes")
    
    def get_statistics(self) -> Dict:
        """Get current simulation statistics"""
        if not self.snapshots:
            return {}
        
        latest = self.snapshots[-1]
        
        # Separate metro and road statistics
        road_congestion = []
        metro_congestion = []
        num_metro_edges = 0
        num_road_edges = 0
        
        for edge_id, cf in self.congestion_factors.items():
            u, v, key = edge_id
            if self.G[u][v][key].get('is_metro', False):
                metro_congestion.append(cf)
                num_metro_edges += 1
            else:
                road_congestion.append(cf)
                num_road_edges += 1
        
        avg_congestion = np.mean(list(self.congestion_factors.values()))
        max_congestion = max(self.congestion_factors.values())
        
        num_congested = sum(1 for cf in road_congestion if cf > 1.5)
        
        stats = {
            'simulation_time': self.simulation_time,
            'total_network_delay': latest.total_network_delay,
            'closed_roads': len(self.closed_edges),
            'average_congestion': avg_congestion,
            'max_congestion': max_congestion,
            'congested_edges': num_congested,
            'total_edges': len(self.congestion_factors),
            'metro_edges': num_metro_edges,
            'road_edges': num_road_edges
        }
        
        # Add metro-specific stats if metro exists
        if metro_congestion:
            stats['metro_avg_congestion'] = np.mean(metro_congestion)
            stats['road_avg_congestion'] = np.mean(road_congestion) if road_congestion else 1.0
        
        return stats
    
    def print_statistics(self):
        """Print current traffic statistics"""
        stats = self.get_statistics()
        
        if not stats:
            print("No statistics available yet")
            return
        
        print("\n" + "="*60)
        print("[CHART] TRAFFIC STATISTICS")
        print("="*60)
        print(f"Simulation Time:     {stats['simulation_time']:.1f} minutes")
        print(f"Total Network Delay: {stats['total_network_delay']:.1f} minutes")
        print(f"Closed Roads:        {stats['closed_roads']}")
        print(f"Average Congestion:  {stats['average_congestion']:.2f}x")
        print(f"Max Congestion:      {stats['max_congestion']:.2f}x")
        print(f"Congested Edges:     {stats['congested_edges']}/{stats['total_edges']}")
        
        # Show metro vs road breakdown if metro exists
        if stats.get('metro_edges', 0) > 0:
            print("-" * 60)
            print("[METRO] METRO vs [ROAD] ROAD COMPARISON")
            print("-" * 60)
            print(f"Metro Edges:         {stats['metro_edges']} (Always clear! [OPEN])")
            print(f"Road Edges:          {stats['road_edges']}")
            print(f"Metro Congestion:    {stats.get('metro_avg_congestion', 1.0):.2f}x (Constant)")
            print(f"Road Congestion:     {stats.get('road_avg_congestion', 1.0):.2f}x")
            
            # Calculate benefit
            if stats.get('road_avg_congestion', 1.0) > 1.0:
                benefit = ((stats.get('road_avg_congestion', 1.0) - 1.0) / stats.get('road_avg_congestion', 1.0)) * 100
                print(f"Metro Advantage:     {benefit:.1f}% faster than roads! 🎉")
        
        print("="*60 + "\n")


def get_user_road_selection(sim: MacroscopicTrafficSimulator):
    """
    Interactive function to let user select a road to block.
    
    Args:
        sim: The traffic simulator instance
        
    Returns:
        Tuple of (u, v, key) representing the selected edge, or None if cancelled
    """
    edges = list(sim.G.edges(keys=True))
    
    if not edges:
        print("[ERROR] No edges available in the graph!")
        return None
    
    print("\n" + "="*60)
    print("🛣️  ROAD SELECTION MENU")
    print("="*60)
    
    # Show sample of available roads with details
    print(f"\n[CHART] Total roads in network: {len(edges)}")
    print("\nSample roads (showing first 20):")
    print("-" * 60)
    print(f"{'#':<5} {'From':<10} {'To':<10} {'Length(km)':<12} {'Speed(km/h)':<12}")
    print("-" * 60)
    
    display_edges = edges[:20]
    for idx, (u, v, key) in enumerate(display_edges, 1):
        edge_data = sim.G[u][v][key]
        length = edge_data.get('length', 1.0)
        speed = edge_data.get('speed_limit', 40.0)
        print(f"{idx:<5} {str(u):<10} {str(v):<10} {length:<12.2f} {speed:<12.1f}")
    
    print("-" * 60)
    print("\n💡 Options:")
    print("   1. Enter road number (1-20) from the list above")
    print("   2. Enter 'r' or 'random' for random road closure")
    print("   3. Enter 'custom' to specify source and target nodes")
    print("   4. Enter 'q' or 'quit' to skip road closure")
    print("="*60)
    
    while True:
        user_input = input("\n👉 Your choice: ").strip().lower()
        
        # Quit option
        if user_input in ['q', 'quit', 'exit', '']:
            print("⏭️  Skipping road closure...")
            return None
        
        # Random option
        if user_input in ['r', 'random']:
            selected = random.choice(edges)
            u, v, key = selected
            print(f"🎲 Randomly selected: Road {u} → {v} (key={key})")
            return selected
        
        # Custom option
        if user_input == 'custom':
            try:
                print("\n[TOOL] Custom Road Selection:")
                source = input("   Enter source node: ").strip()
                target = input("   Enter target node: ").strip()
                
                # Try to parse as integers
                try:
                    source = int(source)
                    target = int(target)
                except ValueError:
                    pass
                
                # Check if edge exists
                if sim.G.has_edge(source, target):
                    keys = list(sim.G[source][target].keys())
                    if len(keys) == 1:
                        selected = (source, target, keys[0])
                        print(f"[OK] Selected: Road {source} → {target}")
                        return selected
                    else:
                        print(f"   Multiple edges found ({len(keys)}). Using first one.")
                        selected = (source, target, keys[0])
                        return selected
                else:
                    print(f"[FAIL] No road exists from {source} to {target}")
                    print("   Try again or choose a different option.")
                    continue
            except Exception as e:
                print(f"[FAIL] Error: {e}")
                continue
        
        # Number option
        try:
            choice = int(user_input)
            if 1 <= choice <= len(display_edges):
                selected = display_edges[choice - 1]
                u, v, key = selected
                print(f"[OK] Selected: Road #{choice} ({u} → {v})")
                return selected
            else:
                print(f"[FAIL] Please enter a number between 1 and {len(display_edges)}")
        except ValueError:
            print("[ERROR] Invalid input. Please try again.")


def demo_simulation():
    """
    Demonstration of the macroscopic traffic simulation.
    
    This shows the pressure model in action without needing individual agents.
    Includes interactive user input for road closure.
    """
    print("[Loading city graph...]")
    
    try:
        G = nx.read_graphml('city_graph.graphml')
        print(f"[OK] Loaded graph with {G.number_of_nodes()} nodes")
    except FileNotFoundError:
        print("[WARNING] city_graph.graphml not found. Generating a simple test graph...")
        G = nx.MultiDiGraph()
        
        # Create a simple grid network for demo
        for i in range(10):
            for j in range(10):
                node_id = i * 10 + j
                G.add_node(node_id, x=j, y=i)
                
                # Connect to right neighbor
                if j < 9:
                    G.add_edge(node_id, node_id + 1, 
                              length=1.0, speed_limit=50.0)
                
                # Connect to bottom neighbor
                if i < 9:
                    G.add_edge(node_id, node_id + 10,
                              length=1.0, speed_limit=50.0)
    
    # Initialize simulator
    config = SimulationConfig(
        base_congestion_multiplier=3.0,
        ripple_decay=0.7,
        ripple_depth=3,
        random_event_probability=0.02
    )
    
    sim = MacroscopicTrafficSimulator(G, config)
    
    # Initial state
    print("\n[PIN] Initial State")
    sim.print_statistics()
    
    # Interactive road closure
    print("\n🚧 Road Closure Selection...")
    selected_edge = get_user_road_selection(sim)
    
    if selected_edge:
        u, v, key = selected_edge
        sim.close_road(u, v, key)
    
    # Run simulation for 30 minutes
    print("\n[TIME]️  Running simulation for 30 minutes...")
    for minute in range(30):
        sim.step(delta_time=1.0)
        
        if minute % 10 == 9:
            print(f"\n--- Minute {minute + 1} ---")
            sim.print_statistics()
    
    # Reopen the road
    if selected_edge:
        print(f"\n[OK] Reopening road after 30 minutes...")
        sim.reopen_road(u, v, key)
    
    # Continue simulation
    print("\n[TIME]️  Continuing for 20 more minutes...")
    for minute in range(20):
        sim.step(delta_time=1.0)
    
    print("\n[PIN] Final State")
    sim.print_statistics()
    
    # Export training data
    sim.export_training_data('demo_traffic_data.pkl')
    
    # Test pathfinding
    print("\n🗺️  Testing shortest path calculation...")
    nodes = list(G.nodes())
    if len(nodes) >= 2:
        source = random.choice(nodes)
        target = random.choice(nodes)
        
        path = sim.get_shortest_path(source, target)
        if path:
            travel_time = sim.get_path_travel_time(path)
            print(f"Path from {source} to {target}:")
            print(f"  Nodes: {len(path)}")
            print(f"  Travel time: {travel_time:.1f} minutes")


if __name__ == "__main__":
    demo_simulation()
