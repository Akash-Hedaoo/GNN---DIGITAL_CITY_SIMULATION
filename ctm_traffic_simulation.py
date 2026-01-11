"""
Cell Transmission Model (CTM) for Traffic Simulation
====================================================

Implementation of Daganzo's Cell Transmission Model (1994) for macroscopic
traffic flow simulation on a road network.

Theory Background:
-----------------
The CTM discretizes roads into cells and models traffic flow based on:
1. Conservation of vehicles (mass balance)
2. Fundamental Diagram (flow-density relationship)
3. FIFO principle (First-In-First-Out)

Key Equations:
-------------
1. Flow capacity: Q = min(Demand_i, Supply_{i+1})
2. Demand: D(n) = min(v_free * n, q_max)
3. Supply: S(n) = w * (n_jam - n)
4. Update: n_i(t+1) = n_i(t) + (Q_in - Q_out) * Δt / L

Fundamental Diagram:
------------------
- Free flow: q = v_free * n (for n < n_crit)
- Congested: q = w * (n_jam - n) (for n > n_crit)

Author: Digital Twin City Simulation
Date: December 2025
"""

import networkx as nx
import numpy as np
import random
import pickle
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
import time


@dataclass
class CTMCell:
    """Represents a single cell in the CTM discretization"""
    edge_id: Tuple[int, int, int]  # (u, v, key)
    cell_index: int
    length_km: float
    num_lanes: int
    
    # State variables
    density: float = 0.0  # vehicles/km/lane
    flow: float = 0.0  # vehicles/hour
    
    # Parameters
    v_free: float = 60.0  # km/h
    w: float = 20.0  # backward wave speed km/h
    n_jam: float = 150.0  # vehicles/km/lane
    q_max: float = 2000.0  # vehicles/hour/lane
    n_crit: float = field(init=False)
    
    def __post_init__(self):
        """Calculate critical density"""
        self.n_crit = self.q_max / (self.v_free + self.w)
    
    def get_supply(self) -> float:
        """Calculate cell supply (capacity to receive vehicles)"""
        if self.density >= self.n_jam:
            return 0.0
        return self.w * (self.n_jam - self.density) * self.num_lanes
    
    def get_demand(self) -> float:
        """Calculate cell demand (vehicles wanting to exit)"""
        if self.density <= 0:
            return 0.0
        return min(self.v_free * self.density, self.q_max) * self.num_lanes
    
    def get_flow_capacity(self) -> float:
        """Calculate flow using fundamental diagram"""
        if self.density <= 0:
            return 0.0
        if self.density < self.n_crit:
            flow = self.v_free * self.density * self.num_lanes
        else:
            flow = self.w * (self.n_jam - self.density) * self.num_lanes
        return max(0.0, min(flow, self.q_max * self.num_lanes))
    
    def update_density(self, inflow: float, outflow: float, delta_t: float):
        """Update density based on conservation equation"""
        delta_n = (inflow - outflow) * delta_t / (self.length_km * self.num_lanes)
        self.density = max(0.0, min(self.density + delta_n, self.n_jam))
        self.flow = self.get_flow_capacity()
    
    def get_congestion_level(self) -> float:
        """Get normalized congestion (0.0 to 1.0, where 1.0 = 100% of jam density)"""
        return min(1.0, self.density / self.n_jam) if self.n_jam > 0 else 0.0
    
    def get_travel_time(self) -> float:
        """Calculate travel time in minutes"""
        if self.density < self.n_crit:
            speed = self.v_free
        else:
            speed = self.w * (self.n_jam - self.density) / self.density if self.density > 0 else self.v_free
        speed = max(1.0, speed)
        return (self.length_km / speed) * 60.0


@dataclass
class CTMSnapshot:
    """CTM traffic state snapshot"""
    timestamp: float
    cell_densities: Dict[Tuple[int, int, int, int], float]
    cell_flows: Dict[Tuple[int, int, int, int], float]
    edge_travel_times: Dict[Tuple[int, int, int], float]
    edge_congestion: Dict[Tuple[int, int, int], float]
    total_network_delay: float
    total_vehicles: int
    closed_edges: Set[Tuple[int, int, int]] = field(default_factory=set)


@dataclass
class CTMConfig:
    cell_length_km: float = 500.0  # Cell length in same units as graph (meters)
    time_step_hours: float = 1.0 / 60.0  # 1 minute
    
    # Traffic parameters (adjusted for meters)
    default_free_flow_speed: float = 60000.0  # 60 km/h in meters/hour
    default_backward_wave_speed: float = 20000.0  # 20 km/h in meters/hour
    default_jam_density: float = 0.15  # vehicles/meter/lane
    default_max_flow: float = 2000.0  # vehicles/hour/lane
    
    # Metro parameters
    metro_free_flow_speed: float = 80000.0  # 80 km/h in meters/hour
    metro_jam_density: float = 0.3  # vehicles/meter/lane
    metro_max_flow: float = 5000.0  # vehicles/hour/lane
    
    # Initial conditions
    initial_density_ratio: float = 0.15  # 15% of jam density
    demand_generation_rate: float = 50.0  # vehicles/hour entering network
    closed_road_blocking: bool = True
    
    # Performance options
    fast_mode: bool = True  # Skip topology cache building for faster init


class CTMTrafficSimulator:
    """
    Cell Transmission Model traffic simulator
    
    Implements Daganzo's CTM with:
    - Cell discretization
    - Fundamental diagram
    - Supply-demand flow calculations
    - Queue formation and dissipation
    """
    
    def __init__(self, graph: nx.MultiDiGraph, config: Optional[CTMConfig] = None):
        """Initialize CTM simulator"""
        start_time = time.time()
        print("[CTM] Starting initialization...", flush=True)
        
        # Use reference instead of copy - much faster
        # We don't modify graph structure, only read it
        if isinstance(graph, nx.MultiDiGraph):
            self.G = graph
        else:
            self.G = nx.MultiDiGraph(graph)
        
        self.config = config or CTMConfig()
        self.cells: Dict[Tuple[int, int, int], List[CTMCell]] = {}
        self.closed_edges: Set[Tuple[int, int, int]] = set()
        self.snapshots: List[CTMSnapshot] = []
        self.simulation_time: float = 0.0
        self.total_vehicles: int = 0
        
        # Performance optimization: cache graph topology
        self._successor_cache: Dict[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]] = {}
        self._predecessor_cache: Dict[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]] = {}
        
        # Ultra-fast cell lookup map (cell_id -> cell object)
        self._cell_map: Dict[Tuple[int, int, int, int], CTMCell] = {}
        
        print(f"[CTM] Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges", flush=True)
        
        self._discretize_network()
        self._initialize_traffic()
        
        # Only build caches if not in fast mode
        if not self.config.fast_mode:
            self._build_topology_cache()
        else:
            print("[CTM] Fast mode: Skipping topology cache (will build on-demand)", flush=True)
        
        self._build_cell_map()
        
        elapsed = time.time() - start_time
        print(f"[CTM] Initialized in {elapsed:.2f}s", flush=True)
        print(f"   Total Cells: {sum(len(cells) for cells in self.cells.values())}", flush=True)
    
    def _discretize_network(self):
        """Discretize each edge into cells"""
        print("[CTM] Discretizing network into cells...", flush=True)
        start = time.time()
        total_cells = 0
        metro_count = 0
        road_count = 0
        total_edges = self.G.number_of_edges()
        
        for idx, (u, v, key, data) in enumerate(self.G.edges(keys=True, data=True)):
            length_km = data.get('length', 1.0)
            speed_kmh = data.get('speed_limit', self.config.default_free_flow_speed)
            is_metro = data.get('is_metro', False) or data.get('transport_mode') == 'metro'
            num_lanes = data.get('lanes', 2)
            
            num_cells = max(1, int(np.ceil(length_km / self.config.cell_length_km)))
            cell_length = length_km / num_cells
            
            edge_cells = []
            for cell_idx in range(num_cells):
                if is_metro:
                    cell = CTMCell(
                        edge_id=(u, v, key),
                        cell_index=cell_idx,
                        length_km=cell_length,
                        num_lanes=num_lanes,
                        v_free=self.config.metro_free_flow_speed,
                        w=self.config.default_backward_wave_speed,
                        n_jam=self.config.metro_jam_density,
                        q_max=self.config.metro_max_flow
                    )
                else:
                    cell = CTMCell(
                        edge_id=(u, v, key),
                        cell_index=cell_idx,
                        length_km=cell_length,
                        num_lanes=num_lanes,
                        v_free=speed_kmh,
                        w=self.config.default_backward_wave_speed,
                        n_jam=self.config.default_jam_density,
                        q_max=self.config.default_max_flow
                    )
                
                edge_cells.append(cell)
                total_cells += 1
            
            self.cells[(u, v, key)] = edge_cells
            self.G[u][v][key]['num_cells'] = num_cells
            self.G[u][v][key]['is_metro'] = is_metro
            
            if is_metro:
                metro_count += 1
            else:
                road_count += 1
        
        elapsed = time.time() - start
        print(f"   Discretization done in {elapsed:.2f}s", flush=True)
        print(f"   Metro edges: {metro_count}", flush=True)
        print(f"   Road edges: {road_count}", flush=True)
        print(f"   Total cells created: {total_cells}", flush=True)
    
    def _initialize_traffic(self):
        """Initialize with baseline vehicle density"""
        print("[CTM] Initializing traffic density...", flush=True)
        start = time.time()
        vehicles_added = 0
        
        for edge_id, edge_cells in self.cells.items():
            u, v, key = edge_id
            is_metro = self.G[u][v][key].get('is_metro', False)
            initial_ratio = self.config.initial_density_ratio
            
            if is_metro:
                initial_ratio *= 0.5
            
            for cell in edge_cells:
                cell.density = cell.n_jam * initial_ratio * random.uniform(0.5, 1.5)
                cell.flow = cell.get_flow_capacity()
                vehicles_added += int(cell.density * cell.length_km * cell.num_lanes)
        
        self.total_vehicles = vehicles_added
        elapsed = time.time() - start
        print(f"   Traffic initialized in {elapsed:.2f}s", flush=True)
        print(f"   Initial vehicles: {self.total_vehicles:,}", flush=True)
    
    def _build_topology_cache(self):
        """Pre-compute successor/predecessor relationships for O(1) lookups"""
        print("[CTM] Building topology cache...", flush=True)
        cache_start = time.time()
        
        # OPTIMIZATION: Build node adjacency maps once to avoid repeated NetworkX queries
        # Pre-compute all successors and predecessors at the node level
        node_successors = {}
        node_predecessors = {}
        
        print("   Building node adjacency maps...", flush=True)
        nodes_list = list(self.G.nodes())
        for idx, node in enumerate(nodes_list):
            node_successors[node] = list(self.G.successors(node))
            node_predecessors[node] = list(self.G.predecessors(node))
        
        print(f"   Cached adjacency for {len(node_successors)} nodes", flush=True)
        print("   Building cell-level connections...", flush=True)
        
        # Now iterate through cells and use the pre-computed maps
        cell_count = 0
        total_cells = sum(len(cells) for cells in self.cells.values())
        
        for edge_id, edge_cells in self.cells.items():
            u, v, key = edge_id
            num_cells = len(edge_cells)
            
            for i in range(num_cells):
                cell_id = (u, v, key, i)
                
                # Cache successors
                successors = []
                if i < num_cells - 1:
                    # Internal successor within same edge
                    successors.append((u, v, key, i + 1))
                else:
                    # End of edge - connect to next edges using pre-computed adjacency
                    for next_node in node_successors.get(v, []):
                        for next_key in self.G[v][next_node].keys():
                            next_edge_id = (v, next_node, next_key)
                            if next_edge_id in self.cells:
                                successors.append((v, next_node, next_key, 0))
                self._successor_cache[cell_id] = successors
                
                # Cache predecessors
                predecessors = []
                if i > 0:
                    # Internal predecessor within same edge
                    predecessors.append((u, v, key, i - 1))
                else:
                    # Start of edge - connect to previous edges using pre-computed adjacency
                    for prev_node in node_predecessors.get(u, []):
                        for prev_key in self.G[prev_node][u].keys():
                            prev_edge_id = (prev_node, u, prev_key)
                            if prev_edge_id in self.cells:
                                last_idx = len(self.cells[prev_edge_id]) - 1
                                predecessors.append((prev_node, u, prev_key, last_idx))
                self._predecessor_cache[cell_id] = predecessors
                cell_count += 1
        
        cache_elapsed = time.time() - cache_start
        total_cached = len(self._successor_cache)
        print(f"   Cached {total_cached:,} cell connections in {cache_elapsed:.2f}s", flush=True)
    
    def _build_cell_map(self):
        """Build fast cell lookup map for O(1) access"""
        print("[CTM] Building cell lookup map...", flush=True)
        for edge_id, edge_cells in self.cells.items():
            u, v, key = edge_id
            for i, cell in enumerate(edge_cells):
                self._cell_map[(u, v, key, i)] = cell
        print(f"   Mapped {len(self._cell_map):,} cells", flush=True)
    
    def close_road(self, u: int, v: int, key: int = 0):
        """Close a road segment"""
        edge_id = (u, v, key)
        if edge_id not in self.cells:
            print(f"[WARNING] Edge {edge_id} not found", flush=True)
            return
        
        print(f"🚧 Closing road: {u} -> {v} (key={key})")
        self.closed_edges.add(edge_id)
        
        if self.config.closed_road_blocking:
            for cell in self.cells[edge_id]:
                cell.density = cell.n_jam
                cell.flow = 0.0
    
    def reopen_road(self, u: int, v: int, key: int = 0):
        """Reopen a closed road"""
        edge_id = (u, v, key)
        if edge_id not in self.closed_edges:
            print(f"[WARNING] Edge {edge_id} was not closed")
            return
        
        print(f"✅ Reopening road: {u} -> {v} (key={key})")
        self.closed_edges.remove(edge_id)
        
        for cell in self.cells[edge_id]:
            cell.density = cell.n_crit * 0.8
            cell.flow = cell.get_flow_capacity()
    
    def step(self, delta_t: Optional[float] = None, save_snapshot: bool = True, show_progress: bool = False):
        """Advance simulation by one time step.
        Setting save_snapshot=False skips expensive snapshot capture, which speeds up
        batched simulations where intermediate states are not needed."""
        if delta_t is None:
            delta_t = self.config.time_step_hours
        
        step_start = time.time() if show_progress else None
        if show_progress:
            step_num = len(self.snapshots) + 1
            print(f"[CTM] Step {step_num}: Flows -> Densities -> Demand")
        
        flows = self._calculate_flows()
        self._update_densities(flows, delta_t)
        self._generate_demand(delta_t)
        self.simulation_time += delta_t
        
        if save_snapshot:
            self._save_snapshot()
        
        if show_progress:
            elapsed = time.time() - step_start
            print(f"[CTM] Step {len(self.snapshots)} done in {elapsed:.2f}s")
    
    def _calculate_flows(self) -> Dict[Tuple[int, int, int, int], float]:
        """Calculate flows between cells using CTM equations (ULTRA-OPTIMIZED)"""
        flows = {}
        
        # Lazy cache building in fast mode - build on first step
        if self.config.fast_mode and not self._successor_cache:
            print("[CTM] Building topology cache on first step...")
            self._build_topology_cache()
        
        # Single pass: compute demands, supplies, and flows together
        for cell_id, cell in self._cell_map.items():
            edge_id = cell_id[:3]
            
            # Fast path for closed edges
            if edge_id in self.closed_edges:
                flows[cell_id] = 0.0
                continue
            
            demand = cell.get_demand()
            
            # Use cached successors
            successors = self._successor_cache.get(cell_id, [])
            if successors:
                # Sum supplies from all successors (direct cell access)
                total_supply = 0.0
                for succ_id in successors:
                    succ_edge = succ_id[:3]
                    if succ_edge not in self.closed_edges:
                        succ_cell = self._cell_map.get(succ_id)
                        if succ_cell:
                            total_supply += succ_cell.get_supply()
                flows[cell_id] = min(demand, total_supply) if total_supply > 0 else 0.0
            else:
                # Sink cell (no successors)
                flows[cell_id] = demand
        
        return flows
    
    def _update_densities(self, flows: Dict, delta_t: float):
        """Update cell densities based on flows (ULTRA-OPTIMIZED)"""
        # Pre-compute flow splits ONCE to avoid repeated calculations
        flow_splits = {}
        for cell_id, pred_successors in self._successor_cache.items():
            if pred_successors:
                num_open = sum(1 for s in pred_successors if s[:3] not in self.closed_edges)
                if num_open > 0:
                    flow_splits[cell_id] = num_open
        
        # Single pass through all cells using cell map
        for cell_id, cell in self._cell_map.items():
            # Calculate inflow using cached predecessors
            predecessors = self._predecessor_cache.get(cell_id, [])
            inflow = 0.0
            
            for pred_id in predecessors:
                if pred_id[:3] not in self.closed_edges:
                    pred_flow = flows.get(pred_id, 0.0)
                    num_successors = flow_splits.get(pred_id, 1)
                    inflow += pred_flow / num_successors
            
            outflow = flows.get(cell_id, 0.0)
            cell.update_density(inflow, outflow, delta_t)
    
    def _generate_demand(self, delta_t: float):
        """Generate new vehicles entering network (ULTRA-OPTIMIZED)"""
        vehicles_to_add = int(self.config.demand_generation_rate * delta_t)
        if vehicles_to_add == 0:
            return
        
        # Cache source nodes AND their edges to avoid repeated lookups
        if not hasattr(self, '_source_edges_cache'):
            source_nodes = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]
            if not source_nodes:
                avg_in = sum(self.G.in_degree(n) for n in self.G.nodes()) / self.G.number_of_nodes()
                source_nodes = [n for n in self.G.nodes() if self.G.in_degree(n) < avg_in * 0.5]
            
            # Pre-compute source edges
            self._source_edges_cache = []
            for source in source_nodes:
                successors = list(self.G.successors(source))
                if successors:
                    target = successors[0]
                    for key in self.G[source][target].keys():
                        edge_id = (source, target, key)
                        if edge_id in self.cells:
                            self._source_edges_cache.append(edge_id)
                            break
        
        if not self._source_edges_cache:
            return
        
        # Fast random selection from cached edges
        num_edges = len(self._source_edges_cache)
        selected_indices = np.random.randint(0, num_edges, size=vehicles_to_add)
        unique_indices, counts = np.unique(selected_indices, return_counts=True)
        
        # Add vehicles in batches
        for idx, count in zip(unique_indices, counts):
            edge_id = self._source_edges_cache[idx]
            if edge_id not in self.closed_edges:
                first_cell = self.cells[edge_id][0]
                if first_cell.density < first_cell.n_jam * 0.95:
                    density_per_vehicle = 1.0 / (first_cell.length_km * first_cell.num_lanes)
                    first_cell.density += density_per_vehicle * count
                    self.total_vehicles += int(count)
    
    def _save_snapshot(self):
        """Save current state (ULTRA-OPTIMIZED - minimal computation)"""
        cell_densities = {}
        cell_flows = {}
        edge_travel_times = {}
        edge_congestion = {}
        total_delay = 0.0
        
        # Ultra-fast: single pass, direct calculations
        for edge_id, edge_cells in self.cells.items():
            u, v, key = edge_id
            
            # Direct aggregation without intermediate arrays
            edge_tt = 0.0
            edge_cong_sum = 0.0
            num_cells = len(edge_cells)
            
            for i, cell in enumerate(edge_cells):
                cell_densities[(u, v, key, i)] = cell.density
                cell_flows[(u, v, key, i)] = cell.flow
                tt = cell.get_travel_time()
                edge_tt += tt
                edge_cong_sum += cell.get_congestion_level()
            
            edge_travel_times[edge_id] = edge_tt
            edge_congestion[edge_id] = edge_cong_sum / num_cells if num_cells > 0 else 0.0
            
            # Calculate delay
            length = self.G[u][v][key].get('length', 1.0)
            speed = self.G[u][v][key].get('speed_limit', 60.0)
            free_flow_time = (length / speed) * 60.0
            total_delay += max(0.0, edge_tt - free_flow_time)
        
        # Create snapshot WITHOUT copying
        snapshot = CTMSnapshot(
            timestamp=self.simulation_time * 60.0,
            cell_densities=cell_densities,
            cell_flows=cell_flows,
            edge_travel_times=edge_travel_times,
            edge_congestion=edge_congestion,
            total_network_delay=total_delay,
            total_vehicles=self.total_vehicles,
            closed_edges=set(self.closed_edges)
        )
        
        self.snapshots.append(snapshot)
    
    def get_statistics(self) -> Dict:
        """Get current simulation statistics (OPTIMIZED - uses cached snapshot data)"""
        if not self.snapshots:
            return {}
        
        latest = self.snapshots[-1]
        
        # Use edge-level congestion from snapshot (already computed during snapshot)
        congestion_values = list(latest.edge_congestion.values())
        
        # Calculate proper congestion as percentage (0.0 to 1.0)
        avg_congestion = np.mean(congestion_values) if congestion_values else 0.0
        max_congestion = max(congestion_values) if congestion_values else 0.0
        
        # Ensure values are capped at 1.0 (100%)
        avg_congestion = min(1.0, avg_congestion)
        max_congestion = min(1.0, max_congestion)
        
        # Split metro vs road congestion by edge type (cached from graph)
        road_congestion = []
        metro_congestion = []
        for edge_id, cong_val in latest.edge_congestion.items():
            u, v, key = edge_id
            is_metro = self.G[u][v][key].get('is_metro', False)
            if is_metro:
                metro_congestion.append(cong_val)
            else:
                road_congestion.append(cong_val)
        
        stats = {
            'simulation_time': latest.timestamp,
            'total_network_delay': latest.total_network_delay,
            'closed_roads': len(self.closed_edges),
            'average_congestion': avg_congestion,
            'max_congestion': max_congestion,
            'total_vehicles': self.total_vehicles,
            'road_edges': len(road_congestion),
            'metro_edges': len(metro_congestion),
            'model_type': 'CTM'
        }
        
        if metro_congestion:
            stats['metro_avg_congestion'] = min(1.0, np.mean(metro_congestion))
            stats['road_avg_congestion'] = min(1.0, np.mean(road_congestion)) if road_congestion else 0.0
        
        return stats
    
    def print_statistics(self):
        """Print statistics"""
        stats = self.get_statistics()
        if not stats:
            print("No statistics available")
            return
        
        print("\n" + "="*70)
        print("[CTM] CELL TRANSMISSION MODEL - TRAFFIC STATISTICS")
        print("="*70)
        print(f"Simulation Time:      {stats['simulation_time']:.1f} minutes")
        print(f"Total Vehicles:       {stats['total_vehicles']:,}")
        print(f"Total Network Delay:  {stats['total_network_delay']:.1f} minutes")
        print(f"Closed Roads:         {stats['closed_roads']}")
        print(f"Average Congestion:   {stats['average_congestion']:.2%}")
        print(f"Max Congestion:       {stats['max_congestion']:.2%}")
        
        if stats.get('metro_cells', 0) > 0:
            print("-" * 70)
            print("🚇 Metro Cells:          ", stats['metro_cells'])
            print("🚗 Road Cells:           ", stats['road_cells'])
            print("🚇 Metro Congestion:     ", f"{stats.get('metro_avg_congestion', 0.0):.2%}")
            print("🚗 Road Congestion:      ", f"{stats.get('road_avg_congestion', 0.0):.2%}")
        
        print("="*70 + "\n")
    
    def export_training_data(self, filename: str = 'ctm_training_data.pkl'):
        """Export for GNN training"""
        training_data = {
            'snapshots': self.snapshots,
            'graph_info': {
                'num_nodes': self.G.number_of_nodes(),
                'num_edges': self.G.number_of_edges(),
                'total_cells': sum(len(cells) for cells in self.cells.values())
            },
            'config': self.config,
            'simulation_duration': self.simulation_time * 60.0,
            'model_type': 'CTM'
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"[SAVE] CTM data exported to {filename}")
        print(f"   Snapshots: {len(self.snapshots)}")
    
    def get_edge_travel_time(self, u: int, v: int, key: int = 0) -> float:
        """Get travel time for edge"""
        edge_id = (u, v, key)
        if edge_id not in self.cells or edge_id in self.closed_edges:
            return float('inf')
        return sum(cell.get_travel_time() for cell in self.cells[edge_id])
    
    def get_shortest_path(self, source: int, target: int) -> Optional[List[int]]:
        """Calculate shortest path with current CTM travel times"""
        for edge_id, edge_cells in self.cells.items():
            u, v, key = edge_id
            travel_time = sum(cell.get_travel_time() for cell in edge_cells)
            self.G[u][v][key]['current_travel_time'] = travel_time
        
        try:
            return nx.shortest_path(self.G, source=source, target=target, weight='current_travel_time')
        except nx.NetworkXNoPath:
            return None
