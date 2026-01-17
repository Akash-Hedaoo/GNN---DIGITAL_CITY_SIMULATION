"""
Phase 4.3: Multi-Road Failure Propagation Simulation
=====================================================
Demonstrate how congestion propagates through the road network
when multiple critical edges are blocked simultaneously.

STRICT RULES:
- No retraining
- No dataset/network modification
- Use trained ST-GCN for prediction
- Use SUMO for live simulation

Author: Traffic Digital Twin Project
"""

import os
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import traci
import sumolib

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NETWORK_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "networks", "pune.net.xml")
TRIPS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sumo", "trips.xml")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "stgcn", "traffic_twin_stgcn.pt")
EDGE_ERROR_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "datasets", "edge_error_stats.csv")
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "simulations", "failure_propagation_log.csv")

# Simulation parameters
BASELINE_STEPS = 500
TOTAL_STEPS = 2000
LOG_INTERVAL = 10

# ST-GCN parameters (SAME AS TRAINING)
WINDOW_SIZE = 3
PREDICTION_HORIZON = 5
HIDDEN_DIM = 64
NUM_STGCN_BLOCKS = 2

# =============================================================================
# Device Setup
# =============================================================================

def setup_device():
    """Setup compute device with CUDA if available."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Using GPU for inference")
    else:
        device = torch.device('cpu')
        print("[INFO] CUDA not available")
        print("[INFO] Using CPU for inference")
    return device

# =============================================================================
# Load Critical Edges from Error Stats
# =============================================================================

def load_critical_edges(path, top_n=5):
    """Load top N edges with highest prediction error."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Edge error stats not found at: {path}")
    
    df = pd.read_csv(path)
    top_edges = df.nlargest(top_n, 'mean_absolute_error')['edge_id'].tolist()
    
    print(f"[INFO] Top {top_n} critical edges:")
    for i, edge in enumerate(top_edges, 1):
        mae = df[df['edge_id'] == edge]['mean_absolute_error'].values[0]
        print(f"  {i}. {edge} (MAE: {mae:.4f})")
    
    return top_edges

# =============================================================================
# Adjacency Matrix Construction
# =============================================================================

def build_adjacency_matrix(network_path, edge_to_idx):
    """Build adjacency matrix from SUMO network file."""
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"Network file not found at: {network_path}")
    
    num_edges = len(edge_to_idx)
    
    tree = ET.parse(network_path)
    root = tree.getroot()
    
    junction_outgoing = defaultdict(list)
    
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')
        
        if edge_id.startswith(':'):
            continue
            
        from_junction = edge.get('from')
        to_junction = edge.get('to')
        
        if from_junction and to_junction and edge_id in edge_to_idx:
            junction_outgoing[from_junction].append(edge_id)
    
    adjacency = np.zeros((num_edges, num_edges), dtype=np.float32)
    
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')
        if edge_id.startswith(':') or edge_id not in edge_to_idx:
            continue
            
        to_junction = edge.get('to')
        if to_junction:
            for outgoing_edge in junction_outgoing.get(to_junction, []):
                if outgoing_edge in edge_to_idx:
                    i = edge_to_idx[edge_id]
                    j = edge_to_idx[outgoing_edge]
                    adjacency[i, j] = 1.0
    
    adjacency = adjacency + np.eye(num_edges, dtype=np.float32)
    
    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    
    adjacency_normalized = D_inv_sqrt @ adjacency @ D_inv_sqrt
    
    return torch.FloatTensor(adjacency_normalized)

# =============================================================================
# ST-GCN Model Architecture
# =============================================================================

class SpatialGraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return output


class STGCNBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_nodes=None):
        super().__init__()
        self.spatial_conv = SpatialGraphConv(in_features, hidden_features)
        self.temporal_conv = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, padding=1)
        self.output_conv = SpatialGraphConv(hidden_features, out_features)
        self.norm = nn.LayerNorm(out_features)
    
    def forward(self, x, adj):
        h = self.spatial_conv(x, adj)
        h = F.relu(h)
        h = h.transpose(1, 2)
        h = self.temporal_conv(h)
        h = h.transpose(1, 2)
        h = F.relu(h)
        h = self.output_conv(h, adj)
        h = self.norm(h)
        return F.relu(h)


class STGCN(nn.Module):
    def __init__(self, num_nodes, input_features, hidden_dim, num_blocks=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.blocks = nn.ModuleList()
        
        in_dim = input_features
        for i in range(num_blocks):
            self.blocks.append(STGCNBlock(in_dim, hidden_dim, hidden_dim))
            in_dim = hidden_dim
        
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, adj):
        h = x
        for block in self.blocks:
            h = block(h, adj)
        out = self.output_layer(h)
        out = out.squeeze(-1)
        return out

# =============================================================================
# SUMO Simulation Manager
# =============================================================================

class FailureSimulator:
    def __init__(self, net_file, trips_file):
        self.net_file = net_file
        self.trips_file = trips_file
        self.blocked_edges = set()
        
    def start(self):
        """Start SUMO simulation."""
        sumo_bin = sumolib.checkBinary("sumo")
        
        sumo_cmd = [
            sumo_bin,
            "-n", self.net_file,
            "-r", self.trips_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--ignore-route-errors", "true"
        ]
        
        traci.start(sumo_cmd)
        print("[INFO] SUMO simulation started")
        
    def step(self):
        """Advance one simulation step."""
        traci.simulationStep()
        
    def close(self):
        """Close simulation."""
        traci.close()
        print("[INFO] SUMO simulation closed")
        
    def block_edges(self, edge_ids):
        """Block multiple edges simultaneously."""
        for edge_id in edge_ids:
            try:
                traci.edge.setMaxSpeed(edge_id, 0.1)
                self.blocked_edges.add(edge_id)
                print(f"[INFO] Blocked edge: {edge_id}")
            except traci.exceptions.TraCIException as e:
                print(f"[WARN] Could not block edge {edge_id}: {e}")
                
    def get_edge_occupancy(self, edge_id):
        """Get occupancy for a specific edge."""
        try:
            return traci.edge.getLastStepOccupancy(edge_id)
        except:
            return 0.0
            
    def get_all_edges(self):
        """Get list of all edges (excluding internal)."""
        return [e for e in traci.edge.getIDList() if not e.startswith(':')]

# =============================================================================
# Temporal Buffer for ST-GCN Input
# =============================================================================

class TemporalBuffer:
    def __init__(self, edge_to_idx, window_size=3):
        self.edge_to_idx = edge_to_idx
        self.num_edges = len(edge_to_idx)
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        
    def update(self, edge_occupancies):
        """Add new timestep data to buffer."""
        state = np.zeros(self.num_edges, dtype=np.float32)
        for edge_id, occ in edge_occupancies.items():
            if edge_id in self.edge_to_idx:
                state[self.edge_to_idx[edge_id]] = occ
        self.buffer.append(state)
        
    def ready(self):
        """Check if buffer has enough data for prediction."""
        return len(self.buffer) == self.window_size
        
    def get_input(self):
        """Get input tensor for ST-GCN."""
        # Shape: (num_edges, window_size)
        data = np.stack(list(self.buffer), axis=1)
        # Add batch dimension: (1, num_edges, window_size)
        return torch.FloatTensor(data).unsqueeze(0)

# =============================================================================
# Main Simulation
# =============================================================================

def main():
    print("\n" + "="*60)
    print("Phase 4.3: Multi-Road Failure Propagation Simulation")
    print("="*60 + "\n")
    
    # Setup device
    device = setup_device()
    
    # Load critical edges
    critical_edges = load_critical_edges(EDGE_ERROR_PATH, top_n=5)
    
    # Select 3 edges for triangle-style blockage
    edges_to_block = critical_edges[:3]
    print(f"\n[INFO] Edges to block at step {BASELINE_STEPS}:")
    for edge in edges_to_block:
        print(f"  - {edge}")
    
    # Initialize simulator
    print(f"\n[INFO] Initializing SUMO simulation...")
    sim = FailureSimulator(NETWORK_PATH, TRIPS_PATH)
    sim.start()
    
    # Get all edges and create mapping
    all_edges = sim.get_all_edges()
    edge_to_idx = {edge: idx for idx, edge in enumerate(all_edges)}
    idx_to_edge = {idx: edge for edge, idx in edge_to_idx.items()}
    num_edges = len(edge_to_idx)
    
    print(f"[INFO] Total edges in simulation: {num_edges}")
    
    # Build adjacency matrix
    print(f"[INFO] Building adjacency matrix...")
    adj = build_adjacency_matrix(NETWORK_PATH, edge_to_idx)
    adj = adj.to(device)
    
    # Load ST-GCN model
    print(f"[INFO] Loading trained model...")
    model = STGCN(
        num_nodes=num_edges,
        input_features=WINDOW_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_STGCN_BLOCKS
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"[INFO] Model loaded successfully")
    
    # Initialize temporal buffer
    temporal_buffer = TemporalBuffer(edge_to_idx, WINDOW_SIZE)
    
    # Results collection
    results = []
    baseline_occupancies = {}
    
    # Metrics tracking
    congestion_start_step = None
    affected_edges_count = 0
    max_congestion_rise = 0.0
    
    print(f"\n[INFO] Starting simulation...")
    print(f"[INFO] Baseline phase: steps 0-{BASELINE_STEPS}")
    print(f"[INFO] Blockage phase: steps {BASELINE_STEPS}-{TOTAL_STEPS}")
    
    try:
        for step in range(TOTAL_STEPS):
            sim.step()
            
            # Block edges at step 500
            if step == BASELINE_STEPS:
                print(f"\n[INFO] Step {step}: Blocking {len(edges_to_block)} edges simultaneously!")
                sim.block_edges(edges_to_block)
                # Store baseline occupancies just before blockage
                for edge in all_edges:
                    baseline_occupancies[edge] = sim.get_edge_occupancy(edge)
            
            # Log every 10 steps
            if step % LOG_INTERVAL == 0:
                # Collect current occupancies
                current_occupancies = {}
                for edge in all_edges:
                    current_occupancies[edge] = sim.get_edge_occupancy(edge)
                
                # Update temporal buffer
                temporal_buffer.update(current_occupancies)
                
                # Run ST-GCN prediction if buffer is ready
                predictions = None
                if temporal_buffer.ready():
                    with torch.no_grad():
                        input_tensor = temporal_buffer.get_input().to(device)
                        pred_output = model(input_tensor, adj)
                        predictions = pred_output.squeeze(0).cpu().numpy()
                
                # Log data for each edge
                for edge_id in all_edges:
                    edge_idx = edge_to_idx[edge_id]
                    actual_occ = current_occupancies[edge_id]
                    pred_occ = predictions[edge_idx] if predictions is not None else np.nan
                    is_blocked = edge_id in sim.blocked_edges
                    
                    results.append({
                        'timestep': step,
                        'edge_id': edge_id,
                        'actual_occupancy': actual_occ,
                        'predicted_occupancy': pred_occ,
                        'is_blocked': is_blocked
                    })
                    
                    # Track congestion propagation after blockage
                    if step > BASELINE_STEPS and edge_id in baseline_occupancies:
                        baseline = baseline_occupancies[edge_id]
                        rise = actual_occ - baseline
                        
                        if rise > 0.1:  # Significant congestion rise
                            if congestion_start_step is None:
                                congestion_start_step = step
                            affected_edges_count += 1
                            
                        if rise > max_congestion_rise:
                            max_congestion_rise = rise
                
                # Progress update every 100 steps
                if step % 100 == 0:
                    print(f"[INFO] Step {step}/{TOTAL_STEPS} completed")
                    
    except Exception as e:
        print(f"[ERROR] Simulation error: {e}")
        raise
    finally:
        sim.close()
    
    # Save results
    print(f"\n[INFO] Saving results...")
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[INFO] Results saved to: {OUTPUT_CSV_PATH}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("FAILURE PROPAGATION ANALYSIS")
    print("="*60)
    
    time_to_spread = (congestion_start_step - BASELINE_STEPS) if congestion_start_step else "N/A"
    print(f"\n--- Congestion Propagation Metrics ---")
    print(f"  Time to congestion spread:  {time_to_spread} steps")
    print(f"  Number of affected edges:   {affected_edges_count}")
    print(f"  Max predicted congestion rise: {max_congestion_rise:.6f}")
    
    print(f"\n--- Simulation Summary ---")
    print(f"  Total steps simulated:      {TOTAL_STEPS}")
    print(f"  Edges blocked at step {BASELINE_STEPS}:  {len(edges_to_block)}")
    print(f"  Total log entries:          {len(results)}")
    
    print("\n" + "="*60)
    print("\nPHASE 4.3 FAILURE PROPAGATION COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
