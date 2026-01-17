"""
Dashboard Backend Worker
========================
Runs SUMO simulation + ST-GCN inference in background.
Called as subprocess from dashboard.

Author: Traffic Digital Twin Project
"""

import os
import sys
import json
import argparse
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "simulations")

# Simulation parameters
BASELINE_STEPS = 500
TOTAL_STEPS = 2000
LOG_INTERVAL = 10

# ST-GCN parameters
WINDOW_SIZE = 3
PREDICTION_HORIZON = 5
HIDDEN_DIM = 64
NUM_STGCN_BLOCKS = 2

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
# Adjacency Matrix
# =============================================================================

def build_adjacency_matrix(edge_to_idx):
    """Build adjacency matrix from SUMO network file."""
    num_edges = len(edge_to_idx)
    
    tree = ET.parse(NETWORK_PATH)
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
# Temporal Buffer
# =============================================================================

class TemporalBuffer:
    def __init__(self, edge_to_idx, window_size=3):
        self.edge_to_idx = edge_to_idx
        self.num_edges = len(edge_to_idx)
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        
    def update(self, edge_occupancies):
        state = np.zeros(self.num_edges, dtype=np.float32)
        for edge_id, occ in edge_occupancies.items():
            if edge_id in self.edge_to_idx:
                state[self.edge_to_idx[edge_id]] = occ
        self.buffer.append(state)
        
    def ready(self):
        return len(self.buffer) == self.window_size
        
    def get_input(self):
        data = np.stack(list(self.buffer), axis=1)
        return torch.FloatTensor(data).unsqueeze(0)

# =============================================================================
# Main Worker
# =============================================================================

def run_analysis(blocked_edges_list, output_file="failure_propagation_log.csv"):
    """Run SUMO simulation with blocked edges and ST-GCN predictions."""
    
    print(f"[BACKEND] Starting analysis with {len(blocked_edges_list)} blocked edges")
    print(f"[BACKEND] Blocked edges: {blocked_edges_list}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[BACKEND] Using device: {device}")
    
    # Start SUMO
    sumo_bin = sumolib.checkBinary("sumo")
    sumo_cmd = [
        sumo_bin,
        "-n", NETWORK_PATH,
        "-r", TRIPS_PATH,
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
        "--time-to-teleport", "-1",
        "--ignore-route-errors", "true"
    ]
    
    traci.start(sumo_cmd)
    print("[BACKEND] SUMO started")
    
    try:
        # Get all edges and create mapping
        all_edges = [e for e in traci.edge.getIDList() if not e.startswith(':')]
        edge_to_idx = {edge: idx for idx, edge in enumerate(all_edges)}
        idx_to_edge = {idx: edge for edge, idx in edge_to_idx.items()}
        num_edges = len(edge_to_idx)
        
        print(f"[BACKEND] Total edges: {num_edges}")
        
        # Build adjacency matrix
        adj = build_adjacency_matrix(edge_to_idx).to(device)
        
        # Load model
        model = STGCN(
            num_nodes=num_edges,
            input_features=WINDOW_SIZE,
            hidden_dim=HIDDEN_DIM,
            num_blocks=NUM_STGCN_BLOCKS
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        print("[BACKEND] Model loaded")
        
        # Initialize temporal buffer
        temporal_buffer = TemporalBuffer(edge_to_idx, WINDOW_SIZE)
        
        # Track blocked edges
        blocked_set = set()
        
        # Results collection
        results = []
        
        print(f"[BACKEND] Running simulation for {TOTAL_STEPS} steps...")
        
        for step in range(TOTAL_STEPS):
            traci.simulationStep()
            
            # Block edges at baseline step
            if step == BASELINE_STEPS:
                print(f"[BACKEND] Step {step}: Blocking edges")
                for edge_id in blocked_edges_list:
                    try:
                        traci.edge.setMaxSpeed(edge_id, 0.1)
                        blocked_set.add(edge_id)
                    except:
                        print(f"[BACKEND] Warning: Could not block {edge_id}")
            
            # Log every LOG_INTERVAL steps
            if step % LOG_INTERVAL == 0:
                # Collect current occupancies
                current_occupancies = {}
                for edge in all_edges:
                    try:
                        current_occupancies[edge] = traci.edge.getLastStepOccupancy(edge)
                    except:
                        current_occupancies[edge] = 0.0
                
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
                    is_blocked = edge_id in blocked_set
                    
                    results.append({
                        'timestep': step,
                        'edge_id': edge_id,
                        'actual_occupancy': actual_occ,
                        'predicted_occupancy': pred_occ,
                        'is_blocked': is_blocked
                    })
                
                if step % 100 == 0:
                    print(f"[BACKEND] Step {step}/{TOTAL_STEPS}")
        
        print("[BACKEND] Simulation complete")
        
    finally:
        traci.close()
        print("[BACKEND] SUMO closed")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_file)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"[BACKEND] Results saved to: {output_path}")
    
    # Save metadata
    metadata = {
        'blocked_edges': blocked_edges_list,
        'total_steps': TOTAL_STEPS,
        'baseline_steps': BASELINE_STEPS,
        'log_interval': LOG_INTERVAL,
        'total_edges': num_edges
    }
    metadata_path = os.path.join(OUTPUT_DIR, "analysis_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[BACKEND] Metadata saved to: {metadata_path}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run traffic analysis")
    parser.add_argument("--blocked-edges", type=str, default="",
                        help="Comma-separated list of edge IDs to block")
    parser.add_argument("--output", type=str, default="failure_propagation_log.csv",
                        help="Output filename")
    
    args = parser.parse_args()
    
    blocked_edges = []
    if args.blocked_edges:
        blocked_edges = [e.strip() for e in args.blocked_edges.split(",") if e.strip()]
    
    output_path = run_analysis(blocked_edges, args.output)
    print(f"[BACKEND] Analysis complete: {output_path}")
