"""
Phase 4.1: ST-GCN Evaluation Sanity Check
==========================================
Evaluate the trained ST-GCN model WITHOUT retraining.
This phase verifies numeric stability and prediction sanity.

STRICT RULES:
- No retraining
- No dataset/network modification
- Inference ONLY
- No hyperparameter changes

Author: Traffic Digital Twin Project
"""

import os
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Configuration (SAME AS TRAINING - DO NOT MODIFY)
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "datasets", "sumo_dataset.csv")
NETWORK_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "networks", "pune.net.xml")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "stgcn", "traffic_twin_stgcn.pt")

# Hyperparameters (SAME AS TRAINING)
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
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] CUDA available: {device_name}")
        print(f"[INFO] Using GPU for inference")
        batch_size = 64
    else:
        device = torch.device('cpu')
        print("[INFO] CUDA not available")
        print("[INFO] Using CPU for inference")
        batch_size = 16
    
    return device, batch_size

# =============================================================================
# Data Loading (REUSED FROM TRAINING)
# =============================================================================

def load_dataset(path):
    """Load and validate the SUMO dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ['timestep', 'edge_id', 'occupancy']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Sort by timestep then edge_id for proper temporal alignment
    df = df.sort_values(['timestep', 'edge_id']).reset_index(drop=True)
    
    # Create edge index mapping
    unique_edges = df['edge_id'].unique()
    edge_to_idx = {edge: idx for idx, edge in enumerate(unique_edges)}
    idx_to_edge = {idx: edge for edge, idx in edge_to_idx.items()}
    
    df['edge_idx'] = df['edge_id'].map(edge_to_idx)
    
    print(f"[INFO] Dataset loaded: {len(df)} records")
    print(f"[INFO] Unique edges: {len(unique_edges)}")
    print(f"[INFO] Timesteps: {df['timestep'].min()} to {df['timestep'].max()}")
    
    return df, edge_to_idx, idx_to_edge

# =============================================================================
# Adjacency Matrix Construction (REUSED FROM TRAINING)
# =============================================================================

def build_adjacency_matrix(network_path, edge_to_idx):
    """Build adjacency matrix from SUMO network file."""
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"Network file not found at: {network_path}")
    
    print(f"[INFO] Parsing network from: {network_path}")
    
    num_edges = len(edge_to_idx)
    
    # Parse the network XML
    tree = ET.parse(network_path)
    root = tree.getroot()
    
    # Build junction-to-edges mapping
    junction_outgoing = defaultdict(list)
    junction_incoming = defaultdict(list)
    
    edge_count = 0
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')
        
        # Skip internal edges (those starting with ':')
        if edge_id.startswith(':'):
            continue
            
        from_junction = edge.get('from')
        to_junction = edge.get('to')
        
        if from_junction and to_junction and edge_id in edge_to_idx:
            junction_outgoing[from_junction].append(edge_id)
            junction_incoming[to_junction].append(edge_id)
            edge_count += 1
    
    print(f"[INFO] Found {edge_count} edges with junction connectivity")
    
    # Build adjacency: edge A -> edge B if A's to_junction == B's from_junction
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
    
    # Add self-loops
    adjacency = adjacency + np.eye(num_edges, dtype=np.float32)
    
    # Symmetric normalization: A_hat = D^(-1/2) @ A @ D^(-1/2)
    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    
    adjacency_normalized = D_inv_sqrt @ adjacency @ D_inv_sqrt
    
    print(f"[INFO] Adjacency matrix shape: {adjacency_normalized.shape}")
    print(f"[INFO] Non-zero connections: {np.count_nonzero(adjacency_normalized)}")
    
    return torch.FloatTensor(adjacency_normalized)

# =============================================================================
# Temporal Windowing & Dataset (REUSED FROM TRAINING)
# =============================================================================

def create_temporal_windows(df, edge_to_idx, window_size=3, prediction_horizon=5):
    """Create sliding windows for temporal prediction."""
    num_edges = len(edge_to_idx)
    timesteps = sorted(df['timestep'].unique())
    
    # Pivot to get edge x timestep matrix
    pivot = df.pivot_table(
        index='edge_idx', 
        columns='timestep', 
        values='occupancy',
        fill_value=0.0
    )
    
    # Reindex to ensure all edges are present
    pivot = pivot.reindex(range(num_edges), fill_value=0.0)
    
    # Convert to numpy array: shape (num_edges, num_timesteps)
    data = pivot.values.astype(np.float32)
    
    print(f"[INFO] Data matrix shape: {data.shape} (edges x timesteps)")
    
    X_list = []
    Y_list = []
    
    num_timesteps = data.shape[1]
    
    # Create sliding windows
    for t in range(window_size - 1, num_timesteps - prediction_horizon):
        X = data[:, t - window_size + 1 : t + 1]
        Y = data[:, t + prediction_horizon]
        
        X_list.append(X)
        Y_list.append(Y)
    
    X_array = np.array(X_list, dtype=np.float32)
    Y_array = np.array(Y_list, dtype=np.float32)
    
    print(f"[INFO] Created {len(X_list)} temporal windows")
    print(f"[INFO] X shape: {X_array.shape}, Y shape: {Y_array.shape}")
    
    return X_array, Y_array


class TrafficDataset(Dataset):
    """PyTorch dataset for traffic prediction."""
    
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# =============================================================================
# ST-GCN Model Architecture (REUSED FROM TRAINING - DO NOT MODIFY)
# =============================================================================

class SpatialGraphConv(nn.Module):
    """Spatial graph convolution layer."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return output


class TemporalConv(nn.Module):
    """Temporal convolution layer using 1D convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=kernel_size // 2
        )
    
    def forward(self, x):
        batch, nodes, time = x.shape
        x = x.reshape(batch * nodes, 1, time)
        x = x.expand(-1, self.conv.in_channels, -1) if self.conv.in_channels > 1 else x
        x = self.conv(x)
        x = x.reshape(batch, nodes, -1)
        return x


class STGCNBlock(nn.Module):
    """Spatio-Temporal Graph Convolution Block."""
    
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
    """Spatio-Temporal Graph Convolutional Network for traffic prediction."""
    
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
# Evaluation (Inference Only)
# =============================================================================

def run_inference(model, val_loader, adj, device):
    """Run inference on validation set and collect predictions."""
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device)
            
            predictions = model(X_batch, adj)
            
            all_predictions.append(predictions.cpu().numpy())
            all_ground_truth.append(Y_batch.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)
    
    return predictions, ground_truth


def print_sanity_statistics(predictions, ground_truth):
    """Print numeric sanity statistics for predictions and ground truth."""
    print("\n" + "="*60)
    print("SANITY CHECK STATISTICS")
    print("="*60)
    
    print("\n--- Prediction Statistics ---")
    print(f"  Min:    {predictions.min():.6f}")
    print(f"  Max:    {predictions.max():.6f}")
    print(f"  Mean:   {predictions.mean():.6f}")
    print(f"  Std:    {predictions.std():.6f}")
    
    print("\n--- Ground Truth Statistics ---")
    print(f"  Min:    {ground_truth.min():.6f}")
    print(f"  Max:    {ground_truth.max():.6f}")
    print(f"  Mean:   {ground_truth.mean():.6f}")
    print(f"  Std:    {ground_truth.std():.6f}")
    
    print("\n" + "="*60)

# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*60)
    print("Phase 4.1: ST-GCN Evaluation Sanity Check")
    print("="*60 + "\n")
    
    # Setup device
    device, batch_size = setup_device()
    
    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")
    print(f"[INFO] Model path: {MODEL_PATH}")
    
    # Load dataset
    df, edge_to_idx, idx_to_edge = load_dataset(DATASET_PATH)
    num_edges = len(edge_to_idx)
    
    # Build adjacency matrix
    adj = build_adjacency_matrix(NETWORK_PATH, edge_to_idx)
    adj = adj.to(device)
    
    # Create temporal windows
    X, Y = create_temporal_windows(df, edge_to_idx, WINDOW_SIZE, PREDICTION_HORIZON)
    
    # Train/val split (SAME AS TRAINING: 80/20, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    print(f"[INFO] Validation samples: {len(X_val)}")
    
    # Create validation data loader
    val_dataset = TrafficDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with SAME architecture
    model = STGCN(
        num_nodes=num_edges,
        input_features=WINDOW_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_STGCN_BLOCKS
    )
    
    # Load trained weights
    print(f"[INFO] Loading trained model weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    print(f"[INFO] Model loaded successfully")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}")
    
    # Run inference on validation set
    print(f"\n[INFO] Running inference on validation set...")
    predictions, ground_truth = run_inference(model, val_loader, adj, device)
    
    print(f"[INFO] Predictions shape: {predictions.shape}")
    print(f"[INFO] Ground truth shape: {ground_truth.shape}")
    
    # Print sanity statistics
    print_sanity_statistics(predictions, ground_truth)
    
    # Completion message
    print("\nPHASE 4.1 SANITY CHECK COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
