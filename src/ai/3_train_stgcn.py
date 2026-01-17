"""
Phase 3: ST-GCN Traffic Prediction Model Training
==================================================
Train a Spatio-Temporal Graph Convolutional Network on SUMO dataset
for congestion forecasting. Automatically uses GPU if available.

Input:  Past 3 traffic states [t-2, t-1, t] with occupancy
Output: Future occupancy at t+5

Author: Traffic Digital Twin Project
"""

import os
import sys
import time
import json
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
# Configuration
# =============================================================================

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "datasets", "sumo_dataset.csv")
NETWORK_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "networks", "pune.net.xml")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "stgcn", "traffic_twin_stgcn.pt")
METRICS_SAVE_PATH = os.path.join(PROJECT_ROOT, "logs", "training", "stgcn_training_metrics.json")

# Hyperparameters
WINDOW_SIZE = 3      # Past timesteps: t-2, t-1, t
PREDICTION_HORIZON = 5  # Predict t+5
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
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
        print(f"[INFO] Using GPU for training")
        batch_size = 64
    else:
        device = torch.device('cpu')
        print("[INFO] CUDA not available")
        print("[INFO] Using CPU for training (this will be slower)")
        batch_size = 16
    
    return device, batch_size

# =============================================================================
# Data Loading
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
# Adjacency Matrix Construction
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
    # Edge goes FROM junction_from TO junction_to
    junction_outgoing = defaultdict(list)  # junction -> edges leaving it
    junction_incoming = defaultdict(list)  # junction -> edges entering it
    
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
    # This means vehicles can flow from A to B
    adjacency = np.zeros((num_edges, num_edges), dtype=np.float32)
    
    # For each junction, connect incoming edges to outgoing edges
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')
        if edge_id.startswith(':') or edge_id not in edge_to_idx:
            continue
            
        to_junction = edge.get('to')
        if to_junction:
            # This edge leads to to_junction
            # Connect it to all edges leaving to_junction
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
# Temporal Windowing & Dataset
# =============================================================================

def create_temporal_windows(df, edge_to_idx, window_size=3, prediction_horizon=5):
    """Create sliding windows for temporal prediction."""
    num_edges = len(edge_to_idx)
    timesteps = sorted(df['timestep'].unique())
    
    # Pivot to get edge x timestep matrix
    # Fill missing values with 0 (no occupancy)
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
        # Input: [t-2, t-1, t] -> shape: (num_edges, window_size)
        X = data[:, t - window_size + 1 : t + 1]
        
        # Output: t+5 -> shape: (num_edges,)
        Y = data[:, t + prediction_horizon]
        
        X_list.append(X)
        Y_list.append(Y)
    
    X_array = np.array(X_list, dtype=np.float32)  # (num_samples, num_edges, window_size)
    Y_array = np.array(Y_list, dtype=np.float32)  # (num_samples, num_edges)
    
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
# ST-GCN Model Architecture
# =============================================================================

class SpatialGraphConv(nn.Module):
    """Spatial graph convolution layer."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        """
        Args:
            x: Input features (batch, num_nodes, features)
            adj: Adjacency matrix (num_nodes, num_nodes)
        Returns:
            Output features (batch, num_nodes, out_features)
        """
        # Graph convolution: X' = A_hat @ X @ W
        support = self.linear(x)  # (batch, num_nodes, out_features)
        output = torch.matmul(adj, support)  # (batch, num_nodes, out_features)
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
        """
        Args:
            x: Input (batch, num_nodes, time_steps)
        Returns:
            Output (batch, num_nodes, time_steps)
        """
        batch, nodes, time = x.shape
        # Reshape for 1D conv: (batch * nodes, 1, time)
        x = x.reshape(batch * nodes, 1, time)
        # Expand channels
        x = x.expand(-1, self.conv.in_channels, -1) if self.conv.in_channels > 1 else x
        x = self.conv(x)
        # Reshape back: (batch, nodes, time)
        x = x.reshape(batch, nodes, -1)
        return x


class STGCNBlock(nn.Module):
    """Spatio-Temporal Graph Convolution Block."""
    
    def __init__(self, in_features, hidden_features, out_features, num_nodes=None):
        super().__init__()
        self.spatial_conv = SpatialGraphConv(in_features, hidden_features)
        self.temporal_conv = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, padding=1)
        self.output_conv = SpatialGraphConv(hidden_features, out_features)
        # Use LayerNorm instead of BatchNorm for better stability with graphs
        self.norm = nn.LayerNorm(out_features)
    
    def forward(self, x, adj):
        """
        Args:
            x: Input (batch, num_nodes, time_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
        """
        # Spatial convolution
        h = self.spatial_conv(x, adj)
        h = F.relu(h)
        
        # Temporal convolution: (batch, nodes, features) -> (batch, features, nodes) for Conv1d
        h = h.transpose(1, 2)  # (batch, features, nodes)
        h = self.temporal_conv(h)
        h = h.transpose(1, 2)  # (batch, nodes, features)
        h = F.relu(h)
        
        # Output spatial convolution
        h = self.output_conv(h, adj)
        
        # Layer normalization
        h = self.norm(h)
        
        return F.relu(h)


class STGCN(nn.Module):
    """Spatio-Temporal Graph Convolutional Network for traffic prediction."""
    
    def __init__(self, num_nodes, input_features, hidden_dim, num_blocks=2):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # ST-GCN blocks - maintains feature dimension throughout
        self.blocks = nn.ModuleList()
        
        in_dim = input_features
        for i in range(num_blocks):
            self.blocks.append(STGCNBlock(in_dim, hidden_dim, hidden_dim))
            in_dim = hidden_dim
        
        # Output layer: map from hidden_dim features to single occupancy prediction
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, adj):
        """
        Args:
            x: Input (batch, num_nodes, time_window)  e.g., (64, 1074, 3)
            adj: Adjacency matrix (num_nodes, num_nodes)
        Returns:
            Predicted occupancy (batch, num_nodes)
        """
        h = x  # (batch, num_nodes, input_features)
        
        # Apply ST-GCN blocks
        for block in self.blocks:
            h = block(h, adj)  # (batch, num_nodes, hidden_dim)
        
        # h is now (batch, num_nodes, hidden_dim)
        # Apply output layer to each node independently
        # Linear expects (..., hidden_dim) and outputs (..., 1)
        out = self.output_layer(h)  # (batch, num_nodes, 1)
        out = out.squeeze(-1)  # (batch, num_nodes)
        
        return out

# =============================================================================
# Training
# =============================================================================

def train_model(model, train_loader, val_loader, adj, device, epochs, lr, weight_decay):
    """Train the ST-GCN model."""
    
    model = model.to(device)
    adj = adj.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    epoch_times = []
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch, adj)
            loss = criterion(predictions, Y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                
                predictions = model(X_batch, adj)
                loss = criterion(predictions, Y_batch)
                
                val_loss += loss.item()
                num_val_batches += 1
        
        val_loss /= num_val_batches
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Print every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {epoch_time:.2f}s")
    
    return train_losses, val_losses, epoch_times

# =============================================================================
# Saving
# =============================================================================

def save_model(model, path):
    """Save model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to: {path}")


def save_metrics(train_losses, val_losses, epoch_times, device_name, path):
    """Save training metrics to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    metrics = {
        'device': device_name,
        'epochs': len(train_losses),
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epoch_times': epoch_times,
        'total_training_time': sum(epoch_times)
    }
    
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[INFO] Metrics saved to: {path}")

# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*60)
    print("Phase 3: ST-GCN Traffic Prediction Model Training")
    print("="*60 + "\n")
    
    # Setup device
    device, batch_size = setup_device()
    device_name = str(device)
    
    # Load dataset
    df, edge_to_idx, idx_to_edge = load_dataset(DATASET_PATH)
    num_edges = len(edge_to_idx)
    
    # Build adjacency matrix
    adj = build_adjacency_matrix(NETWORK_PATH, edge_to_idx)
    
    # Create temporal windows
    X, Y = create_temporal_windows(df, edge_to_idx, WINDOW_SIZE, PREDICTION_HORIZON)
    
    # Train/test split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    print(f"[INFO] Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    # Create data loaders
    train_dataset = TrafficDataset(X_train, Y_train)
    val_dataset = TrafficDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = STGCN(
        num_nodes=num_edges,
        input_features=WINDOW_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_STGCN_BLOCKS
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}")
    
    # Train
    train_losses, val_losses, epoch_times = train_model(
        model, train_loader, val_loader, adj, device,
        EPOCHS, LEARNING_RATE, WEIGHT_DECAY
    )
    
    # Save model and metrics
    save_model(model, MODEL_SAVE_PATH)
    save_metrics(train_losses, val_losses, epoch_times, device_name, METRICS_SAVE_PATH)
    
    # Completion logs
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Device used: {device_name}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Model saved: {MODEL_SAVE_PATH}")
    print(f"Metrics saved: {METRICS_SAVE_PATH}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
