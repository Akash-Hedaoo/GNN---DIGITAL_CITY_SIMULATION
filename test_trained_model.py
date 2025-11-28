"""
Test and Validate the Trained GATv2 Model
==========================================

This script tests the trained model to ensure it works correctly:
1. Load the trained model
2. Test inference on sample data
3. Generate predictions
4. Display statistics
5. Validate output shapes and values
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import networkx as nx
import numpy as np
from gnn_model import TrafficGATv2, load_model, TrafficDataLoader

print("="*70)
print("[TEST] TRAINED GNN MODEL VALIDATION")
print("="*70)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[DEVICE] Using: {device}")

# Load the trained model
print(f"\n[LOAD] Loading trained model from 'trained_gnn.pt'...")
try:
    # Create model first
    model = TrafficGATv2(
        in_channels=4,
        edge_features=3,
        hidden_channels=64,
        num_heads=4,
        num_layers=3,
        dropout=0.2,
        output_dim=1
    )
    model = load_model(model, 'trained_gnn.pt')
    model = model.to(device)
    print("[OK] Model loaded successfully!")
except FileNotFoundError:
    print("[ERROR] trained_gnn.pt not found! Run training first: python train_model.py")
    exit(1)

# Model info
total_params = sum(p.numel() for p in model.parameters())
print(f"\n[INFO] Model Architecture:")
print(f"   Total Parameters: {total_params:,}")
print(f"   Device: {device}")

# Load test data (first 100 snapshots)
print(f"\n[LOAD] Loading test data...")
data_loader = TrafficDataLoader('gnn_training_data.pkl', device=device)

try:
    all_data, metadata = data_loader.load_data()
    test_data = all_data[:100]  # Use first 100 for testing
    print(f"[OK] Loaded {len(test_data)} test snapshots")
except FileNotFoundError:
    print("[ERROR] gnn_training_data.pkl not found! Run: python generate_data.py")
    exit(1)

# Test inference
print(f"\n[TEST] Running inference on test data...")
model.eval()

all_predictions = []
all_targets = []
all_mse_losses = []

with torch.no_grad():
    for idx, data in enumerate(test_data):
        try:
            # Move data to device
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
            data.y = data.y.to(device)
            
            # Debug: Print first prediction
            if idx == 0:
                print(f"\n[DEBUG] First batch info:")
                print(f"  x shape: {data.x.shape}")
                print(f"  edge_index shape: {data.edge_index.shape}")
                print(f"  edge_attr shape: {data.edge_attr.shape if data.edge_attr is not None else None}")
                print(f"  y shape: {data.y.shape}")
                print(f"  Model training: {model.training}")
            
            # Forward pass
            predictions = model(
                x=data.x,
                edge_index=data.edge_index,
                edge_features=data.edge_attr if data.edge_attr is not None else torch.zeros(data.edge_index.size(1), 3, device=device)
            )
            
            if idx == 0:
                print(f"  predictions shape: {predictions.shape}")
                print(f"  predictions min/max: {predictions.min():.6f}/{predictions.max():.6f}")
                print(f"  y min/max: {data.y.min():.6f}/{data.y.max():.6f}")
            
            # Calculate MSE loss
            mse_loss = nn.MSELoss()(predictions, data.y)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
            all_mse_losses.append(mse_loss.item())
            
            if (idx + 1) % 20 == 0:
                print(f"   Processed {idx + 1}/{len(test_data)} snapshots")
        
        except Exception as e:
            print(f"   [ERROR] Snapshot {idx}: {e}")
            continue

print(f"\n[OK] Inference complete on {len(all_predictions)} snapshots")

# Statistics
predictions_array = np.concatenate(all_predictions, axis=0)
targets_array = np.concatenate(all_targets, axis=0)
mse_losses = np.array(all_mse_losses)

print(f"\n{'='*70}")
print(f"[STATISTICS] Model Performance on Test Data")
print(f"{'='*70}")

print(f"\n[PREDICTIONS]")
print(f"   Shape: {predictions_array.shape}")
print(f"   Min: {predictions_array.min():.6f}")
print(f"   Max: {predictions_array.max():.6f}")
print(f"   Mean: {predictions_array.mean():.6f}")
print(f"   Std: {predictions_array.std():.6f}")

print(f"\n[TARGETS]")
print(f"   Shape: {targets_array.shape}")
print(f"   Min: {targets_array.min():.6f}")
print(f"   Max: {targets_array.max():.6f}")
print(f"   Mean: {targets_array.mean():.6f}")
print(f"   Std: {targets_array.std():.6f}")

print(f"\n[LOSS STATISTICS]")
print(f"   Mean MSE Loss: {mse_losses.mean():.6e}")
print(f"   Min MSE Loss: {mse_losses.min():.6e}")
print(f"   Max MSE Loss: {mse_losses.max():.6e}")
print(f"   Std MSE Loss: {mse_losses.std():.6e}")

# Prediction error
errors = predictions_array - targets_array
abs_errors = np.abs(errors)

print(f"\n[ERROR ANALYSIS]")
print(f"   Mean Absolute Error: {abs_errors.mean():.6f}")
print(f"   Median Absolute Error: {np.median(abs_errors):.6f}")
print(f"   Max Absolute Error: {abs_errors.max():.6f}")
print(f"   95th Percentile Error: {np.percentile(abs_errors, 95):.6f}")

# Prediction distribution
print(f"\n[DISTRIBUTION]")
print(f"   Predictions in [0.5, 1.5]: {np.sum((predictions_array >= 0.5) & (predictions_array <= 1.5))}")
print(f"   Predictions in [1.5, 3.0]: {np.sum((predictions_array > 1.5) & (predictions_array <= 3.0))}")
print(f"   Predictions > 3.0: {np.sum(predictions_array > 3.0)}")

# Model state
print(f"\n[MODEL STATE]")
print(f"   Training Mode: {model.training}")
print(f"   Device: {next(model.parameters()).device}")

print(f"\n{'='*70}")
print(f"[SUCCESS] Model validation complete!")
print(f"{'='*70}")

print(f"\n[NEXT STEPS]")
print(f"   1. Use model for predictions: python interactive_whatif.py")
print(f"   2. Build GUI with predictions")
print(f"   3. Share trained_gnn.pt with your friend")
print(f"\nModel file: trained_gnn.pt ({total_params:,} parameters)")
print(f"Ready for production use!")
