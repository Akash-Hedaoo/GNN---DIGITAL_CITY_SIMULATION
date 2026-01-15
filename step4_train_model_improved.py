import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import networkx as nx
import pickle
import time
import random
import numpy as np

# Configuration
GRAPH_FILE = 'real_city_processed.graphml'
DATA_FILE = 'gnn_training_data.pkl'
MODEL_FILE = 'real_city_gnn_improved.pt'  # Improved model name

# IMPROVED HYPERPARAMETERS
EPOCHS = 100  # Increased to 100 epochs for maximum accuracy
BATCH_SIZE = 8  # Keep same for 6GB VRAM
INITIAL_LR = 0.001  # Reduced from 0.002 for more stable training
WEIGHT_DECAY = 1e-5  # L2 regularization
PATIENCE = 10  # Early stopping patience

# ARCHITECTURE IMPROVEMENTS
HIDDEN_CHANNELS = 96  # Increased from 64 to 96 for more capacity
ATTENTION_HEADS = 6  # Increased from 4 to 6 for better attention
DROPOUT_RATE = 0.1  # Add dropout for regularization

# ==========================================
# 1. IMPROVED MODEL ARCHITECTURE
# ==========================================
class TrafficGATv2Improved(nn.Module):
    def __init__(self, in_channels=4, edge_features=3, hidden_channels=96, heads=6, dropout=0.1):
        super(TrafficGATv2Improved, self).__init__()
        
        # Enhanced encoders with dropout
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GATv2 Layers with residual connections and dropout
        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, edge_dim=hidden_channels, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, edge_dim=hidden_channels, dropout=dropout)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, edge_dim=hidden_channels, dropout=dropout)
        
        # Layer normalization for better training stability
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.ln3 = nn.LayerNorm(hidden_channels)
        
        # Enhanced Predictor Head (deeper and with dropout)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2 + hidden_channels, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Less dropout in final layers
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # Encode nodes and edges
        x = self.node_encoder(x)
        edge_emb = self.edge_encoder(edge_attr)
        
        # GATv2 layers with residual connections and layer norm
        x1 = F.elu(self.conv1(x, edge_index, edge_attr=edge_emb))
        x1 = self.ln1(x1)
        x = x + x1  # Residual connection
        
        x2 = F.elu(self.conv2(x, edge_index, edge_attr=edge_emb))
        x2 = self.ln2(x2)
        x = x + x2  # Residual connection
        
        x3 = self.conv3(x, edge_index, edge_attr=edge_emb)
        x3 = self.ln3(x3)
        x = x + x3  # Residual connection
        
        # Edge-level prediction
        src, dest = edge_index[0], edge_index[1]
        edge_context = torch.cat([x[src], x[dest], edge_emb], dim=1)
        return self.predictor(edge_context)

# ==========================================
# 2. DATA LOADING (Same as before)
# ==========================================
def load_dataset():
    print(f"🔄 Loading Graph & Data to System RAM...")
    G = nx.read_graphml(GRAPH_FILE)
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    # Static Node Features
    num_nodes = len(node_list)
    static_x = torch.zeros((num_nodes, 4), dtype=torch.float)
    for i, node in enumerate(node_list):
        d = G.nodes[node]
        try: pop = float(d.get('population', 0))
        except: pop = 0.0
        is_metro = 1.0 if str(d.get('is_metro_station', 'False')) == 'True' else 0.0
        try: x, y = float(d.get('x',0)), float(d.get('y',0))
        except: x, y = 0.0, 0.0
        
        static_x[i] = torch.tensor([pop/10000.0, is_metro, x, y])
    
    # Load Snapshots
    with open(DATA_FILE, 'rb') as f:
        snapshots = pickle.load(f)

    print(f"📦 Processing {len(snapshots)} snapshots...")
    data_list = []
    
    for snap in snapshots:
        u_list, v_list, edge_feats, targets = [], [], [], []
        
        for (u, v, k), _ in snap.edge_travel_times.items():
            if u not in node_to_idx or v not in node_to_idx: continue
            
            # Robust Edge Features
            if G.has_edge(u, v, k): d = G[u][v][k]
            elif G.has_edge(u, v): d = G[u][v][0]
            else: d = {}
            
            try: base_time = float(d.get('base_travel_time', 1.0))
            except: base_time = 1.0
            
            is_metro = 1.0 if str(d.get('is_metro', 'False')) == 'True' else 0.0
            is_closed = 1.0 if (u, v, k) in snap.closed_edges else 0.0
            
            u_list.append(node_to_idx[u])
            v_list.append(node_to_idx[v])
            edge_feats.append([base_time, is_closed, is_metro])
            targets.append([snap.edge_congestion.get((u, v, k), 1.0)])

        data = Data(
            x=static_x.clone(), 
            edge_index=torch.tensor([u_list, v_list], dtype=torch.long),
            edge_attr=torch.tensor(edge_feats, dtype=torch.float),
            y=torch.tensor(targets, dtype=torch.float)
        )
        data_list.append(data)
        
    return data_list

# ==========================================
# 3. IMPROVED TRAINING LOOP
# ==========================================
def train():
    # Check Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        print(f"✅ GPU DETECTED: {props.name} ({props.total_memory / 1024**3:.1f} GB VRAM)")
    else:
        device = torch.device('cpu')
        print("⚠️  WARNING: Running on CPU.")

    # Load Data
    full_dataset = load_dataset()
    
    # Better data split (shuffle first)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    
    train_size = int(0.8 * len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_data = [full_dataset[i] for i in train_indices]
    val_data = [full_dataset[i] for i in val_indices]
    
    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create improved model
    model = TrafficGATv2Improved(
        hidden_channels=HIDDEN_CHANNELS,
        heads=ATTENTION_HEADS,
        dropout=DROPOUT_RATE
    ).to(device)
    
    # Improved optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler (reduces LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Huber loss (more robust to outliers than MSE)
    criterion = nn.HuberLoss(delta=1.0)
    
    print(f"\n🔥 Starting IMPROVED Training ({EPOCHS} Epochs)...")
    print(f"   Architecture: Hidden={HIDDEN_CHANNELS}, Heads={ATTENTION_HEADS}, Dropout={DROPOUT_RATE}")
    print(f"   Optimizer: AdamW, LR={INITIAL_LR}, Weight Decay={WEIGHT_DECAY}")
    print(f"   Loss: Huber Loss (delta=1.0)")
    print(f"   Scheduler: ReduceLROnPlateau (patience=5)")
    print(f"   Early Stopping: Patience={PATIENCE}")
    
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                val_loss += criterion(out, batch.y).item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print LR reduction if it changed
        lr_reduced = ""
        if current_lr < old_lr:
            lr_reduced = f" 🔽 LR reduced to {current_lr:.6f}"
        
        # Early stopping check
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"   Epoch {epoch+1:02d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} ⭐ BEST | LR: {current_lr:.6f}{lr_reduced}")
        else:
            patience_counter += 1
            print(f"   Epoch {epoch+1:02d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {current_lr:.6f}{lr_reduced}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            print(f"   Best validation loss: {best_val_loss:.6f}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model (val loss: {best_val_loss:.6f})")
    
    total_time = time.time() - start_time
    print(f"\n✅ Training Complete in {total_time/60:.1f} minutes!")
    
    # Save model
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"💾 Model saved to {MODEL_FILE}")
    
    # Print final statistics
    print(f"\n📊 Training Statistics:")
    print(f"   Final Train Loss: {train_losses[-1]:.6f}")
    print(f"   Best Val Loss: {best_val_loss:.6f}")
    print(f"   Improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}% reduction")
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    train()
