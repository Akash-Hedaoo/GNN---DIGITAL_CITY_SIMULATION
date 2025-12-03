import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader  # Critical: Use efficient loader
import networkx as nx
import pickle
import time
import random

# Configuration
GRAPH_FILE = 'real_city_processed.graphml'
DATA_FILE = 'gnn_training_data.pkl'
MODEL_FILE = 'real_city_gnn.pt'
EPOCHS = 30
BATCH_SIZE = 8  # Reduced to 8 to be 100% safe on 6GB VRAM

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class TrafficGATv2(nn.Module):
    def __init__(self, in_channels=4, edge_features=3, hidden_channels=64, heads=4):
        super(TrafficGATv2, self).__init__()
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, hidden_channels)
        
        # GATv2 Layers
        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, edge_dim=hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, edge_dim=hidden_channels)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, edge_dim=hidden_channels)
        
        # Predictor Head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2 + hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_emb = self.edge_encoder(edge_attr)
        
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_emb))
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_emb))
        x = self.conv3(x, edge_index, edge_attr=edge_emb)
        
        src, dest = edge_index[0], edge_index[1]
        edge_context = torch.cat([x[src], x[dest], edge_emb], dim=1)
        return self.predictor(edge_context)

# ==========================================
# 2. CPU-BASED DATA LOADING (Memory Safe)
# ==========================================
def load_dataset():
    print(f"🔄 Loading Graph & Data to System RAM...")
    G = nx.read_graphml(GRAPH_FILE)
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    # 1. Static Node Features (Keep on CPU)
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
    
    # 2. Load Snapshots
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

        # Create Data object (Tensors remain on CPU for now)
        data = Data(
            x=static_x.clone(), 
            edge_index=torch.tensor([u_list, v_list], dtype=torch.long),
            edge_attr=torch.tensor(edge_feats, dtype=torch.float),
            y=torch.tensor(targets, dtype=torch.float)
        )
        data_list.append(data)
        
    return data_list

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    # 1. Check Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        print(f"✅ GPU DETECTED: {props.name} ({props.total_memory / 1024**3:.1f} GB VRAM)")
    else:
        device = torch.device('cpu')
        print("⚠️  WARNING: Running on CPU.")

    # 2. Load Data (CPU RAM)
    full_dataset = load_dataset()
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data = full_dataset[:train_size]
    test_data = full_dataset[train_size:]
    
    # Use PyG DataLoader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    model = TrafficGATv2().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    print(f"\n🔥 Starting Memory-Safe Training ({EPOCHS} Epochs)...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Move only this small batch to GPU
            batch = batch.to(device)
            
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                val_loss += criterion(out, batch.y).item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"   Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    total_time = time.time() - start_time
    print(f"\n✅ Training Complete in {total_time:.1f} seconds!")
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"💾 Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train()