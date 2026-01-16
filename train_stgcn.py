import os
import sys
import csv
import logging
import sumolib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Setup logging
log_file = os.path.join("logs", "ai_training.log")
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

NETWORK_FILE = os.path.join("outputs", "metro_corridor.net.xml")
DATA_FILE = os.path.join("outputs", "coarse_training_data.csv")
EDGE_LIST_FILE = os.path.join("outputs", "coarse_edges.txt")
MODEL_FILE = os.path.join("outputs", "sumo_gnn_model.pt")

def load_coarse_edges():
    with open(EDGE_LIST_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def build_graph(net_file, edge_subset=None):
    logging.info("Building graph from SUMO network...")
    net = sumolib.net.readNet(net_file)
    edges = net.getEdges()
    
    if edge_subset:
        edge_list = edge_subset
        edge_set = set(edge_list)
    else:
        # Fallback to previous logic if no subset
        edge_list = [e.getID() for e in edges if not e.getID().startswith(":")]
        edge_set = set(edge_list)

    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    
    num_nodes = len(edge_list)
    adj_matrix = np.eye(num_nodes) # Self loops
    
    for e in edges:
        eid = e.getID()
        # Even if 'e' isn't in our subset, it might connect two edges that ARE.
        # But for adjacent matrix A_ij = 1 if i -> j.
        # If we removed intermediate edges, we lose connectivity unless we bridge them.
        # For this mission, let's assume direct connectivity or skip deep reduction.
        # We only check connections between kept edges.
        
        if eid not in edge_set:
            continue
            
        # Outgoing connections
        for out_edge in e.getOutgoing():
            out_id = out_edge.getID()
            if out_id in edge_set:
                src = edge_to_idx[eid]
                dst = edge_to_idx[out_id]
                adj_matrix[src, dst] = 1
                
    # Normalize adjacency
    degree = np.sum(adj_matrix, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt)
    
    return edge_list, torch.tensor(norm_adj, dtype=torch.float32)

def load_data(csv_file, edge_list):
    logging.info("Loading traffic data...")
    df = pd.read_csv(csv_file)
    
    # Pivot to get matrix: Time x Edges
    # We aggregate by step and edge_id.
    # occupancy is our target feature.
    pivot_df = df.pivot(index='step', columns='edge_id', values='occupancy')
    
    # Filter columns to match our graph edge list
    valid_edges = [e for e in edge_list if e in pivot_df.columns]
    # If some edges from graph are missing in data, fill 0
    missing_edges = list(set(edge_list) - set(valid_edges))
    
    data_df = pivot_df[valid_edges]
    for e in missing_edges:
        data_df[e] = 0.0
        
    # Reorder columns to match edge_list indices
    data_df = data_df[edge_list]
    
    # Fill NaN with 0
    data_df = data_df.fillna(0.0)
    
    return data_df.values

def create_dataset(data, window_size=90, horizon=90):
    # window_size = 90 steps (approx 15 mins if 10s per step recorded)
    # Actually step recorded every 10 sim steps. Sim step default 1s. So 1 data row = 10s.
    # 15 mins = 900s = 90 rows.
    
    X, Y = [], []
    num_samples = len(data) - window_size - horizon + 1
    
    if num_samples < 1:
        logging.warning("Not enough data for the requested window size.")
        return torch.tensor([]), torch.tensor([])

    for i in range(num_samples):
        X.append(data[i : i+window_size])
        Y.append(data[i+window_size : i+window_size+horizon]) # Predict sequence
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.adj = adj
        
    def forward(self, x):
        x_trans = self.fc(x)
        out = torch.matmul(self.adj, x_trans)
        return out

class STGCN_Model(nn.Module):
    def __init__(self, num_nodes, steps_in, steps_out, adj):
        super().__init__()
        self.adj = nn.Parameter(adj, requires_grad=False)
        
        # Reduced model size
        self.gcn = GraphConv(1, 2, self.adj) # 1 -> 2 features
        self.lstm = nn.LSTM(2 * num_nodes, 16, batch_first=True) # Flatten: N*2 -> 16 hidden
        self.fc = nn.Linear(16, num_nodes * steps_out)
        self.nodes = num_nodes
        self.steps_out = steps_out
        
    def forward(self, x):
        b, t, n = x.shape
        x = x.unsqueeze(-1)
        
        x = self.gcn(x)
        x = torch.relu(x)
        
        x = x.view(b, t, -1)
        
        _, (hn, _) = self.lstm(x)
        
        out = self.fc(hn.squeeze(0))
        out = out.view(b, self.steps_out, self.nodes)
        return out

def train():
    coarse_edges = load_coarse_edges()
    edge_list, adj = build_graph(NETWORK_FILE, edge_subset=coarse_edges)
    logging.info(f"Graph built with {len(edge_list)} nodes (Coarsened).")
    
    raw_data = load_data(DATA_FILE, edge_list)
    logging.info(f"Data shape: {raw_data.shape}")
    
    # Params
    WINDOW_STEPS = 90
    HORIZON_STEPS = 90
    
    X, Y = create_dataset(raw_data, WINDOW_STEPS, HORIZON_STEPS)
    logging.info(f"Dataset created. Samples: {X.shape[0]}")
    
    if len(X) == 0:
        logging.error("No data to train.")
        sys.exit(1)
        
    dataset = torch.utils.data.TensorDataset(X, Y)
    
    # Model
    model = STGCN_Model(len(edge_list), WINDOW_STEPS, HORIZON_STEPS, adj)
    # Move to CPU explicitly (default) but ensure we don't try GPU if not available/configured
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    epochs = 10
    BATCH_SIZE = 8
    logging.info(f"Starting training with batch size {BATCH_SIZE}...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for bx, by in dataloader:
            optimizer.zero_grad()
            output = model(bx)
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
    torch.save(model.state_dict(), MODEL_FILE)
    logging.info(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train()
