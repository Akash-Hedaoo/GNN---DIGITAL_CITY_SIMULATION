"""
Dashboard Utilities
===================
Cached resources for high-performance dashboard.
Uses @st.cache_resource for one-time loading.

Author: Traffic Digital Twin Project
"""

import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NETWORK_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "networks", "pune.net.xml")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "stgcn", "traffic_twin_stgcn.pt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# ST-GCN parameters
WINDOW_SIZE = 3
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
# Cached Resources
# =============================================================================

@st.cache_resource
def load_edge_geometries():
    """
    Parse SUMO network XML and extract edge geometries.
    Returns dict: {edge_id: {'coords': [(lat, lon), ...], 'center': (lat, lon)}}
    """
    if not os.path.exists(NETWORK_PATH):
        st.error(f"Network file not found: {NETWORK_PATH}")
        return {}
    
    tree = ET.parse(NETWORK_PATH)
    root = tree.getroot()
    
    # Get location offset for coordinate conversion
    location = root.find('location')
    if location is not None:
        net_offset = location.get('netOffset', '0,0')
        offset_x, offset_y = map(float, net_offset.split(','))
        conv_boundary = location.get('convBoundary', '0,0,0,0')
        orig_boundary = location.get('origBoundary', '0,0,0,0')
    else:
        offset_x, offset_y = 0, 0
    
    edges = {}
    
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')
        
        # Skip internal edges
        if edge_id.startswith(':'):
            continue
        
        # Get lane shapes
        lane = edge.find('lane')
        if lane is not None:
            shape = lane.get('shape', '')
            if shape:
                coords = []
                for point in shape.split():
                    try:
                        x, y = map(float, point.split(','))
                        # Convert to approximate lat/lon
                        lon = x + offset_x
                        lat = y + offset_y
                        coords.append((lat, lon))
                    except:
                        continue
                
                if coords:
                    center_lat = sum(c[0] for c in coords) / len(coords)
                    center_lon = sum(c[1] for c in coords) / len(coords)
                    edges[edge_id] = {
                        'coords': coords,
                        'center': (center_lat, center_lon)
                    }
    
    return edges


@st.cache_resource
def get_map_center():
    """Calculate the center coordinates for the map."""
    edges = load_edge_geometries()
    if not edges:
        # Default to Pune coordinates
        return 18.5913, 73.7389
    
    all_lats = []
    all_lons = []
    for edge_data in edges.values():
        center = edge_data['center']
        all_lats.append(center[0])
        all_lons.append(center[1])
    
    return sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)


@st.cache_resource
def load_stgcn_model(num_nodes):
    """Load the trained ST-GCN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = STGCN(
        num_nodes=num_nodes,
        input_features=WINDOW_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_STGCN_BLOCKS
    )
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
    
    return model, device


@st.cache_resource
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


def get_occupancy_color(occupancy, is_blocked=False):
    """Get color based on occupancy level."""
    if is_blocked:
        return "#000000"  # Black
    
    if occupancy > 0.8:
        return "#FF0000"  # Red
    elif occupancy > 0.5:
        return "#FFA500"  # Orange
    elif occupancy > 0.2:
        return "#FFFF00"  # Yellow
    else:
        return "#00FF00"  # Green
