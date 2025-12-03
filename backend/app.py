"""
🚦 Digital Twin City Simulation - Flask Backend API
====================================================

RESTful API for GNN-based traffic prediction system.

Run: python app.py
API Docs: http://localhost:5000/

Author: Digital Twin City Simulation Team
Date: December 2025
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
import networkx as nx
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_model import TrafficGATv2, load_model

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)  # Enable CORS for frontend

# ============================================================
# GLOBAL STATE
# ============================================================

model = None
device = None
graph = None
model_loaded = False
graph_loaded = False


def init_model():
    """Initialize the GNN model"""
    global model, device, model_loaded
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INIT] Using device: {device}")
        
        model = TrafficGATv2(
            in_channels=4,
            edge_features=3,
            hidden_channels=64,
            num_heads=4,
            num_layers=3,
            dropout=0.2,
            output_dim=1
        )
        
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trained_gnn.pt')
        if os.path.exists(model_path):
            model = load_model(model, model_path)
            model = model.to(device)
            model.eval()
            model_loaded = True
            print("[OK] Model loaded successfully")
        else:
            print(f"[WARN] Model not found at {model_path}")
            model_loaded = False
            
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        model_loaded = False


def init_graph():
    """Load the city graph"""
    global graph, graph_loaded
    
    try:
        graph_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'city_graph.graphml')
        if os.path.exists(graph_path):
            graph = nx.read_graphml(graph_path)
            # Convert to MultiDiGraph if not already
            if not isinstance(graph, nx.MultiDiGraph):
                graph = nx.MultiDiGraph(graph)
            graph_loaded = True
            print(f"[OK] Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        else:
            print(f"[WARN] Graph not found at {graph_path}")
            graph_loaded = False
            
    except Exception as e:
        print(f"[ERROR] Failed to load graph: {e}")
        graph_loaded = False


# ============================================================
# API ROUTES
# ============================================================

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'status': 'online',
        'model_loaded': model_loaded,
        'graph_loaded': graph_loaded,
        'device': device if device else 'unknown',
        'nodes': graph.number_of_nodes() if graph_loaded else 0,
        'edges': graph.number_of_edges() if graph_loaded else 0
    })


@app.route('/api/graph')
def get_graph_data():
    """Get graph data for visualization"""
    if not graph_loaded:
        return jsonify({'error': 'Graph not loaded'}), 500
    
    nodes = []
    edges = []
    
    # Extract nodes with attributes
    for node_id, data in graph.nodes(data=True):
        amenity = data.get('amenity', 'none')
        # Check if node is a metro station (from amenity field)
        is_metro = 'metro_station' in str(amenity) or data.get('is_metro_station', 'False') == 'True'
        
        nodes.append({
            'id': node_id,
            'x': float(data.get('x', 0)),
            'y': float(data.get('y', 0)),
            'zone': data.get('zone', 'unknown'),
            'population': int(float(data.get('population', 0))),
            'amenity': amenity,
            'is_metro': is_metro
        })
    
    # Extract edges with attributes
    for u, v, data in graph.edges(data=True):
        # Check if metro edge using is_metro or transport_mode attributes
        is_metro = data.get('is_metro', False)
        if isinstance(is_metro, str):
            is_metro = is_metro.lower() == 'true'
        is_metro = is_metro or data.get('transport_mode', '') == 'metro' or data.get('highway', '') == 'metro_railway'
        
        edges.append({
            'source': u,
            'target': v,
            'is_metro': is_metro,
            'travel_time': float(data.get('base_travel_time', 1.0)),
            'metro_line': data.get('line_name', data.get('name', None)),
            'line_color': data.get('line_color', None)
        })
    
    return jsonify({
        'nodes': nodes,
        'edges': edges,
        'node_count': len(nodes),
        'edge_count': len(edges)
    })


@app.route('/api/metro-lines')
def get_metro_lines():
    """Get metro line data"""
    if not graph_loaded:
        return jsonify({'error': 'Graph not loaded'}), 500
    
    metro_lines = {}
    metro_stations = []
    
    for u, v, data in graph.edges(data=True):
        key = data.get('key', '0')
        if key == 'metro' or data.get('edge_type', '') == 'metro':
            line_name = data.get('metro_line', 'Line 1')
            if line_name not in metro_lines:
                metro_lines[line_name] = {'edges': [], 'stations': set()}
            
            metro_lines[line_name]['edges'].append({
                'source': u,
                'target': v
            })
            metro_lines[line_name]['stations'].add(u)
            metro_lines[line_name]['stations'].add(v)
    
    # Get station details
    for node_id, data in graph.nodes(data=True):
        if data.get('is_metro_station', 'False') == 'True' or data.get('amenity', '') == 'metro_station':
            metro_stations.append({
                'id': node_id,
                'x': float(data.get('x', 0)),
                'y': float(data.get('y', 0)),
                'name': data.get('name', f'Station {node_id}')
            })
    
    # Convert sets to lists for JSON
    for line in metro_lines.values():
        line['stations'] = list(line['stations'])
    
    return jsonify({
        'lines': metro_lines,
        'stations': metro_stations,
        'total_stations': len(metro_stations)
    })


@app.route('/api/predict', methods=['POST'])
def predict_traffic():
    """Predict traffic congestion"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if not graph_loaded:
        return jsonify({'error': 'Graph not loaded'}), 500
    
    try:
        data = request.json or {}
        closed_roads = data.get('closed_roads', [])
        hour = data.get('hour', 8)  # Default to 8 AM
        
        # Build node features and edge features
        node_to_idx = {n: i for i, n in enumerate(graph.nodes())}
        num_nodes = len(node_to_idx)
        
        # Node features: [population, is_metro, x, y]
        node_features = np.zeros((num_nodes, 4), dtype=np.float32)
        for node_id, idx in node_to_idx.items():
            node_data = graph.nodes[node_id]
            node_features[idx, 0] = float(node_data.get('population', 0)) / 10000.0
            node_features[idx, 1] = 1.0 if node_data.get('is_metro_station', 'False') == 'True' else 0.0
            node_features[idx, 2] = float(node_data.get('x', 0))
            node_features[idx, 3] = float(node_data.get('y', 0))
        
        # Edge features: [base_time, is_closed, is_metro]
        edge_list = []
        edge_features = []
        edge_info = []
        
        for u, v, data in graph.edges(data=True):
            key = data.get('key', '0')
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_list.append([u_idx, v_idx])
            
            is_closed = 1.0 if f"{u}-{v}" in closed_roads or f"{v}-{u}" in closed_roads else 0.0
            is_metro = 1.0 if key == 'metro' or data.get('edge_type', '') == 'metro' else 0.0
            base_time = float(data.get('base_travel_time', 1.0))
            
            edge_features.append([base_time, is_closed, is_metro])
            edge_info.append({
                'source': u,
                'target': v,
                'key': key,
                'is_metro': is_metro == 1.0,
                'is_closed': is_closed == 1.0
            })
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32).to(device)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32).to(device)
        
        # Run prediction
        with torch.no_grad():
            predictions = model(x, edge_index, edge_attr)
            predictions = predictions.cpu().numpy().flatten()
        
        # Build response
        results = []
        for i, info in enumerate(edge_info):
            results.append({
                **info,
                'congestion': float(predictions[i])
            })
        
        # Calculate statistics
        road_preds = [p for i, p in enumerate(predictions) if edge_info[i]['is_metro'] == False]
        metro_preds = [p for i, p in enumerate(predictions) if edge_info[i]['is_metro'] == True]
        
        stats = {
            'mean_congestion': float(np.mean(predictions)),
            'max_congestion': float(np.max(predictions)),
            'min_congestion': float(np.min(predictions)),
            'road_mean': float(np.mean(road_preds)) if road_preds else 0,
            'metro_mean': float(np.mean(metro_preds)) if metro_preds else 0,
            'closed_roads': len(closed_roads)
        }
        
        return jsonify({
            'predictions': results,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/zones')
def get_zones():
    """Get zone statistics"""
    if not graph_loaded:
        return jsonify({'error': 'Graph not loaded'}), 500
    
    zones = {}
    for node_id, data in graph.nodes(data=True):
        zone = data.get('zone', 'unknown')
        if zone not in zones:
            zones[zone] = {'count': 0, 'population': 0}
        zones[zone]['count'] += 1
        zones[zone]['population'] += int(data.get('population', 0))
    
    return jsonify({'zones': zones})


@app.route('/api/amenities')
def get_amenities():
    """Get amenity locations"""
    if not graph_loaded:
        return jsonify({'error': 'Graph not loaded'}), 500
    
    amenities = {}
    for node_id, data in graph.nodes(data=True):
        amenity = data.get('amenity', 'none')
        if amenity and amenity != 'none':
            if amenity not in amenities:
                amenities[amenity] = []
            amenities[amenity].append({
                'id': node_id,
                'x': float(data.get('x', 0)),
                'y': float(data.get('y', 0)),
                'zone': data.get('zone', 'unknown')
            })
    
    return jsonify({'amenities': amenities})


@app.route('/api/search')
def search_nodes():
    """Search for nodes by query"""
    if not graph_loaded:
        return jsonify({'error': 'Graph not loaded'}), 500
    
    query = request.args.get('q', '').lower().strip()
    if len(query) < 2:
        return jsonify({'results': []})
    
    results = []
    for node_id, data in graph.nodes(data=True):
        amenity = str(data.get('amenity', '')).lower()
        zone = str(data.get('zone', '')).lower()
        node_id_str = str(node_id).lower()
        
        # Check for matches
        if query in node_id_str or query in amenity or query in zone:
            results.append({
                'id': node_id,
                'x': float(data.get('x', 0)),
                'y': float(data.get('y', 0)),
                'zone': data.get('zone', 'unknown'),
                'population': int(float(data.get('population', 0))),
                'amenity': data.get('amenity', 'none')
            })
    
    # Limit results
    return jsonify({'results': results[:20]})


@app.route('/api/shortest-path', methods=['POST'])
def find_shortest_path():
    """Find shortest path between two nodes"""
    if not graph_loaded:
        return jsonify({'error': 'Graph not loaded'}), 500
    
    try:
        data = request.json
        source = data.get('source')
        target = data.get('target')
        
        if not source or not target:
            return jsonify({'error': 'Source and target required'}), 400
        
        # Find path
        try:
            path = nx.shortest_path(graph, source, target, weight='base_travel_time')
            length = nx.shortest_path_length(graph, source, target, weight='base_travel_time')
            
            return jsonify({
                'path': path,
                'length': length,
                'hops': len(path) - 1
            })
        except nx.NetworkXNoPath:
            return jsonify({'error': 'No path found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚦 Digital Twin City Simulation - Flask Backend")
    print("=" * 60)
    
    # Initialize
    init_model()
    init_graph()
    
    print("\n📡 Starting server...")
    print("   Frontend: http://localhost:5000")
    print("   API Docs: http://localhost:5000/api/status")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
