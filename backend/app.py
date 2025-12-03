"""
Enhanced backend with Pune city graph visualization
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import networkx as nx
import json
import os
from backend.model import ModelWrapper

app = Flask(__name__)
CORS(app)

# Model setup
MODEL_CANDIDATES = [
    'real_city_gnn.pt',
    'trained_gnn.pt',
]
model = ModelWrapper(MODEL_CANDIDATES)

# Load Pune graph once at startup
GRAPH_FILE = 'real_city_processed.graphml'
G = None
GRAPH_DATA_CACHE = None

def load_graph():
    """Load Pune city graph"""
    global G
    if G is None and os.path.exists(GRAPH_FILE):
        print(f"📂 Loading {GRAPH_FILE}...")
        G = nx.read_graphml(GRAPH_FILE)
        print(f"✅ Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

def normalize_coordinates(x, y):
    """
    Convert projected coordinates to Pune lat/lon approximation
    """
    if isinstance(x, str) or isinstance(y, str):
        try:
            x, y = float(x), float(y)
        except:
            return None, None
    
    # Bounds from graph
    x_min, x_max = 73.82, 379410.92
    y_min, y_max = 18.61, 2061677.82
    
    # Approximate Pune bounds
    pune_lat_min, pune_lat_max = 18.45, 18.65
    pune_lon_min, pune_lon_max = 73.75, 73.95
    
    # Normalize
    try:
        x_norm = (x - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0.5
        y_norm = (y - y_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0.5
    except:
        return None, None
    
    # Scale to Pune region
    lat = pune_lat_min + y_norm * (pune_lat_max - pune_lat_min)
    lon = pune_lon_min + x_norm * (pune_lon_max - pune_lon_min)
    
    return lat, lon

def build_graph_data_cache():
    """Build complete visualization data from graph"""
    global G, GRAPH_DATA_CACHE
    
    load_graph()
    if G is None:
        return None
    
    nodes = []
    edges = []
    metros = []
    amenities = []
    
    # Build node map for quick access
    node_mapping = {}
    
    # First pass: create nodes
    for node_id, attrs in G.nodes(data=True):
        try:
            x = float(attrs.get('x', 0))
            y = float(attrs.get('y', 0))
            
            lat, lon = normalize_coordinates(x, y)
            if lat is None:
                continue
            
            is_metro = str(attrs.get('is_metro_station', 'False')).lower() == 'true'
            amenity_type = str(attrs.get('amenity_type', 'residential'))
            
            node_obj = {
                'id': str(node_id),
                'lat': lat,
                'lon': lon,
                'x': x,
                'y': y,
                'name': str(attrs.get('name', f'Node {node_id}'))[:50],
                'is_metro': is_metro,
                'amenity_type': amenity_type,
                'population': int(attrs.get('population', 0)),
                'street_count': int(attrs.get('street_count', 0))
            }
            
            node_mapping[str(node_id)] = node_obj
            nodes.append(node_obj)
            
            # Categorize
            if is_metro:
                metros.append(node_obj)
            elif amenity_type not in ['residential', '']:
                amenities.append(node_obj)
                
        except (ValueError, TypeError):
            pass
    
    # Second pass: create edges with references
    for u, v, attrs in G.edges(data=True):
        try:
            u_str, v_str = str(u), str(v)
            
            if u_str not in node_mapping or v_str not in node_mapping:
                continue
            
            source_node = node_mapping[u_str]
            target_node = node_mapping[v_str]
            
            length = float(attrs.get('length', 0))
            congestion = float(attrs.get('congestion_level', 0.5))
            is_metro = str(attrs.get('is_metro', 'False')).lower() == 'true'
            highway = str(attrs.get('highway', 'road'))
            
            edge_obj = {
                'id': f"{u_str}_{v_str}",
                'source': u_str,
                'target': v_str,
                'source_lat': source_node['lat'],
                'source_lon': source_node['lon'],
                'target_lat': target_node['lat'],
                'target_lon': target_node['lon'],
                'length': round(length, 2),
                'congestion': max(0, min(1, congestion)),
                'is_metro': is_metro,
                'highway': highway,
                'name': str(attrs.get('name', ''))[:50]
            }
            
            edges.append(edge_obj)
            
        except (ValueError, TypeError, KeyError):
            pass
    
    # Build cache
    GRAPH_DATA_CACHE = {
        'city': 'Pune, Maharashtra, India',
        'nodes': nodes[:2000],
        'edges': edges[:5000],
        'metros': metros,
        'amenities': amenities,
        'statistics': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'metro_stations': len(metros),
            'amenities': len(amenities),
            'coordinate_system': 'Projected → Lat/Lon (Pune Region)'
        }
    }
    
    print(f"✅ Graph cache built: {len(nodes)} nodes, {len(edges)} edges")
    return GRAPH_DATA_CACHE

# Pre-load graph data
print("🚀 Initializing Flask backend...")
load_graph()
build_graph_data_cache()

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'city': 'Pune'})

@app.route('/city-data', methods=['GET'])
def city_data():
    """Return Pune city graph data for visualization"""
    global GRAPH_DATA_CACHE
    
    if GRAPH_DATA_CACHE is None:
        build_graph_data_cache()
    
    if GRAPH_DATA_CACHE is None:
        return jsonify({
            'error': 'Graph data not available',
            'fallback': True
        }), 503
    
    return jsonify(GRAPH_DATA_CACHE)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict traffic on edges given features"""
    payload = request.get_json(force=True, silent=True) or {}
    features = payload.get('features', [])
    
    preds = model.predict(features)
    
    if isinstance(preds, dict) and 'error' in preds:
        return jsonify(preds), 400
    
    return jsonify({
        'predictions': preds,
        'count': len(preds) if isinstance(preds, list) else 0
    })

@app.route('/whatif', methods=['POST'])
def whatif():
    """What-if scenario analysis"""
    payload = request.get_json(force=True, silent=True) or {}
    features = payload.get('features', [])
    scenario = payload.get('scenario', {})
    
    preds_before = model.predict(features)
    
    scenario_type = scenario.get('type', 'event')
    severity = float(scenario.get('severity', 0.5))
    duration = int(scenario.get('duration', 30))
    
    # Mock impact calculation
    impact_multiplier = 1.0 + (severity * 0.5)
    preds_after = [round(p * impact_multiplier, 4) for p in preds_before] if isinstance(preds_before, list) else preds_before
    
    return jsonify({
        'before': preds_before,
        'after': preds_after,
        'scenario': scenario,
        'impact_summary': {
            'avg_before': round(sum(preds_before) / len(preds_before), 4) if preds_before else 0,
            'avg_after': round(sum(preds_after) / len(preds_after), 4) if preds_after else 0,
            'affected_edges': len(preds_after)
        }
    })

@app.route('/edge/<edge_id>', methods=['GET'])
def get_edge(edge_id):
    """Get details for a specific edge"""
    global G, GRAPH_DATA_CACHE
    
    if G is None:
        load_graph()
    
    if GRAPH_DATA_CACHE is None:
        build_graph_data_cache()
    
    for edge in GRAPH_DATA_CACHE.get('edges', []):
        if edge['id'] == edge_id:
            return jsonify(edge)
    
    return jsonify({'error': 'Edge not found'}), 404

@app.route('/node/<node_id>', methods=['GET'])
def get_node(node_id):
    """Get details for a specific node"""
    global G, GRAPH_DATA_CACHE
    
    if G is None:
        load_graph()
    
    if GRAPH_DATA_CACHE is None:
        build_graph_data_cache()
    
    for node in GRAPH_DATA_CACHE.get('nodes', []):
        if node['id'] == node_id:
            return jsonify(node)
    
    return jsonify({'error': 'Node not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
