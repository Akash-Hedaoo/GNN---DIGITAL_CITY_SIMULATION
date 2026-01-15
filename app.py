from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import networkx as nx
import torch
import numpy as np
import osmnx as ox
import copy
from shapely import wkt
from pyproj import Transformer
from step4_train_model_improved import TrafficGATv2Improved
import json

app = Flask(__name__)
CORS(app)

# Configuration
GRAPH_FILE = 'real_city_processed.graphml'
MODEL_FILE = 'real_city_gnn.pt'

# Initialize Digital Twin
class CityDigitalTwin:
    def __init__(self):
        print("🔄 SYSTEM: Initializing Digital Twin Engine...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Graph
        print(f"   📂 Loading map from {GRAPH_FILE}...")
        self.G = nx.read_graphml(GRAPH_FILE)
        
        # Clean Data
        for n, d in self.G.nodes(data=True):
            d['population'] = float(d.get('population', 0))
            d['x'] = float(d.get('x', 0))
            d['y'] = float(d.get('y', 0))
        
        # Load AI Model
        print(f"   🧠 Loading AI Brain from {MODEL_FILE}...")
        self.model = TrafficGATv2Improved().to(self.device)
        try:
            self.model.load_state_dict(torch.load(MODEL_FILE, map_location=self.device))
            self.model.eval()  # Set to evaluation mode (important for dropout, batch norm)
            print("   ✅ AI Model Loaded (Improved Architecture: 96 hidden, 6 heads).")
        except FileNotFoundError:
            print("   ❌ ERROR: Model file not found.")
            raise
        
        # Mappings
        self.nodes = list(self.G.nodes())
        self.node_map = {n: i for i, n in enumerate(self.nodes)}
        
        # Get graph bounds for coordinate conversion
        self._setup_coordinate_transform()
        
        # Establish Baseline
        print("   📊 Establishing Baseline Traffic State...")
        self.baseline_predictions = self._predict_traffic(self.G)
        print("   ✅ Digital Twin Ready!")
    
    def _setup_coordinate_transform(self):
        """Setup coordinate transformer - OSMnx uses UTM projection"""
        # OSMnx projects graphs to UTM, but we need lat/lon for map display
        try:
            if hasattr(self.G, 'graph') and 'crs' in self.G.graph:
                crs = self.G.graph['crs']
                print(f"   📍 Graph CRS: {crs}")
                # OSMnx typically uses UTM, convert to WGS84 (lat/lon)
                self.transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
            else:
                print("   ⚠️ No CRS found in graph, attempting to infer...")
                # Try to infer from coordinates
                sample_node = self.nodes[0] if self.nodes else None
                if sample_node:
                    x, y = self.G.nodes[sample_node].get('x', 0), self.G.nodes[sample_node].get('y', 0)
                    # If coordinates are in reasonable lat/lon range, assume already converted
                    if -90 <= y <= 90 and -180 <= x <= 180:
                        print("   ✅ Coordinates appear to be in lat/lon format")
                        self.transformer = None
                    else:
                        # Assume UTM zone 43N for Maharashtra (Pune area)
                        print("   📍 Assuming UTM zone 43N (EPSG:32643)")
                        self.transformer = Transformer.from_crs('EPSG:32643', 'EPSG:4326', always_xy=True)
                else:
                    self.transformer = None
        except Exception as e:
            print(f"   ⚠️ Coordinate transform setup warning: {e}")
            self.transformer = None
    
    def _utm_to_latlon(self, x, y):
        """Convert UTM coordinates to lat/lon"""
        if self.transformer:
            try:
                lon, lat = self.transformer.transform(x, y)
                return lat, lon
            except Exception as e:
                print(f"   ⚠️ Transform error for ({x}, {y}): {e}")
        
        # Fallback: check if already in lat/lon range (OSMnx sometimes stores original coords)
        if -90 <= y <= 90 and -180 <= x <= 180:
            return y, x  # Already lat/lon (lat, lon format)
        
        # Last resort: If we have original lat/lon in node data, use that
        # This is a fallback - should not normally be needed
        return None, None  # Return None to indicate conversion failed
    
    def _predict_traffic(self, graph_obj):
        """Runs Hybrid Inference: GNN (Structure) + Physics (Density)."""
        # PART A: AI Prediction (GNN)
        x_data = []
        for n in self.nodes:
            d = graph_obj.nodes[n]
            is_metro = 1.0 if str(d.get('is_metro_station', 'False')) == 'True' else 0.0
            x_data.append([d['population'] / 10000.0, is_metro, d['x'], d['y']])
        x_tensor = torch.tensor(x_data, dtype=torch.float).to(self.device)
        
        u_list, v_list, attr_list = [], [], []
        edge_keys = []
        closed_edges_set = set()  # Track closed edges
        
        for u, v, k, d in graph_obj.edges(keys=True, data=True):
            if u not in self.node_map or v not in self.node_map:
                continue
            u_list.append(self.node_map[u])
            v_list.append(self.node_map[v])
            edge_keys.append((u, v, k))
            
            try:
                bt = float(d.get('base_travel_time', 1.0))
            except:
                bt = 1.0
            is_closed = 1.0 if str(d.get('is_closed', 'False')) == 'True' else 0.0
            is_metro = 1.0 if str(d.get('is_metro', 'False')) == 'True' else 0.0
            attr_list.append([bt, is_closed, is_metro])
            
            # Track closed edges
            if is_closed > 0.5:
                closed_edges_set.add((u, v, k))
        
        edge_index = torch.tensor([u_list, v_list], dtype=torch.long).to(self.device)
        edge_attr = torch.tensor(attr_list, dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            preds = self.model(x_tensor, edge_index, edge_attr)
        
        # PART B: Physics Density Adjustment + Closed Edge Override
        results = {}
        pred_vals = preds.cpu().numpy().flatten()
        
        for i, key in enumerate(edge_keys):
            u, v, k = key
            ai_congestion = float(pred_vals[i])
            
            # CRITICAL FIX: If edge is closed, set very high congestion
            if key in closed_edges_set:
                # Closed roads get maximum congestion (effectively impassable)
                results[key] = 10.0  # Very high congestion factor
                continue
            
            pop_u = graph_obj.nodes[u].get('population', 0)
            pop_v = graph_obj.nodes[v].get('population', 0)
            avg_pop = (pop_u + pop_v) / 2.0
            
            density_risk = 1.0 + (avg_pop / 10000.0)
            final_congestion = ai_congestion * density_risk
            
            # Also boost congestion for edges near closed roads (ripple effect)
            # Check if any neighboring edges are closed
            neighbors_closed = False
            for neighbor_u, neighbor_v, neighbor_k in graph_obj.edges(keys=True):
                if (neighbor_u == u or neighbor_v == u or neighbor_u == v or neighbor_v == v):
                    if (neighbor_u, neighbor_v, neighbor_k) in closed_edges_set:
                        neighbors_closed = True
                        break
            
            if neighbors_closed:
                # Boost congestion for edges adjacent to closed roads
                final_congestion = final_congestion * 1.5
            
            results[key] = final_congestion
        
        return results

# Initialize Digital Twin (global instance)
# Initialize immediately instead of using deprecated before_first_request
twin = CityDigitalTwin()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/maps')
def dual_maps():
    return render_template('results-map.html')

@app.route('/api/graph-data')
def get_graph_data():
    """Get graph nodes and edges with coordinates"""
    nodes_data = []
    edges_data = []
    
    # Get nodes
    for node_id in twin.nodes:
        node_data = twin.G.nodes[node_id]
        x, y = node_data.get('x', 0), node_data.get('y', 0)
        
        # Convert to lat/lon
        lat, lon = twin._utm_to_latlon(x, y)
        
        # Skip if conversion failed
        if lat is None or lon is None:
            continue
        
        nodes_data.append({
            'id': str(node_id),
            'lat': lat,
            'lon': lon,
            'x': x,
            'y': y,
            'population': node_data.get('population', 0),
            'is_metro_station': str(node_data.get('is_metro_station', 'False')) == 'True'
        })
    
    # Get edges with geometry
    for u, v, k, data in twin.G.edges(keys=True, data=True):
        # Skip metro edges for road visualization (optional)
        is_metro = str(data.get('is_metro', 'False')) == 'True'
        
        # Get coordinates
        u_data = twin.G.nodes[u]
        v_data = twin.G.nodes[v]
        
        u_x, u_y = u_data.get('x', 0), u_data.get('y', 0)
        v_x, v_y = v_data.get('x', 0), v_data.get('y', 0)
        
        u_lat, u_lon = twin._utm_to_latlon(u_x, u_y)
        v_lat, v_lon = twin._utm_to_latlon(v_x, v_y)
        
        # Skip if conversion failed
        if u_lat is None or v_lat is None:
            continue
        
        # Try to get geometry if available
        geometry = None
        if 'geometry' in data:
            geom_str = data['geometry']
            if isinstance(geom_str, str) and geom_str != 'None':
                try:
                    geom = wkt.loads(geom_str)
                    if hasattr(geom, 'coords'):
                        coords = list(geom.coords)
                        geometry = []
                        for coord_x, coord_y in coords:
                            coord_lat, coord_lon = twin._utm_to_latlon(coord_x, coord_y)
                            if coord_lat is not None:
                                geometry.append([coord_lat, coord_lon])
                        if len(geometry) < 2:
                            geometry = None
                except Exception as e:
                    pass
        
        # If no geometry, use straight line
        if geometry is None:
            geometry = [[u_lat, u_lon], [v_lat, v_lon]]
        
        edges_data.append({
            'id': f"{u}_{v}_{k}",
            'source': str(u),
            'target': str(v),
            'key': k,
            'geometry': geometry,
            'is_metro': is_metro,
            'length': float(data.get('length', 0)),
            'base_travel_time': float(data.get('base_travel_time', 1.0))
        })
    
    # Calculate map bounds
    lats = [n['lat'] for n in nodes_data]
    lons = [n['lon'] for n in nodes_data]
    
    bounds = {
        'north': max(lats),
        'south': min(lats),
        'east': max(lons),
        'west': min(lons),
        'center': {
            'lat': (max(lats) + min(lats)) / 2,
            'lon': (max(lons) + min(lons)) / 2
        }
    }
    
    return jsonify({
        'nodes': nodes_data,
        'edges': edges_data,
        'bounds': bounds,
        'baseline_predictions': {str(k): float(v) for k, v in twin.baseline_predictions.items()}
    })

@app.route('/api/simulate-closure', methods=['POST'])
def simulate_closure():
    """Simulate road closure and return predictions"""
    data = request.json
    closed_edges = data.get('closed_edges', [])
    
    # Create modified graph
    G_mod = copy.deepcopy(twin.G)
    
    # Close specified edges
    print(f"   🔒 Closing {len(closed_edges)} edges...")
    closed_count = 0
    for edge_id in closed_edges:
        parts = edge_id.split('_')
        if len(parts) >= 2:
            u, v = parts[0], parts[1]
            k = int(parts[2]) if len(parts) > 2 else 0
            
            # Try both directions (u->v and v->u) since graph might be undirected
            if G_mod.has_edge(u, v, k):
                G_mod[u][v][k]['is_closed'] = 'True'
                closed_count += 1
            elif G_mod.has_edge(v, u, k):
                G_mod[v][u][k]['is_closed'] = 'True'
                closed_count += 1
    
    print(f"   ✅ Successfully closed {closed_count} edges")
    
    # Get predictions
    new_predictions = twin._predict_traffic(G_mod)
    
    # Calculate impact metrics
    deltas = []
    impacted_edges = []  # Edges with significant impact (>2% change)
    all_edges_impact = []
    
    for key, new_val in new_predictions.items():
        base_val = twin.baseline_predictions.get(key, 1.0)
        diff = new_val - base_val
        pct_change = (diff / base_val) * 100 if base_val > 0 else 0
        deltas.append(diff)
        
        u, v, k = key
        edge_id = f"{u}_{v}_{k}"
        edge_data = {
            'edge_id': edge_id,
            'source_node': str(u),
            'target_node': str(v),
            'key': k,
            'baseline_congestion': float(base_val),
            'current_congestion': float(new_val),
            'congestion': float(new_val),
            'change': float(diff),
            'pct_change': float(pct_change)
        }
        all_edges_impact.append(edge_data)
        
        if abs(pct_change) > 2.0:  # Significant impact
            impacted_edges.append(edge_data)
    
    # Sort by percentage change to get top 5 critical edges
    top_5_critical = sorted(all_edges_impact, key=lambda x: abs(x['pct_change']), reverse=True)[:5]
    
    avg_new = np.mean(list(new_predictions.values()))
    avg_base = np.mean(list(twin.baseline_predictions.values()))
    net_change = ((avg_new - avg_base) / avg_base) * 100 if avg_base > 0 else 0
    
    return jsonify({
        'predictions': {f"{k[0]}_{k[1]}_{k[2]}": float(v) for k, v in new_predictions.items()},
        'metrics': {
            'net_traffic_change_pct': float(net_change),
            'impacted_segments': len(impacted_edges),
            'peak_bottleneck_spike': float(max(deltas)) if deltas else 0.0,
            'avg_congestion': float(avg_new)
        },
        'top_5_critical_edges': top_5_critical,
        'impacted_edges': impacted_edges[:100]  # Limit to top 100
    })

if __name__ == '__main__':
    # Twin is already initialized at module level
    print("Starting Flask server...")
    print("=" * 60)
    print("📍 Server will be available at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

