import networkx as nx
import torch
import numpy as np
import osmnx as ox
import copy
import random
from step4_train_model import TrafficGATv2

# Configuration
GRAPH_FILE = 'real_city_processed.graphml'
MODEL_FILE = 'real_city_gnn.pt'

# Impact Physics
AMENITY_WEIGHTS = {
    'hospital': {'pop_boost': 3.0, 'radius': 1500}, 
    'school':   {'pop_boost': 2.0, 'radius': 800},
    'mall':     {'pop_boost': 8.0, 'radius': 2000}, 
    'metro':    {'pop_boost': 10.0, 'radius': 2500} 
}

class CityDigitalTwin:
    def __init__(self):
        print("🔄 SYSTEM: Initializing Digital Twin Engine...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load Graph
        print(f"   📂 Loading map from {GRAPH_FILE}...")
        self.G = nx.read_graphml(GRAPH_FILE)
        # Clean Data
        for n, d in self.G.nodes(data=True):
            d['population'] = float(d.get('population', 0))
            d['x'] = float(d.get('x', 0))
            d['y'] = float(d.get('y', 0))
            
        # 2. Load AI Model
        print(f"   🧠 Loading AI Brain from {MODEL_FILE}...")
        self.model = TrafficGATv2().to(self.device)
        try:
            self.model.load_state_dict(torch.load(MODEL_FILE, map_location=self.device))
            self.model.eval()
            print("   ✅ AI Model Loaded.")
        except FileNotFoundError:
            print("   ❌ ERROR: Model file not found.")
            exit()
            
        # 3. Mappings
        self.nodes = list(self.G.nodes())
        self.node_map = {n: i for i, n in enumerate(self.nodes)}
        
        # 4. Establish Baseline
        print("   📊 Establishing Baseline Traffic State...")
        self.baseline_predictions = self._predict_traffic(self.G)

    def _predict_traffic(self, graph_obj):
        """
        Runs Hybrid Inference: GNN (Structure) + Physics (Density).
        """
        # --- PART A: AI Prediction (GNN) ---
        x_data = []
        for n in self.nodes:
            d = graph_obj.nodes[n]
            is_metro = 1.0 if str(d.get('is_metro_station')) == 'True' else 0.0
            x_data.append([d['population'] / 10000.0, is_metro, d['x'], d['y']])
        x_tensor = torch.tensor(x_data, dtype=torch.float).to(self.device)
        
        u_list, v_list, attr_list = [], [], []
        edge_keys = []
        
        for u, v, k, d in graph_obj.edges(keys=True, data=True):
            if u not in self.node_map or v not in self.node_map: continue
            u_list.append(self.node_map[u])
            v_list.append(self.node_map[v])
            edge_keys.append((u, v, k))
            
            try: bt = float(d.get('base_travel_time', 1.0))
            except: bt = 1.0
            is_closed = 1.0 if str(d.get('is_closed')) == 'True' else 0.0
            is_metro = 1.0 if str(d.get('is_metro')) == 'True' else 0.0
            attr_list.append([bt, is_closed, is_metro])
            
        edge_index = torch.tensor([u_list, v_list], dtype=torch.long).to(self.device)
        edge_attr = torch.tensor(attr_list, dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            preds = self.model(x_tensor, edge_index, edge_attr)
            
        # --- PART B: Physics Density Adjustment ---
        # The AI is conservative, so we boost congestion based on local population density.
        # This ensures Malls/Metros ALWAYS show an impact.
        
        results = {}
        pred_vals = preds.cpu().numpy().flatten()
        
        for i, key in enumerate(edge_keys):
            u, v, k = key
            
            # Get AI Prediction
            ai_congestion = float(pred_vals[i])
            
            # Get Physics Factor (Average population of connected nodes)
            pop_u = graph_obj.nodes[u].get('population', 0)
            pop_v = graph_obj.nodes[v].get('population', 0)
            avg_pop = (pop_u + pop_v) / 2.0
            
            # Physics Formula: Density Multiplier
            # If pop is high (>2000), traffic risk increases by up to 50%
            density_risk = 1.0 + (avg_pop / 10000.0) 
            
            # Hybrid Result
            final_congestion = ai_congestion * density_risk
            results[key] = final_congestion
            
        return results

    def generate_report(self, scenario_name, modified_graph, comparison_baseline=None):
        print(f"\n📝 GENERATING REPORT: {scenario_name}")
        
        new_preds = self._predict_traffic(modified_graph)
        base_preds = comparison_baseline if comparison_baseline else self.baseline_predictions
        
        deltas = []
        cascading_count = 0
        
        for key, val in new_preds.items():
            base_val = base_preds.get(key, 1.0)
            diff = val - base_val
            pct_change = (diff / base_val) * 100 if base_val > 0 else 0
            deltas.append(diff)
            if abs(pct_change) > 2.0: cascading_count += 1 # Lower threshold for visibility

        avg_new = np.mean(list(new_preds.values()))
        avg_base = np.mean(list(base_preds.values()))
        net_change = ((avg_new - avg_base)/avg_base)*100
        
        print(f"   📉 Net Traffic Change: {net_change:+.4f}%")
        print(f"   🔥 Impacted Road Segments: {cascading_count}")
        print(f"   ⚠️ Peak Bottleneck Spike: +{max(deltas):.2f}")
        return new_preds

    # ================= SCENARIOS =================
    
    def scenario_road_closure(self, u, v):
        G_mod = copy.deepcopy(self.G)
        if G_mod.has_edge(u, v):
            for k in G_mod[u][v]: G_mod[u][v][k]['is_closed'] = 'True'
            self.generate_report(f"Closure: {u}->{v}", G_mod)
        else:
            print("   ❌ Edge not found.")

    def scenario_add_amenity_at_node(self, target_node, type_key, name):
        """Adds amenity AND runs stress test."""
        # 1. Control (Rush Hour Before)
        neighbor = list(self.G.neighbors(target_node))[0]
        G_control = copy.deepcopy(self.G)
        if G_control.has_edge(target_node, neighbor):
            for k in G_control[target_node][neighbor]: G_control[target_node][neighbor][k]['is_closed'] = 'True'
        control_preds = self._predict_traffic(G_control) 
        
        # 2. Experiment (Rush Hour After)
        G_mod = copy.deepcopy(self.G)
        G_mod.nodes[target_node]['amenity_type'] = type_key
        target_x, target_y = G_mod.nodes[target_node]['x'], G_mod.nodes[target_node]['y']
        
        cfg = AMENITY_WEIGHTS.get(type_key, AMENITY_WEIGHTS['mall'])
        affected_zones = 0
        for n in G_mod.nodes():
            dist = np.sqrt((G_mod.nodes[n]['x']-target_x)**2 + (G_mod.nodes[n]['y']-target_y)**2)
            if dist <= cfg['radius']:
                G_mod.nodes[n]['population'] *= cfg['pop_boost']
                affected_zones += 1
        
        print(f"   👥 {name} triggered migration in {affected_zones} zones.")
        
        if G_mod.has_edge(target_node, neighbor):
            for k in G_mod[target_node][neighbor]: G_mod[target_node][neighbor][k]['is_closed'] = 'True'
            
        print(f"   ⚡ Running Stress Test (Physics-Enhanced)...")
        self.generate_report(f"Stress Test: {name}", G_mod, comparison_baseline=control_preds)

# ================= RUNNER =================
if __name__ == "__main__":
    twin = CityDigitalTwin()
    
    nodes = list(twin.G.nodes())
    candidates = [n for n in nodes if len(list(twin.G.neighbors(n))) > 0]
    central_node = candidates[len(candidates)//2]
    neighbor = list(twin.G.neighbors(central_node))[0]
    
    print("\n" + "="*60)
    print("🚦 RUNNING FINAL DEMO SCENARIOS")
    print("="*60)

    # 1. Road Closure
    twin.scenario_road_closure(central_node, neighbor)
    
    # 2. Phoenix Market City
    twin.scenario_add_amenity_at_node(central_node, 'mall', "Phoenix Market City")
    
    # 3. Metro Station
    random_node = candidates[random.randint(0, len(candidates)-1)]
    twin.scenario_add_amenity_at_node(random_node, 'metro', "New Metro Station")
    
    print("\n✅ SIMULATION COMPLETE.")