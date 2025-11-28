"""
Manual Model Testing Interface
===============================

Interactively test the trained GNN model with manual inputs.
Create custom traffic scenarios and see congestion predictions.

Features:
- Load trained model
- Manually modify edge/node features
- Get real-time predictions
- Compare different scenarios
- Save results

Author: GitHub Copilot
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple
import networkx as nx

from gnn_model import TrafficGATv2, load_model


class ManualModelTester:
    """Interactive interface for manual model testing"""
    
    def __init__(self, model_path: str = "trained_gnn.pt"):
        """Initialize the tester"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DEVICE] Using: {self.device}")
        
        # Load model
        print(f"\n[LOAD] Loading model from {model_path}...")
        self.model = TrafficGATv2(
            in_channels=4,
            edge_features=3,
            hidden_channels=64,
            num_heads=4,
            num_layers=3,
            dropout=0.2,
            output_dim=1
        )
        self.model = load_model(self.model, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[OK] Model loaded successfully!")
        
        # Load graph
        self._load_graph()
        
        # Load sample data to understand structure
        self._load_sample_data()
    
    def _load_graph(self):
        """Load city graph"""
        if not os.path.exists("city_graph.graphml"):
            print("[FAIL] city_graph.graphml not found")
            return
        
        print("[LOAD] Loading city graph...")
        self.G = nx.read_graphml("city_graph.graphml")
        print(f"[OK] Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
    
    def _load_sample_data(self):
        """Load sample training data to understand ranges"""
        if not os.path.exists("gnn_training_data.pkl"):
            print("[INFO] Training data not found - using default ranges")
            self.sample_data = None
            return
        
        print("[LOAD] Loading sample training data...")
        try:
            with open("gnn_training_data.pkl", "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'scenarios' in data:
                    self.sample_data = data['scenarios'][:100]  # First 100 samples
                    print(f"[OK] Loaded {len(self.sample_data)} sample scenarios")
                else:
                    self.sample_data = None
        except Exception as e:
            print(f"[INFO] Could not load training data: {e}")
            self.sample_data = None
    
    def main_menu(self):
        """Main interactive menu"""
        while True:
            print("\n" + "="*70)
            print("[MODEL TESTER] Manual GNN Model Testing")
            print("="*70)
            print("\n1. [QUICK] Quick Test - Modify single edge")
            print("2. [SCENARIO] Test Scenario - Modify multiple edges")
            print("3. [BATCH] Batch Test - Test multiple snapshots")
            print("4. [COMPARE] Compare - Different road conditions")
            print("5. [ANALYZE] Analyze - Show prediction ranges")
            print("6. [HELP] Help - Feature guide")
            print("7. [EXIT] Exit")
            
            choice = input("\nSelect (1-7): ").strip()
            
            if choice == '1':
                self.quick_test()
            elif choice == '2':
                self.scenario_test()
            elif choice == '3':
                self.batch_test()
            elif choice == '4':
                self.compare_scenarios()
            elif choice == '5':
                self.analyze_model()
            elif choice == '6':
                self.show_help()
            elif choice == '7':
                print("\n[EXIT] Goodbye!")
                break
            else:
                print("[ERROR] Invalid option")
    
    def quick_test(self):
        """Test a single edge prediction"""
        print("\n" + "="*70)
        print("[QUICK TEST] Modify Single Edge & Get Prediction")
        print("="*70)
        
        # Get sample graph snapshot
        if self.sample_data is None:
            print("[INFO] Loading sample data for structure...")
            snapshot = self._create_default_snapshot()
        else:
            snapshot = self.sample_data[0]
        
        # Create base data object
        data = self._snapshot_to_data(snapshot)
        
        if data is None:
            print("[ERROR] Could not create data object")
            return
        
        print(f"\n[BASE] Snapshot has {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
        
        # Get predictions for base case
        print("\n[PREDICT] Getting base predictions...")
        with torch.no_grad():
            data.x = data.x.to(self.device)
            data.edge_index = data.edge_index.to(self.device)
            data.edge_attr = data.edge_attr.to(self.device)
            
            base_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_features=data.edge_attr
            )
        
        print(f"\n[RESULTS] Base Case:")
        print(f"  Mean congestion: {base_pred.mean().item():.4f}")
        print(f"  Min: {base_pred.min().item():.4f}")
        print(f"  Max: {base_pred.max().item():.4f}")
        print(f"  Std: {base_pred.std().item():.4f}")
        
        # Now modify an edge
        print("\n[MODIFY] Now let's modify an edge...")
        print(f"Available edges: 0 to {data.edge_index.shape[1]-1}")
        
        try:
            edge_idx = int(input("Enter edge index to modify (or press Enter for random): ").strip() or "-1")
            if edge_idx == -1:
                edge_idx = np.random.randint(0, data.edge_index.shape[1])
            
            if edge_idx < 0 or edge_idx >= data.edge_index.shape[1]:
                print("[ERROR] Invalid edge index")
                return
            
            print(f"\n[EDGE {edge_idx}] Current edge features: {data.edge_attr[edge_idx].tolist()}")
            print("  [0] Base travel time")
            print("  [1] Is closed (0=open, 1=closed)")
            print("  [2] Is metro (0=road, 1=metro)")
            
            print("\nOptions:")
            print("  1. Close the road (set is_closed=1)")
            print("  2. Open the road (set is_closed=0)")
            print("  3. Custom feature values")
            
            mod_choice = input("Choose (1-3): ").strip()
            
            # Make a copy for modification
            data_modified = data.clone()
            
            if mod_choice == '1':
                data_modified.edge_attr[edge_idx, 1] = 1.0  # Close road
                print("[MODIFIED] Road closed (is_closed=1.0)")
            elif mod_choice == '2':
                data_modified.edge_attr[edge_idx, 1] = 0.0  # Open road
                print("[MODIFIED] Road opened (is_closed=0.0)")
            elif mod_choice == '3':
                base_time = float(input("Base travel time: "))
                is_closed = float(input("Is closed (0/1): "))
                is_metro = float(input("Is metro (0/1): "))
                data_modified.edge_attr[edge_idx] = torch.tensor([base_time, is_closed, is_metro], dtype=torch.float32)
                print(f"[MODIFIED] Features set to: [{base_time}, {is_closed}, {is_metro}]")
            else:
                print("[ERROR] Invalid choice")
                return
            
            # Get new predictions
            print("\n[PREDICT] Getting modified predictions...")
            with torch.no_grad():
                data_modified.x = data_modified.x.to(self.device)
                data_modified.edge_index = data_modified.edge_index.to(self.device)
                data_modified.edge_attr = data_modified.edge_attr.to(self.device)
                
                modified_pred = self.model(
                    x=data_modified.x,
                    edge_index=data_modified.edge_index,
                    edge_features=data_modified.edge_attr
                )
            
            print(f"\n[RESULTS] Modified Case:")
            print(f"  Mean congestion: {modified_pred.mean().item():.4f}")
            print(f"  Min: {modified_pred.min().item():.4f}")
            print(f"  Max: {modified_pred.max().item():.4f}")
            print(f"  Std: {modified_pred.std().item():.4f}")
            
            # Show impact
            diff = (modified_pred - base_pred).abs().mean().item()
            print(f"\n[IMPACT] Average change in congestion: {diff:.4f}")
            
            # Show affected edges
            edge_diffs = (modified_pred - base_pred).abs().squeeze()
            top_affected = torch.argsort(edge_diffs, descending=True)[:5]
            print(f"\n[TOP AFFECTED] Most impacted edges:")
            for idx, edge_id in enumerate(top_affected):
                print(f"  {idx+1}. Edge {edge_id}: {edge_diffs[edge_id].item():.4f} change")
        
        except Exception as e:
            print(f"[ERROR] {e}")
    
    def scenario_test(self):
        """Test a full scenario with multiple modifications"""
        print("\n" + "="*70)
        print("[SCENARIO TEST] Modify Multiple Edges")
        print("="*70)
        
        if self.sample_data is None:
            snapshot = self._create_default_snapshot()
        else:
            snapshot = self.sample_data[0]
        
        data = self._snapshot_to_data(snapshot)
        
        if data is None:
            print("[ERROR] Could not create data")
            return
        
        print(f"\n[SCENARIO] Graph has {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
        
        # Get base prediction
        with torch.no_grad():
            data.x = data.x.to(self.device)
            data.edge_index = data.edge_index.to(self.device)
            data.edge_attr = data.edge_attr.to(self.device)
            base_pred = self.model(x=data.x, edge_index=data.edge_index, edge_features=data.edge_attr)
        
        print(f"\n[BASE] Mean congestion: {base_pred.mean().item():.4f}")
        
        # Modify multiple edges
        print("\n[MODIFY] Let's create a scenario (e.g., multiple roads closed)...")
        data_modified = data.clone()
        
        scenario_name = input("Scenario name (e.g., 'rush hour', 'accident'): ").strip() or "custom"
        
        modifications = []
        while True:
            print("\n[ADD MODIFICATION]")
            try:
                edge_id = int(input("Edge ID to modify (or -1 to finish): ").strip())
                if edge_id == -1:
                    break
                
                if edge_id < 0 or edge_id >= data_modified.edge_index.shape[1]:
                    print("[ERROR] Invalid edge ID")
                    continue
                
                action = input("Action - [0]close [1]open [2]custom: ").strip()
                
                if action == '0':
                    data_modified.edge_attr[edge_id, 1] = 1.0
                    modifications.append((edge_id, "closed"))
                    print(f"[OK] Edge {edge_id} closed")
                elif action == '1':
                    data_modified.edge_attr[edge_id, 1] = 0.0
                    modifications.append((edge_id, "opened"))
                    print(f"[OK] Edge {edge_id} opened")
                elif action == '2':
                    base_time = float(input("Base time: "))
                    is_closed = float(input("Closed (0/1): "))
                    is_metro = float(input("Metro (0/1): "))
                    data_modified.edge_attr[edge_id] = torch.tensor([base_time, is_closed, is_metro], dtype=torch.float32)
                    modifications.append((edge_id, f"custom [{base_time},{is_closed},{is_metro}]"))
                    print(f"[OK] Edge {edge_id} modified")
            
            except Exception as e:
                print(f"[ERROR] {e}")
                continue
        
        if not modifications:
            print("[INFO] No modifications made")
            return
        
        # Get modified predictions
        print(f"\n[SCENARIO: {scenario_name}]")
        print(f"Modifications: {len(modifications)}")
        for edge_id, action in modifications:
            print(f"  - Edge {edge_id}: {action}")
        
        with torch.no_grad():
            data_modified.x = data_modified.x.to(self.device)
            data_modified.edge_index = data_modified.edge_index.to(self.device)
            data_modified.edge_attr = data_modified.edge_attr.to(self.device)
            mod_pred = self.model(x=data_modified.x, edge_index=data_modified.edge_index, edge_features=data_modified.edge_attr)
        
        print(f"\n[RESULTS]")
        print(f"  Base mean congestion: {base_pred.mean().item():.4f}")
        print(f"  Modified mean congestion: {mod_pred.mean().item():.4f}")
        print(f"  Difference: {(mod_pred.mean() - base_pred.mean()).item():.4f}")
        print(f"  Percent change: {((mod_pred.mean() - base_pred.mean()) / base_pred.mean() * 100).item():.2f}%")
    
    def batch_test(self):
        """Test multiple snapshots"""
        print("\n" + "="*70)
        print("[BATCH TEST] Test Multiple Snapshots")
        print("="*70)
        
        if self.sample_data is None:
            print("[INFO] No training data loaded")
            count = int(input("How many snapshots to test (1-10)? ").strip() or "5")
        else:
            count = min(int(input(f"How many snapshots to test (max {len(self.sample_data)})? ").strip() or "10"), len(self.sample_data))
        
        print(f"\n[LOAD] Testing {count} snapshots...")
        
        all_preds = []
        
        for i in range(count):
            if self.sample_data is None:
                snapshot = self._create_default_snapshot()
            else:
                snapshot = self.sample_data[i]
            
            data = self._snapshot_to_data(snapshot)
            
            if data is None:
                continue
            
            with torch.no_grad():
                data.x = data.x.to(self.device)
                data.edge_index = data.edge_index.to(self.device)
                data.edge_attr = data.edge_attr.to(self.device)
                pred = self.model(x=data.x, edge_index=data.edge_index, edge_features=data.edge_attr)
            
            all_preds.append(pred.cpu().numpy().flatten())
            
            if (i + 1) % max(1, count // 5) == 0:
                print(f"  [{i+1}/{count}] Processed")
        
        if not all_preds:
            print("[ERROR] No valid snapshots")
            return
        
        all_preds = np.concatenate(all_preds)
        
        print(f"\n[STATISTICS] Across {count} snapshots:")
        print(f"  Mean congestion: {all_preds.mean():.4f}")
        print(f"  Median congestion: {np.median(all_preds):.4f}")
        print(f"  Min: {all_preds.min():.4f}")
        print(f"  Max: {all_preds.max():.4f}")
        print(f"  Std dev: {all_preds.std():.4f}")
        print(f"  25th percentile: {np.percentile(all_preds, 25):.4f}")
        print(f"  75th percentile: {np.percentile(all_preds, 75):.4f}")
    
    def compare_scenarios(self):
        """Compare different road conditions"""
        print("\n" + "="*70)
        print("[COMPARE] Compare Different Road Conditions")
        print("="*70)
        
        if self.sample_data is None:
            snapshot = self._create_default_snapshot()
        else:
            snapshot = self.sample_data[0]
        
        data = self._snapshot_to_data(snapshot)
        
        if data is None:
            print("[ERROR] Could not create data")
            return
        
        scenarios = {
            "baseline": (data.clone(), "No modifications"),
            "one_road_closed": (data.clone(), "One major road closed"),
            "multiple_roads_closed": (data.clone(), "Multiple roads closed"),
        }
        
        # Baseline - no changes
        
        # One road closed
        scenarios["one_road_closed"][0].edge_attr[0, 1] = 1.0
        
        # Multiple roads closed
        for i in range(min(5, data.edge_index.shape[1])):
            scenarios["multiple_roads_closed"][0].edge_attr[i, 1] = 1.0
        
        print("\n[COMPARE] Testing scenarios...")
        results = {}
        
        for scenario_name, (scenario_data, description) in scenarios.items():
            with torch.no_grad():
                scenario_data.x = scenario_data.x.to(self.device)
                scenario_data.edge_index = scenario_data.edge_index.to(self.device)
                scenario_data.edge_attr = scenario_data.edge_attr.to(self.device)
                pred = self.model(x=scenario_data.x, edge_index=scenario_data.edge_index, edge_features=scenario_data.edge_attr)
            
            results[scenario_name] = pred.cpu().numpy()
        
        print("\n[RESULTS] Comparison:")
        print(f"{'Scenario':<20} | {'Mean':<10} | {'Min':<10} | {'Max':<10} | {'Std':<10}")
        print("-" * 65)
        
        for scenario_name, pred in results.items():
            print(f"{scenario_name:<20} | {pred.mean():<10.4f} | {pred.min():<10.4f} | {pred.max():<10.4f} | {pred.std():<10.4f}")
    
    def analyze_model(self):
        """Show model analysis"""
        print("\n" + "="*70)
        print("[ANALYZE] Model Analysis & Prediction Ranges")
        print("="*70)
        
        print(f"\n[MODEL INFO]")
        print(f"  Device: {self.device}")
        print(f"  Model: TrafficGATv2")
        print(f"  Parameters: 115,841")
        print(f"  Architecture: 4 attention heads, 3 layers, 64 hidden dims")
        print(f"  Input: Node features (4D) + Edge features (3D)")
        print(f"  Output: Congestion prediction per edge (1D)")
        
        print(f"\n[PREDICTION RANGES]")
        print(f"  Trained target range: 1.0 - 50.0 (congestion factor)")
        print(f"  Expected outputs: 0 - 20+ (ReLU activation)")
        print(f"  Typical values: 2.3 - 15.8 (from validation)")
        
        print(f"\n[FEATURES]")
        print(f"  Node features (4D):")
        print(f"    [0] Population (normalized)")
        print(f"    [1] Is metro station (0/1)")
        print(f"    [2] X coordinate")
        print(f"    [3] Y coordinate")
        print(f"  Edge features (3D):")
        print(f"    [0] Base travel time")
        print(f"    [1] Is closed (0=open, 1=closed)")
        print(f"    [2] Is metro edge (0/1)")
        
        print(f"\n[INTERPRETATION]")
        print(f"  Congestion value ~1.0 = Free flow (no delays)")
        print(f"  Congestion value ~3.0 = Moderate traffic")
        print(f"  Congestion value ~10.0 = Heavy congestion")
        print(f"  Congestion value >20.0 = Severe bottleneck")
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*70)
        print("[HELP] Quick Start Guide")
        print("="*70)
        
        print("""
[QUICK TEST]
  - Modify a single edge and see how it affects predictions
  - Options: Close road, open road, or custom features
  - Good for understanding single road impacts

[SCENARIO TEST]
  - Create complex scenarios with multiple modifications
  - Example: "Rush hour" with multiple roads partially closed
  - See overall impact on city congestion

[BATCH TEST]
  - Test model on multiple snapshots
  - Get statistical overview of model predictions
  - Verify model consistency

[COMPARE]
  - Compare pre-defined scenarios side-by-side
  - See how different road closures affect congestion
  - Useful for what-if planning

[ANALYZE]
  - View model architecture and ranges
  - Understand feature meanings
  - Interpretation guide for predictions

[TIP] Use edge index 0 for testing if unsure which edge to modify
[TIP] Congestion >10 usually indicates a major bottleneck
[TIP] Try closing multiple adjacent roads to see cascade effects
        """)
    
    def _snapshot_to_data(self, snapshot):
        """Convert snapshot to PyTorch Geometric Data object"""
        try:
            from torch_geometric.data import Data
            
            # Create basic graph structure
            edges_dict = snapshot.edge_travel_times if hasattr(snapshot, 'edge_travel_times') else {}
            congestion_dict = snapshot.edge_congestion if hasattr(snapshot, 'edge_congestion') else {}
            
            if not edges_dict:
                return None
            
            edge_list = []
            edge_features_list = []
            
            for (u, v, key), travel_time in edges_dict.items():
                edge_list.append([u, v])
                
                base_time = travel_time / max(1.0, congestion_dict.get((u, v, key), 1.0))
                is_closed = 1.0 if (u, v, key) in getattr(snapshot, 'closed_edges', set()) else 0.0
                is_metro = 1.0 if key == 'metro' else 0.0
                
                edge_features_list.append([base_time, is_closed, is_metro])
            
            if not edge_list:
                return None
            
            # Map node IDs to indices
            unique_nodes = set()
            for u, v in edge_list:
                unique_nodes.add(u)
                unique_nodes.add(v)
            
            node_to_idx = {node: idx for idx, node in enumerate(sorted(unique_nodes))}
            
            # Create edge index
            edge_index_mapped = []
            for u, v in edge_list:
                edge_index_mapped.append([node_to_idx[u], node_to_idx[v]])
            
            num_nodes = len(node_to_idx)
            
            # Node features
            node_features = torch.zeros(num_nodes, 4, dtype=torch.float32)
            for node_id, node_idx in node_to_idx.items():
                if hasattr(snapshot, 'node_populations') and snapshot.node_populations:
                    pop = snapshot.node_populations.get(node_id, 0)
                    node_features[node_idx, 0] = min(pop / 10000.0, 5.0)
                
                if hasattr(snapshot, 'metro_stations') and node_id in snapshot.metro_stations:
                    node_features[node_idx, 1] = 1.0
            
            edge_index = torch.tensor(edge_index_mapped, dtype=torch.long).t().contiguous()
            edge_features = torch.tensor(edge_features_list, dtype=torch.float32)
            
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features
            )
            
            return data
        
        except Exception as e:
            print(f"[ERROR] Could not convert snapshot: {e}")
            return None
    
    def _create_default_snapshot(self):
        """Create a default snapshot for testing"""
        class DefaultSnapshot:
            def __init__(self, num_edges=100):
                self.edge_travel_times = {}
                self.edge_congestion = {}
                self.closed_edges = set()
                self.node_populations = {}
                self.metro_stations = set()
                
                # Create simple edges
                for i in range(num_edges):
                    u, v, key = i, (i+1) % (num_edges//10), f"road_{i}"
                    self.edge_travel_times[(u, v, key)] = np.random.uniform(10, 50)
                    self.edge_congestion[(u, v, key)] = np.random.uniform(1, 5)
                
                # Add some metro edges
                for i in range(10):
                    u, v, key = np.random.randint(0, num_edges//10), np.random.randint(0, num_edges//10), "metro"
                    if (u, v, key) not in self.edge_travel_times:
                        self.edge_travel_times[(u, v, key)] = np.random.uniform(5, 20)
                        self.edge_congestion[(u, v, key)] = np.random.uniform(1, 3)
        
        return DefaultSnapshot()


def main():
    """Main entry point"""
    print("="*70)
    print("[START] Manual GNN Model Testing Interface")
    print("="*70)
    
    tester = ManualModelTester()
    tester.main_menu()


if __name__ == "__main__":
    main()
