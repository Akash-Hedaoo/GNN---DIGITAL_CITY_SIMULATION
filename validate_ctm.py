"""
Quick validation script to test CTM implementation.
This verifies that the CTM simulation produces valid output compatible with training.
"""
import networkx as nx
import sys
from step3_generate_training_data import simulate_congestion_event_ctm, generate_dataset

def test_ctm_basic():
    """Test basic CTM functionality"""
    print("🧪 Testing CTM Implementation...")
    
    try:
        # Load graph
        print("   Loading graph...")
        G = nx.read_graphml('real_city_processed.graphml')
        if not isinstance(G, nx.MultiDiGraph):
            G = nx.MultiDiGraph(G)
        
        # Prepare base data
        road_edges = []
        base_travel_times = {}
        node_populations = {}
        
        for u, v, k, data in G.edges(keys=True, data=True):
            is_metro = str(data.get('is_metro', 'False')) == 'True'
            try:
                val = data.get('base_travel_time', 1.0)
                if isinstance(val, list): val = val[0]
                tt = float(val)
            except: tt = 1.0
            
            base_travel_times[(u, v, k)] = tt
            if not is_metro:
                road_edges.append((u, v, k))
        
        for n, data in G.nodes(data=True):
            try:
                val = data.get('population', 0)
                if isinstance(val, list): val = val[0]
                pop = float(val)
            except: pop = 0.0
            node_populations[n] = pop
        
        base_data = {
            'road_edges': road_edges,
            'travel_times': base_travel_times,
            'populations': node_populations
        }
        
        # Run one simulation
        print("   Running CTM simulation...")
        snapshot = simulate_congestion_event_ctm(G, base_data)
        
        # Validate output structure
        assert hasattr(snapshot, 'edge_travel_times'), "Missing edge_travel_times"
        assert hasattr(snapshot, 'edge_congestion'), "Missing edge_congestion"
        assert hasattr(snapshot, 'closed_edges'), "Missing closed_edges"
        assert hasattr(snapshot, 'node_populations'), "Missing node_populations"
        
        # Validate data types
        assert isinstance(snapshot.edge_travel_times, dict), "edge_travel_times must be dict"
        assert isinstance(snapshot.edge_congestion, dict), "edge_congestion must be dict"
        assert isinstance(snapshot.closed_edges, list), "closed_edges must be list"
        
        # Validate values
        assert len(snapshot.edge_travel_times) > 0, "No travel times generated"
        assert len(snapshot.edge_congestion) > 0, "No congestion factors generated"
        
        # Check that congestion factors are reasonable (>= 1.0)
        for key, cf in snapshot.edge_congestion.items():
            assert cf >= 1.0, f"Congestion factor {cf} < 1.0 for edge {key}"
            assert cf < 100.0, f"Congestion factor {cf} too high for edge {key}"
        
        print("   ✅ All validation checks passed!")
        print(f"   Generated {len(snapshot.edge_travel_times)} edge predictions")
        print(f"   Closed {len(snapshot.closed_edges)} edges")
        
        # Show sample values
        sample_keys = list(snapshot.edge_congestion.keys())[:5]
        print("\n   Sample congestion factors:")
        for key in sample_keys:
            print(f"      {key}: {snapshot.edge_congestion[key]:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ctm_basic()
    sys.exit(0 if success else 1)







