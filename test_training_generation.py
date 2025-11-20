"""
Quick test of training data generation with smaller sample
"""

import networkx as nx
from generate_training_data import generate_training_scenarios

print("🌆 Loading city graph...")

try:
    G = nx.read_graphml('city_graph.graphml')
    print(f"✅ Loaded city graph")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
except FileNotFoundError:
    print("❌ city_graph.graphml not found!")
    print("   Please run: python generate_complex_city.py")
    exit(1)

# Generate small test dataset
print("\n🧪 Running test with 5 scenarios...")
generate_training_scenarios(
    graph=G,
    num_scenarios=5,  # Small sample for testing
    duration_per_scenario=30,  # 30 minutes each
    output_file='test_training_data.pkl'
)

print("\n✅ Test complete! Check test_training_data.pkl")
