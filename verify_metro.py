"""Quick verification script to check metro network in the graph"""
import networkx as nx

print("🔍 Verifying Metro Network in Graph...")
print("="*60)

G = nx.read_graphml("city_graph.graphml")

# Convert to MultiDiGraph if needed
if not isinstance(G, nx.MultiDiGraph):
    G = nx.MultiDiGraph(G)

# Count metro edges
metro_edges = []
metro_stations = set()

for u, v, key, data in G.edges(keys=True, data=True):
    if data.get('is_metro') or data.get('highway') in ['railway', 'metro_railway']:
        metro_edges.append((u, v, key, data))
        metro_stations.add(u)
        metro_stations.add(v)

print(f"\n📊 Graph Summary:")
print(f"   Total Nodes: {G.number_of_nodes()}")
print(f"   Total Edges: {G.number_of_edges()}")

print(f"\n🚇 Metro Network:")
print(f"   Metro Edges: {len(metro_edges)}")
print(f"   Metro Stations: {len(metro_stations)}")

# Check metro lines
lines_found = {}
for u, v, key, data in metro_edges[:10]:  # Sample first 10
    line_name = data.get('line_name', 'Unknown')
    line_num = data.get('line_number', -1)
    if line_name not in lines_found:
        lines_found[line_name] = {
            'number': line_num,
            'color': data.get('line_color', 'N/A'),
            'speed': data.get('maxspeed', 'N/A'),
            'count': 0
        }
    lines_found[line_name]['count'] += 1

print(f"\n📍 Metro Lines Found:")
for line_name, info in lines_found.items():
    print(f"   {line_name}:")
    print(f"      Line Number: {info['number']}")
    print(f"      Color: {info['color']}")
    print(f"      Speed: {info['speed']} km/h")
    print(f"      Sample Edges: {info['count']}")

# Sample metro edge details
if metro_edges:
    print(f"\n🔎 Sample Metro Edge Details:")
    u, v, key, data = metro_edges[0]
    print(f"   From Node: {u} → To Node: {v}")
    print(f"   Edge Key: {key}")
    print(f"   Attributes:")
    for attr, value in sorted(data.items()):
        print(f"      {attr}: {value}")

# Check metro station nodes
metro_station_nodes = [node for node, data in G.nodes(data=True) 
                       if data.get('amenity') == 'metro_station' or 
                       data.get('metro_station')]

print(f"\n🚉 Metro Station Nodes: {len(metro_station_nodes)}")
if metro_station_nodes:
    sample_node = metro_station_nodes[0]
    node_data = G.nodes[sample_node]
    print(f"\n   Sample Station (Node {sample_node}):")
    for attr, value in sorted(node_data.items()):
        if not attr.startswith('_'):
            print(f"      {attr}: {value}")

print("\n" + "="*60)
print("✅ Verification Complete!")
