"""
Analyze Pune city graph and generate visualization data
"""
import networkx as nx
import json
import math

print("🔍 ANALYZING PUNE CITY GRAPH...")
G = nx.read_graphml('real_city_processed.graphml')

print(f"\n📊 GRAPH STATISTICS")
print(f"=" * 60)
print(f"Total Nodes: {G.number_of_nodes()}")
print(f"Total Edges: {G.number_of_edges()}")
print(f"Graph Type: {'Directed' if G.is_directed() else 'Undirected'}")

# Collect nodes
nodes_list = []
metros = []
amenities = []

for node_id, attrs in G.nodes(data=True):
    try:
        x = float(attrs.get('x', 0))
        y = float(attrs.get('y', 0))
        is_metro = str(attrs.get('is_metro_station', 'False')).lower() == 'true'
        amenity_type = attrs.get('amenity_type', 'residential')
        
        if x != 0 and y != 0:
            node_data = {
                'id': str(node_id),
                'x': x,
                'y': y,
                'is_metro': is_metro,
                'amenity_type': amenity_type
            }
            
            if is_metro:
                node_data['name'] = attrs.get('name', f'Station {node_id}')
                metros.append(node_data)
            elif amenity_type != 'residential':
                node_data['name'] = attrs.get('name', f'{amenity_type} {node_id}')
                amenities.append(node_data)
            
            nodes_list.append(node_data)
    except (ValueError, TypeError):
        pass

# Collect edges
edges_list = []
for u, v, attrs in G.edges(data=True):
    try:
        length = float(attrs.get('length', 0))
        congestion = float(attrs.get('congestion_level', 0.5))
        is_metro = str(attrs.get('is_metro', 'False')).lower() == 'true'
        
        edge_data = {
            'id': f"{u}-{v}",
            'source': str(u),
            'target': str(v),
            'length': length,
            'congestion': max(0, min(1, congestion)),  # Clamp 0-1
            'is_metro': is_metro
        }
        edges_list.append(edge_data)
    except (ValueError, TypeError):
        pass

# Calculate bounds
xs = [n['x'] for n in nodes_list]
ys = [n['y'] for n in nodes_list]
bounds = {
    'minX': min(xs),
    'maxX': max(xs),
    'minY': min(ys),
    'maxY': max(ys),
    'centerX': (min(xs) + max(xs)) / 2,
    'centerY': (min(ys) + max(ys)) / 2,
    'rangeX': max(xs) - min(xs),
    'rangeY': max(ys) - min(ys)
}

print(f"\n📍 COORDINATE SYSTEM")
print(f"=" * 60)
print(f"Type: Projected (likely UTM Zone 44N for Pune)")
print(f"X (Easting):  {bounds['minX']:.2f} to {bounds['maxX']:.2f} (range: {bounds['rangeX']:.2f})")
print(f"Y (Northing): {bounds['minY']:.2f} to {bounds['maxY']:.2f} (range: {bounds['rangeY']:.2f})")
print(f"Center: ({bounds['centerX']:.2f}, {bounds['centerY']:.2f})")

print(f"\n🗺️  PUNE AMENITIES")
print(f"=" * 60)
print(f"Total Amenities (POIs): {len(amenities)}")
amenity_types = {}
for a in amenities:
    t = a['amenity_type']
    amenity_types[t] = amenity_types.get(t, 0) + 1

for atype, count in sorted(amenity_types.items(), key=lambda x: -x[1])[:10]:
    print(f"  {atype}: {count}")

print(f"\n🚇 METRO STATIONS")
print(f"=" * 60)
print(f"Total Metro Stations: {len(metros)}")
for metro in metros[:5]:
    print(f"  {metro.get('name', 'Unknown')}")

print(f"\n📝 NODES & EDGES SUMMARY")
print(f"=" * 60)
print(f"Total Nodes in Graph: {len(nodes_list)}")
print(f"Total Edges in Graph: {len(edges_list)}")
print(f"Nodes with coordinates: {len(nodes_list)}")
print(f"Edges with valid data: {len(edges_list)}")

# Export for frontend
output = {
    'city': 'Pune, Maharashtra, India',
    'nodes': nodes_list[:1000],  # Limit for frontend
    'edges': edges_list[:2000],
    'metros': metros,
    'amenities': amenities[:500],
    'bounds': bounds,
    'summary': {
        'total_nodes': len(nodes_list),
        'total_edges': len(edges_list),
        'metro_stations': len(metros),
        'amenities': len(amenities),
        'coordinate_system': 'UTM Zone 44N (projected)'
    }
}

with open('city_graph_data.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ EXPORTED: city_graph_data.json ({len(output['nodes'])} nodes, {len(output['edges'])} edges)")
