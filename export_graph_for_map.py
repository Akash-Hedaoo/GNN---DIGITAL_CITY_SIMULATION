"""
Export GraphML data for mapping on frontend
Extracts all edges with their geometry coordinates
"""
import networkx as nx
import json
import re

def parse_linestring(geometry_str):
    """Parse WKT LINESTRING to coordinate pairs"""
    if not geometry_str:
        return []
    
    # Remove LINESTRING wrapper and parentheses
    coords_str = re.sub(r'^LINESTRING\s*\(|\)$', '', geometry_str.strip())
    
    # Parse coordinates
    coords = []
    for pair in coords_str.split(','):
        try:
            parts = pair.strip().split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                coords.append([x, y])
        except:
            pass
    
    return coords

def normalize_to_lat_lon(x, y):
    """Convert projected UTM coords to approximate Pune lat/lon"""
    # Original bounds
    x_min, x_max = 73.82, 379410.92
    y_min, y_max = 18.61, 2061677.82
    
    # Pune bounds
    pune_lat_min, pune_lat_max = 18.45, 18.65
    pune_lon_min, pune_lon_max = 73.75, 73.95
    
    # Normalize
    x_norm = (x - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0.5
    y_norm = (y - y_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0.5
    
    # Scale to Pune
    lat = pune_lat_min + y_norm * (pune_lat_max - pune_lat_min)
    lon = pune_lon_min + x_norm * (pune_lon_max - pune_lon_min)
    
    return [lat, lon]

def main():
    print("📂 Loading graph...")
    g = nx.read_graphml('real_city_processed.graphml')
    print(f"✅ Loaded: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    
    nodes = []
    edges = []
    
    print("🔄 Processing nodes...")
    for node_id, attrs in g.nodes(data=True):
        try:
            x = float(attrs.get('x', 0))
            y = float(attrs.get('y', 0))
            
            lat, lon = normalize_to_lat_lon(x, y)
            
            nodes.append({
                'id': str(node_id),
                'lat': lat,
                'lon': lon,
                'x': x,
                'y': y,
                'name': str(attrs.get('name', f'Node {node_id}'))[:50],
                'is_metro': attrs.get('is_metro_station', 'False') == 'True',
                'amenity_type': str(attrs.get('amenity_type', 'residential')),
                'population': int(attrs.get('population', 0)),
                'street_count': int(attrs.get('street_count', 0))
            })
        except Exception as e:
            pass
    
    print(f"✅ Processed {len(nodes)} nodes")
    
    print("🔄 Processing edges...")
    edge_count = 0
    for source, target, attrs in g.edges(data=True):
        try:
            # Get source and target node info
            source_node = g.nodes[source]
            target_node = g.nodes[target]
            
            source_x = float(source_node.get('x', 0))
            source_y = float(source_node.get('y', 0))
            target_x = float(target_node.get('x', 0))
            target_y = float(target_node.get('y', 0))
            
            source_lat, source_lon = normalize_to_lat_lon(source_x, source_y)
            target_lat, target_lon = normalize_to_lat_lon(target_x, target_y)
            
            # Parse geometry if available
            geometry = attrs.get('geometry', '')
            coords = parse_linestring(geometry)
            
            # If no geometry, create simple line
            if not coords:
                coords = [[source_x, source_y], [target_x, target_y]]
            
            # Convert all coords to lat/lon
            lat_lon_coords = [normalize_to_lat_lon(c[0], c[1]) for c in coords]
            
            edges.append({
                'id': f"{source}-{target}",
                'source': str(source),
                'target': str(target),
                'source_lat': source_lat,
                'source_lon': source_lon,
                'target_lat': target_lat,
                'target_lon': target_lon,
                'coordinates': lat_lon_coords,  # Full geometry as lat/lon
                'length': float(attrs.get('length', 0)),
                'highway': str(attrs.get('highway', 'unknown')),
                'congestion': float(attrs.get('congestion_level', 0.5)),
                'is_metro': attrs.get('is_metro', 'False') == 'True',
                'oneway': attrs.get('oneway', 'False') == 'True',
                'name': str(attrs.get('name', ''))[:100]
            })
            
            edge_count += 1
            if edge_count % 1000 == 0:
                print(f"  Processed {edge_count} edges...")
        except Exception as e:
            pass
    
    print(f"✅ Processed {len(edges)} edges")
    
    # Export to JSON
    data = {
        'city': 'Pune, Maharashtra, India',
        'nodes': nodes,
        'edges': edges,
        'bounds': {
            'north': 18.65,
            'south': 18.45,
            'east': 73.95,
            'west': 73.75
        },
        'statistics': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'avg_congestion': sum(e['congestion'] for e in edges) / len(edges) if edges else 0
        }
    }
    
    print("💾 Exporting to graph_full_data.json...")
    with open('graph_full_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Exported: {len(nodes)} nodes, {len(edges)} edges")
    print(f"📊 Average congestion: {data['statistics']['avg_congestion']:.2%}")
    print(f"📁 File: graph_full_data.json")

if __name__ == '__main__':
    main()
