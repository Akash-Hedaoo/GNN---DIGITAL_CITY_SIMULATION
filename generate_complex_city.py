import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
import random
import math

# --- CONFIGURATION ---
NUM_NODES = 800  # High density for complexity
CITY_CENTER_LAT = 18.5204
CITY_CENTER_LON = 73.8567
SCALE = 0.02  # Wider area coverage

def generate_organic_city():
    print(f"🏗️  Generating Organic City with {NUM_NODES} nodes...")

    # 1. Generate Random Points (Organic Intersections)
    # We use a normal distribution to cluster more nodes in the center (Downtown)
    # and fewer in the outskirts (Suburbs).
    points = []
    for _ in range(NUM_NODES):
        # Mix of uniform (spread out) and normal (clustered)
        if random.random() > 0.3:
            x = np.random.normal(0, 0.6) # Cluster near center
            y = np.random.normal(0, 0.6)
        else:
            x = np.random.uniform(-1.5, 1.5) # Spread out
            y = np.random.uniform(-1.5, 1.5)
        points.append([x, y])
    
    points = np.array(points)

    # 2. Create Structure using Delaunay Triangulation
    # This creates a "Spider web" of non-overlapping connections
    tri = Delaunay(points)
    
    # Create Graph from Triangulation
    G_temp = nx.Graph()
    for simplex in tri.simplices:
        # Connect the 3 points of the triangle
        G_temp.add_edge(simplex[0], simplex[1])
        G_temp.add_edge(simplex[1], simplex[2])
        G_temp.add_edge(simplex[2], simplex[0])

    # 3. Prune the Graph (Make it look like streets, not math)
    # Remove distinct long edges (outliers) and random edges to create blocks
    edges_to_remove = []
    for u, v in G_temp.edges():
        p1 = points[u]
        p2 = points[v]
        dist = np.linalg.norm(p1 - p2)
        
        # Remove very long edges (outliers) or random 20% of internal edges
        if dist > 0.5 or (dist < 0.1 and random.random() > 0.7):
            edges_to_remove.append((u, v))
            
    G_temp.remove_edges_from(edges_to_remove)
    
    # Remove isolated nodes
    G_temp.remove_nodes_from(list(nx.isolates(G_temp)))

    # 4. Convert to MultiDiGraph (OSM Standard) & Add Attributes
    G = nx.MultiDiGraph()
    
    # Remap nodes to proper GPS
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(G_temp.nodes())}
    
    for old_id, new_id in node_mapping.items():
        x_rel, y_rel = points[old_id]
        
        real_x = CITY_CENTER_LON + (x_rel * SCALE)
        real_y = CITY_CENTER_LAT + (y_rel * SCALE)
        
        # Assign Zones based on distance/direction
        dist_from_center = math.sqrt(x_rel**2 + y_rel**2)
        
        if dist_from_center < 0.4:
            zone = "downtown"
            color = "blue"
        elif x_rel < -0.5 and y_rel < -0.5:
            zone = "industrial"
            color = "red"
        elif x_rel > 0.5 and y_rel > 0.5:
            zone = "residential"
            color = "green"
        else:
            zone = "suburbs"
            color = "gray"
            
        G.add_node(new_id, x=real_x, y=real_y, zone=zone, color=color)

    # 5. Add Edges & Highways
    # We identify a "Ring Road" (nodes at a certain radius)
    
    for u_old, v_old in G_temp.edges():
        u, v = node_mapping[u_old], node_mapping[v_old]
        
        # Calculate Physics
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        dist_deg = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        length_meters = dist_deg * 111000
        
        # Logic for Highways:
        # If both nodes are "Downtown", it's busy.
        # If both nodes are at radius ~0.8, it's a Ring Road.
        dist_u = math.sqrt(points[u_old][0]**2 + points[u_old][1]**2)
        
        is_highway = False
        if 0.7 < dist_u < 0.9 and random.random() > 0.3: # Ring Road
            is_highway = True
        
        # Create Two-Way Street
        for source, target in [(u, v), (v, u)]:
            attr = {
                'osmid': f"{source}-{target}",
                'length': length_meters,
                'is_closed': 0,
                'oneway': False
            }
            
            if is_highway:
                attr['highway'] = 'primary'
                attr['name'] = "Ring Road"
                attr['lanes'] = 3
                attr['maxspeed'] = 60
            else:
                attr['highway'] = 'residential'
                attr['name'] = "Street"
                attr['lanes'] = 1
                attr['maxspeed'] = 30
            
            speed_mps = attr['maxspeed'] / 3.6
            attr['base_travel_time'] = length_meters / speed_mps
            attr['current_travel_time'] = attr['base_travel_time']
            
            G.add_edge(source, target, key=0, **attr)

    return G

if __name__ == "__main__":
    city = generate_organic_city()
    nx.write_graphml(city, "city_graph.graphml")
    print(f"✅ Complex City Generated: {len(city.nodes())} nodes, {len(city.edges())} edges.")
    print("The map is now organic, messy, and realistic.")
    