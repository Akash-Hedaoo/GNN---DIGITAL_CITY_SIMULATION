import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from shapely import wkt
from step5_whatif_analysis import CityDigitalTwin

def repair_graph_geometries(G):
    """
    Converts string-based WKT geometries back to Shapely objects
    so OSMnx can plot them without crashing.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'geometry' in data and isinstance(data['geometry'], str):
            try:
                data['geometry'] = wkt.loads(data['geometry'])
            except:
                pass # If it fails, OSMnx will just draw a straight line (fallback)
    return G

def visualize_impact():
    print("🎨 INITIALIZING VISUALIZATION ENGINE...")
    twin = CityDigitalTwin()
    
    # 1. Pick a Target Node
    nodes = list(twin.G.nodes())
    candidates = [n for n in nodes if len(list(twin.G.neighbors(n))) > 0]
    target_node = candidates[len(candidates)//2]
    
    print(f"   🏗️  Simulating Metro Station at Node {target_node}...")
    
    # 2. Re-create the Scenario Graph
    G_mod = twin.G.copy()
    
    # CRITICAL FIX: Repair geometries before doing anything else
    G_mod = repair_graph_geometries(G_mod)
    
    G_mod.nodes[target_node]['amenity_type'] = 'metro'
    
    # Apply Pop Boost
    cfg = {'radius': 2500, 'pop_boost': 10.0}
    target_x = G_mod.nodes[target_node]['x']
    target_y = G_mod.nodes[target_node]['y']
    
    for n in G_mod.nodes():
        dist = np.sqrt((G_mod.nodes[n]['x']-target_x)**2 + (G_mod.nodes[n]['y']-target_y)**2)
        if dist <= cfg['radius']:
            G_mod.nodes[n]['population'] *= cfg['pop_boost']
            
    # Apply Stress Test
    neighbor = list(twin.G.neighbors(target_node))[0]
    if G_mod.has_edge(target_node, neighbor):
        for k in G_mod[target_node][neighbor]: 
            G_mod[target_node][neighbor][k]['is_closed'] = 'True'
            
    # 3. Get Predictions
    print("   🧠 Calculating Congestion Heatmap...")
    predictions = twin._predict_traffic(G_mod)
    
    # 4. Color the Graph (Manual Calculation)
    print("   🎨 Painting the City Map...")
    
    edge_congestion_values = []
    # Iterate keys to ensure order matches
    for u, v, k in G_mod.edges(keys=True):
        val = predictions.get((u, v, k), 1.0)
        edge_congestion_values.append(val)
        
    # Create Color Map (Green -> Yellow -> Red)
    # 1.0 = Green (No traffic), 5.0+ = Red (Jam)
    norm = mcolors.Normalize(vmin=1.0, vmax=5.0)
    # Use 'matplotlib.colormaps' instead of deprecated 'plt.cm.get_cmap'
    cmap = plt.get_cmap('inferno_r') 
    
    edge_colors = [cmap(norm(val)) for val in edge_congestion_values]
        
    # Create Plot
    fig, ax = ox.plot_graph(
        G_mod,
        node_size=0,
        edge_color=edge_colors, 
        edge_linewidth=1.5,
        bgcolor='black',
        show=False,
        close=False
    )
    
    # Add Title
    plt.title(f"Traffic Heatmap: New Metro at Node {target_node}", color='white', fontsize=12)
    
    # Save
    output_file = 'final_traffic_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ IMAGE SAVED: {output_file}")
    print("   Open this file to see your Digital Twin in action!")

if __name__ == "__main__":
    visualize_impact()