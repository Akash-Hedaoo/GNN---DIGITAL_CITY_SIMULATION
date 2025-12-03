import networkx as nx
import pandas as pd
import numpy as np
import osmnx as ox
from shapely import wkt
import random

# Input/Output Files
INPUT_GRAPH = 'real_city_raw.graphml'
INPUT_AMENITIES = 'real_city_amenities.csv'
OUTPUT_GRAPH = 'real_city_processed.graphml'

# Configuration
DEFAULT_SPEED_KMPH = 30
METRO_SPEED_KMPH = 80

def sanitize_graph_attributes(G):
    """
    Fixes the 'list' error by converting all list attributes to strings.
    """
    print("   🧹 Sanitizing graph attributes for serialization...")
    for node, data in G.nodes(data=True):
        for key, val in data.items():
            if isinstance(val, list): data[key] = str(val)
    
    for u, v, k, data in G.edges(keys=True, data=True):
        for key, val in data.items():
            if isinstance(val, list): data[key] = str(val)
            if key == 'geometry' and not isinstance(val, str):
                try: data[key] = val.wkt
                except: data[key] = str(val)
    return G

def process_graph():
    print(f"🔄 Loading raw graph: {INPUT_GRAPH}...")
    G = ox.load_graphml(INPUT_GRAPH)
    
    # 1. Initialize Default Node Features
    print("   Initializing baseline population...")
    for node, data in G.nodes(data=True):
        # Random baseline for residential areas
        data['population'] = random.randint(50, 200)
        data['is_metro_station'] = False
        data['amenity_type'] = 'residential'
        if 'x' not in data: data['x'] = 0.0
        if 'y' not in data: data['y'] = 0.0

    # 2. Process Amenities (Metro + Regular)
    print("   🏗️ Processing Amenities & Population Density...")
    metro_station_nodes = []
    created_station_names = set()

    try:
        df = pd.read_csv(INPUT_AMENITIES)
        
        def get_pt(geom_str):
            try: return wkt.loads(geom_str)
            except: return None

        for idx, row in df.iterrows():
            pt = get_pt(str(row['geometry']))
            if not pt: continue
            
            raw_name = str(row['name'])
            amenity_type = str(row['amenity'])
            
            # --- CHECK FOR METRO STATIONS ---
            clean_name = None
            if 'PCMC' in raw_name or 'Corporation' in raw_name: clean_name = "PCMC Station"
            elif 'Tukaram' in raw_name: clean_name = "Sant Tukaram Nagar"
            elif 'Nashik' in raw_name or 'Bhosari' in raw_name: clean_name = "Nashik Phata (Bhosari)"
            elif 'Kasarwadi' in raw_name: clean_name = "Kasarwadi Station"
            elif 'Phugewadi' in raw_name: clean_name = "Phugewadi Station"
            
            if clean_name and clean_name not in created_station_names:
                # -> CREATE NEW METRO NODE
                new_node_id = -1 * (len(created_station_names) + 100)
                G.add_node(new_node_id, x=pt.x, y=pt.y, population=5000, 
                           is_metro_station=True, amenity_type='metro_station', name=clean_name)
                
                print(f"   🚇 Created Node {new_node_id}: {clean_name}")
                metro_station_nodes.append(new_node_id)
                created_station_names.add(clean_name)
                
                # Connect to nearest road
                nearest_road_node = ox.distance.nearest_nodes(G, pt.x, pt.y)
                dist = ox.distance.great_circle(pt.y, pt.x, G.nodes[nearest_road_node]['y'], G.nodes[nearest_road_node]['x'])
                edge_attrs = {'length': dist, 'maxspeed': '5', 'is_metro': False, 'base_travel_time': (dist/1000)/5 * 3600}
                G.add_edge(new_node_id, nearest_road_node, **edge_attrs)
                G.add_edge(nearest_road_node, new_node_id, **edge_attrs)
            
            else:
                # -> REGULAR AMENITY (Hospital, School, Mall)
                # Boost population of the nearest existing road node
                nearest_node = ox.distance.nearest_nodes(G, pt.x, pt.y)
                
                # Boost factor depends on type
                boost = 500 # Default (Restaurants, Banks)
                if 'hospital' in amenity_type: boost = 2000
                elif 'school' in amenity_type: boost = 1500
                elif 'mall' in amenity_type or 'market' in amenity_type: boost = 2500
                
                G.nodes[nearest_node]['population'] += boost
                G.nodes[nearest_node]['amenity_type'] = amenity_type
                # (Optional) Print only major ones to keep log clean
                if boost > 1000:
                    print(f"   🏢 Densified Node {nearest_node} for {raw_name} (+{boost} pop)")

    except Exception as e:
        print(f"   ⚠️ Error processing amenities: {e}")

    # 3. Create Metro Track
    print(f"   Constructing Metro Track between {len(metro_station_nodes)} stations...")
    if len(metro_station_nodes) >= 2:
        # Sort North-South
        metro_station_nodes.sort(key=lambda n: G.nodes[n]['y'], reverse=True)
        
        for i in range(len(metro_station_nodes) - 1):
            u, v = metro_station_nodes[i], metro_station_nodes[i+1]
            dist = ox.distance.great_circle(G.nodes[u]['y'], G.nodes[u]['x'],
                                          G.nodes[v]['y'], G.nodes[v]['x'])
            
            edge_attrs = {
                'length': dist,
                'maxspeed': str(METRO_SPEED_KMPH),
                'is_metro': True,
                'transport_mode': 'metro',
                'base_travel_time': (dist / 1000) / METRO_SPEED_KMPH * 3600,
                'current_travel_time': (dist / 1000) / METRO_SPEED_KMPH * 3600,
                'congestion_level': 1.0,
                'is_closed': False,
                'geometry': None
            }
            G.add_edge(u, v, **edge_attrs)
            G.add_edge(v, u, **edge_attrs)
            print(f"   ➕ Added Metro Track: {G.nodes[u]['name']} <==> {G.nodes[v]['name']}")

    # 4. Final Physics Pass
    print("   Calculating physics for all edges...")
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'base_travel_time' not in data:
            try:
                s_val = data.get('maxspeed', DEFAULT_SPEED_KMPH)
                if isinstance(s_val, list): s_val = s_val[0]
                speed = float(s_val)
            except: speed = float(DEFAULT_SPEED_KMPH)
            
            l = float(data.get('length', 50)) / 1000.0
            data['base_travel_time'] = (l / speed) * 3600
            data['current_travel_time'] = data['base_travel_time']
            data['is_closed'] = False
            data['congestion_level'] = 1.0
            data['is_metro'] = False

    G = sanitize_graph_attributes(G)
    nx.write_graphml(G, OUTPUT_GRAPH)
    print(f"\n✅ Success! Processed graph saved to: {OUTPUT_GRAPH}")
    print(f"   Final Stats: {len(G.nodes)} nodes, {len(G.edges)} edges")

if __name__ == "__main__":
    process_graph()