import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Configuration: Nashik Phata is the central anchor
LOC_CENTER = "Nashik Phata, Pimpri-Chinchwad, Maharashtra, India"
RADIUS_METERS = 3500  # 3.5 km Radius covers PCMC (North) to Kasarwadi (South)

def acquire_full_metro_corridor():
    print(f"🌍 Locating Center: {LOC_CENTER}...")
    print(f"   Target Radius: {RADIUS_METERS} meters")
    print(f"   Goal: Include PCMC, Tukaram Nagar, Nashik Phata, Kasarwadi")
    
    try:
        # 1. Download Road Network
        print(f"   📥 Downloading graph (this may take 1-2 minutes)...")
        G = ox.graph_from_address(LOC_CENTER, dist=RADIUS_METERS, network_type='drive', simplify=True)
        
        # 2. Project to UTM
        G_proj = ox.project_graph(G)
        
        # 3. Stats
        nodes = len(G_proj.nodes)
        edges = len(G_proj.edges)
        
        print("\n" + "="*40)
        print(f"📊 REPORT FOR: FULL METRO CORRIDOR")
        print(f"   Nodes:  {nodes}")
        print(f"   Edges:  {edges}")
        print("="*40)
        
        # 4. Fetch Amenities & Metro Stations
        print("   🔍 Fetching Metro Stations and Amenities...")
        tags = {
            'amenity': ['hospital', 'school', 'bank', 'restaurant', 'pharmacy', 'cinema', 'fuel'],
            'railway': ['station', 'subway_entrance'],
            'public_transport': ['station']
        }
        
        try:
            amenities = ox.features_from_address(LOC_CENTER, tags, dist=RADIUS_METERS)
            print(f"   Found {len(amenities)} features.")
            
            # Verify specific stations are present
            targets = ['PCMC', 'Tukaram', 'Nashik', 'Bhosari', 'Kasarwadi']
            found_stations = []
            
            for t in targets:
                # Search for the target string in the 'name' column
                matches = amenities[amenities['name'].astype(str).str.contains(t, case=False, na=False)]
                if not matches.empty:
                    found_stations.append(t)
            
            print(f"   ✅ Confirmed Stations: {found_stations}")
            if len(found_stations) == 5:
                print("   🌟 SUCCESS: All requested metro stations found!")
                
            amenities.to_csv('real_city_amenities.csv')
        except Exception as e:
            print(f"   ⚠️ Warning: {e}")

        # 5. Save the Graph
        output_file = 'real_city_raw.graphml'
        ox.save_graphml(G_proj, output_file)
        print(f"💾 Saved raw graph to: {output_file}")
        
        # 6. Plot
        print("   Generating map preview...")
        ox.plot_graph(G_proj, node_size=2, node_color='r', show=False, close=True)
        plt.savefig('real_city_preview.png')
        print("🖼️  Saved preview to 'real_city_preview.png'")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    acquire_full_metro_corridor()