import networkx as nx
import folium
from folium import plugins
import webbrowser
import os

print("🗺️  Creating Interactive Web Map (Google Maps style)...")

try:
    G = nx.read_graphml("city_graph.graphml")
    # Convert to MultiDiGraph if needed for proper edge handling
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)
except Exception as e:
    print(f"[FAIL] Error loading graph: {e}")
    exit()

# Calculate map center
lats = [float(data['y']) for node, data in G.nodes(data=True) if 'y' in data]
lons = [float(data['x']) for node, data in G.nodes(data=True) if 'x' in data]
center_lat = sum(lats) / len(lats)
center_lon = sum(lons) / len(lons)

# Create base map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles='CartoDB dark_matter',
    control_scale=True
)

# Color scheme
COLORS = {
    'industrial': '#ff3333',
    'residential': '#33ff33',
    'downtown': '#3333ff',
    'suburbs': '#888888',
    'metro_red': '#ff0000',
    'metro_blue': '#0000ff',
    'metro_green': '#00ff00',
    'metro': '#00ffff',
    'hospital': '#ff1744',
    'green_zone': '#00e676',
    'school': '#2196f3',
    'office': '#9e9e9e',
    'mall': '#e91e63',
    'factory': '#ff5722',
    'warehouse': '#795548',
    'community_center': '#00bcd4',
    'government': '#9c27b0',
}

# Create feature groups for layer control
roads_layer = folium.FeatureGroup(name='Roads', show=True)
metro_layer = folium.FeatureGroup(name='Metro Lines', show=True)
zones_layer = folium.FeatureGroup(name='Zone Nodes', show=False)
amenities_layer = folium.FeatureGroup(name='Amenities', show=True)
population_layer = folium.FeatureGroup(name='Population Density', show=True)

# Helper function to get population color
def get_population_color(population):
    """Return color based on population density"""
    if population > 700:
        return '#d62728'  # Dark red
    elif population > 500:
        return '#ff7f0e'  # Orange
    elif population > 300:
        return '#ffff00'  # Yellow
    elif population > 100:
        return '#2ca02c'  # Green
    else:
        return '#1f77b4'  # Blue

# Draw Roads
for u, v, data in G.edges(data=True):
    if data.get('highway') != 'railway':
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        
        if 'x' in u_data and 'y' in u_data and 'x' in v_data and 'y' in v_data:
            coords = [
                [float(u_data['y']), float(u_data['x'])],
                [float(v_data['y']), float(v_data['x'])]
            ]
            
            color = '#ffa500' if data.get('highway') == 'primary' else '#555555'
            weight = 2 if data.get('highway') == 'primary' else 1
            
            folium.PolyLine(
                coords,
                color=color,
                weight=weight,
                opacity=0.6,
                popup=f"{data.get('name', 'Road')}"
            ).add_to(roads_layer)

# Draw Metro Lines
# Check if MultiDiGraph or DiGraph
is_multigraph = isinstance(G, nx.MultiDiGraph) or isinstance(G, nx.MultiGraph)

if is_multigraph:
    edge_iter = G.edges(keys=True, data=True)
else:
    edge_iter = [(u, v, 0, data) for u, v, data in G.edges(data=True)]

for edge_tuple in edge_iter:
    if len(edge_tuple) == 4:
        u, v, key, data = edge_tuple
    else:
        u, v, data = edge_tuple
        key = 0
    
    # Check for metro edges using is_metro attribute or highway type
    if data.get('is_metro') or data.get('highway') in ['railway', 'metro_railway']:
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        
        if 'x' in u_data and 'y' in u_data and 'x' in v_data and 'y' in v_data:
            coords = [
                [float(u_data['y']), float(u_data['x'])],
                [float(v_data['y']), float(v_data['x'])]
            ]
            
            # Get line name and color from edge data
            line_name = data.get('line_name', data.get('name', 'Metro Line'))
            line_color = data.get('line_color', '#00ffff')
            line_num = data.get('line_number', 0)
            
            # Map line colors
            color_map = {0: COLORS['metro_red'], 1: COLORS['metro_blue'], 2: COLORS['metro_green']}
            color = color_map.get(line_num, line_color)
            
            folium.PolyLine(
                coords,
                color=color,
                weight=5,
                opacity=0.9,
                popup=f"🚇 {line_name}<br>Speed: {data.get('maxspeed', 80)} km/h<br>Capacity: {data.get('capacity_multiplier', 5.0)}x road"
            ).add_to(metro_layer)

# Draw Zone Nodes (base layer)
for node, data in G.nodes(data=True):
    if 'x' in data and 'y' in data:
        zone = data.get('zone', 'suburbs')
        color = COLORS.get(zone, '#888888')
        
        folium.CircleMarker(
            location=[float(data['y']), float(data['x'])],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            popup=f"Zone: {zone}"
        ).add_to(zones_layer)

# Draw Amenities (hospitals, schools, etc.)
amenity_config = {
    'metro_station': ('🚇', COLORS['metro'], 10, 'Metro Station'),
    'hospital': ('[HOSPITAL]', COLORS['hospital'], 8, 'Hospital'),
    'school': ('🏫', COLORS['school'], 6, 'School'),
    'mall': ('🛒', COLORS['mall'], 7, 'Shopping Mall'),
    'office': ('🏢', COLORS['office'], 5, 'Office'),
    'factory': ('🏭', COLORS['factory'], 6, 'Factory'),
    'warehouse': ('📦', COLORS['warehouse'], 5, 'Warehouse'),
    'community_center': ('🏛️', COLORS['community_center'], 6, 'Community Center'),
    'government': ('🏛️', COLORS['government'], 6, 'Government Office'),
}

for node, data in G.nodes(data=True):
    amenity = data.get('amenity')
    
    if amenity in amenity_config and 'x' in data and 'y' in data:
        icon_symbol, color, radius, label = amenity_config[amenity]
        
        # Create popup content
        popup_html = f"<b>{icon_symbol} {label}</b><br>"
        if 'station_name' in data:
            popup_html += f"Name: {data['station_name']}<br>"
        if 'facility_name' in data:
            popup_html += f"Name: {data['facility_name']}<br>"
        if 'line' in data:
            popup_html += f"Line: {data['line']}<br>"
        
        folium.CircleMarker(
            location=[float(data['y']), float(data['x'])],
            radius=radius,
            color='white',
            fillColor=color,
            fillOpacity=0.9,
            weight=2,
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(amenities_layer)

# Draw Green Zones
for node, data in G.nodes(data=True):
    if data.get('green_zone') and 'x' in data and 'y' in data:
        popup_html = f"<b>[TREE] {data.get('park_name', 'Park')}</b><br>"
        popup_html += f"Type: {data.get('park_type', 'Park')}<br>"
        popup_html += f"Area: {data.get('green_area_hectares', 'N/A')} hectares"
        
        folium.CircleMarker(
            location=[float(data['y']), float(data['x'])],
            radius=7,
            color='#004d40',
            fillColor=COLORS['green_zone'],
            fillOpacity=0.8,
            weight=1,
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(amenities_layer)

# Add all layers to map
roads_layer.add_to(m)
metro_layer.add_to(m)
zones_layer.add_to(m)
amenities_layer.add_to(m)
population_layer.add_to(m)

# Add layer control
folium.LayerControl(position='topright', collapsed=False).add_to(m)

# Add minimap
plugins.MiniMap(toggle_display=True, position='bottomright').add_to(m)

# Add fullscreen button
plugins.Fullscreen(position='topleft').add_to(m)

# Add measure control
plugins.MeasureControl(position='bottomleft', primary_length_unit='meters').add_to(m)

# Add legend
legend_html = '''
<div style="position: fixed; 
            top: 10px; left: 10px; width: 280px; 
            background-color: rgba(26, 26, 26, 0.95); 
            border: 2px solid white;
            border-radius: 8px;
            z-index: 9999; 
            padding: 15px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 12px;">
    <h4 style="margin-top: 0; color: #00ffff; text-align: center;">URBAN SYMBIOSIS</h4>
    <p style="margin: 5px 0;"><span style="color: #ff0000;">━━</span> Red Line (East-West)</p>
    <p style="margin: 5px 0;"><span style="color: #0000ff;">━━</span> Blue Line (North-South)</p>
    <p style="margin: 5px 0;"><span style="color: #00ff00;">━━</span> Green Line (Diagonal)</p>
    <p style="margin: 5px 0;"><span style="color: #00ffff;">●</span> Metro Stations</p>
    <p style="margin: 5px 0;"><span style="color: #ff1744;">●</span> Hospitals</p>
    <p style="margin: 5px 0;"><span style="color: #00e676;">●</span> Green Zones/Parks</p>
    <p style="margin: 5px 0;"><span style="color: #2196f3;">●</span> Schools</p>
    <p style="margin: 5px 0;"><span style="color: #e91e63;">●</span> Malls</p>
    <p style="margin: 5px 0;"><span style="color: #ff5722;">●</span> Factories</p>
    <p style="margin: 5px 0;"><span style="color: #9c27b0;">●</span> Government</p>
    <hr style="border-color: #555;">
    <p style="margin: 5px 0; font-size: 10px;"><b>Population Density:</b></p>
    <p style="margin: 5px 0;"><span style="color: #d62728;">●</span> Very High (>700)</p>
    <p style="margin: 5px 0;"><span style="color: #ff7f0e;">●</span> High (500-700)</p>
    <p style="margin: 5px 0;"><span style="color: #ffff00;">●</span> Medium (300-500)</p>
    <p style="margin: 5px 0;"><span style="color: #2ca02c;">●</span> Low (100-300)</p>
    <p style="margin: 5px 0;"><span style="color: #1f77b4;">●</span> Sparse (<100)</p>
    <hr style="border-color: #555;">
    <p style="margin: 5px 0; font-size: 10px;"><b>Zones:</b></p>
    <p style="margin: 5px 0;"><span style="color: #ff3333;">●</span> Industrial</p>
    <p style="margin: 5px 0;"><span style="color: #3333ff;">●</span> Downtown</p>
    <p style="margin: 5px 0;"><span style="color: #33ff33;">●</span> Residential</p>
    <p style="margin: 5px 0;"><span style="color: #888888;">●</span> Suburbs</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Add population density visualization
print("[CHART] Adding population density layer...")
for node, data in G.nodes(data=True):
    if 'x' in data and 'y' in data:
        population = int(data.get('population', 0))
        daily_trips = int(data.get('daily_trips', 0))
        zone = data.get('zone', 'unknown')
        amenity = data.get('amenity', 'none')
        
        # Get color based on population
        color = get_population_color(population)
        
        # Circle size based on population (scaled)
        radius = max(3, min(15, 3 + (population / 100)))
        
        # Create popup with detailed information
        popup_text = f"""
        <b>Node {node}</b><br>
        <b>Population:</b> {population:,} people<br>
        <b>Daily Trips:</b> {daily_trips:,}<br>
        <b>Zone:</b> {zone}<br>
        <b>Amenity:</b> {amenity if amenity != 'none' else 'Residential'}<br>
        <b>Coordinates:</b> ({float(data.get('y', 0)):.4f}, {float(data.get('x', 0)):.4f})
        """
        
        folium.CircleMarker(
            location=[float(data['y']), float(data['x'])],
            radius=radius,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=1
        ).add_to(population_layer)

# Save map
output_file = "city_map_interactive.html"
m.save(output_file)

print(f"[OK] Interactive map created: {output_file}")
print(f"🌐 Opening in browser...")

# Open in browser
webbrowser.open('file://' + os.path.realpath(output_file))

print("\n📌 Map Features:")
print("   ✓ Pan & Zoom like Google Maps")
print("   ✓ Click on markers for details")
print("   ✓ Toggle layers on/off (top-right)")
print("   ✓ Fullscreen mode available")
print("   ✓ Measure distances (bottom-left)")
print("   ✓ Mini-map (bottom-right)")
