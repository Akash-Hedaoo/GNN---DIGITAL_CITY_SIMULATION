import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

print("🗺️  Loading Complex City Map...")

try:
    G = nx.read_graphml("city_graph.graphml")
except Exception as e:
    print(f"❌ Error loading graph: {e}")
    exit()

# 1. Extract Positions
pos = {}
for node, data in G.nodes(data=True):
    if 'x' in data and 'y' in data:
        pos[node] = (float(data['x']), float(data['y']))
    else:
        pos = nx.spring_layout(G)
        break

# 2. Setup Interactive Plot
plt.ion()
fig = plt.figure(figsize=(18, 18), facecolor='#1a1a1a')
ax = plt.gca()
ax.set_facecolor('#1a1a1a')
ax.set_aspect('equal')
fig.canvas.toolbar.pan()

# 3. Draw Edges (Roads & Metro)
# Separate metro edges by line
metro_line1_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('line_number') == 1]
metro_line2_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('line_number') == 2]
road_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('highway') != 'railway']

# Draw Roads
nx.draw_networkx_edges(G, pos, edgelist=road_edges, edge_color='#333333', width=0.6, arrows=False, alpha=0.5)

# Draw Metro Line 1 (Cyan - Horizontal)
if metro_line1_edges:
    nx.draw_networkx_edges(G, pos, edgelist=metro_line1_edges, edge_color='white', width=5.0, alpha=0.4, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=metro_line1_edges, edge_color='#00ffff', width=3.0, alpha=1.0, arrows=False)

# Draw Metro Line 2 (Magenta - Vertical)
if metro_line2_edges:
    nx.draw_networkx_edges(G, pos, edgelist=metro_line2_edges, edge_color='white', width=5.0, alpha=0.4, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=metro_line2_edges, edge_color='#ff00ff', width=3.0, alpha=1.0, arrows=False)

# 4. Draw Nodes (Zones & Amenities)
# Base Nodes (Dim)
node_colors = []
for n, d in G.nodes(data=True):
    zone = d.get('zone', 'suburbs')
    if zone == 'industrial': node_colors.append('#ff3333')
    elif zone == 'residential': node_colors.append('#33ff33')
    elif zone == 'downtown': node_colors.append('#3333ff')
    else: node_colors.append('#444444')

nx.draw_networkx_nodes(G, pos, node_size=25, node_color=node_colors, alpha=0.7, node_shape='o', linewidths=0)

# Highlights with smaller sizes
def draw_highlight(amenity_type, color, size):
    nodelist = [n for n, d in G.nodes(data=True) if d.get('amenity') == amenity_type]
    if nodelist:
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=size, node_color=color, node_shape='o', edgecolors='none', alpha=0.95)

# Draw all amenities as circles with different colors and smaller sizes
draw_highlight('hospital', '#ff1744', 80)
draw_highlight('metro_station', '#00ffff', 100)
draw_highlight('school', '#2196f3', 60)
draw_highlight('mall', '#e91e63', 70)
draw_highlight('office', '#9e9e9e', 50)
draw_highlight('factory', '#ff5722', 60)
draw_highlight('warehouse', '#795548', 55)
draw_highlight('community_center', '#00bcd4', 60)
draw_highlight('government', '#9c27b0', 65)

# Green zones
green_nodes = [n for n, d in G.nodes(data=True) if d.get('green_zone')]
if green_nodes:
    nx.draw_networkx_nodes(G, pos, nodelist=green_nodes, node_size=70, node_color='#00e676', node_shape='o', edgecolors='none', alpha=0.85)

# Green zones
green_nodes = [n for n, d in G.nodes(data=True) if d.get('green_zone')]
if green_nodes:
    nx.draw_networkx_nodes(G, pos, nodelist=green_nodes, node_size=70, node_color='#00e676', node_shape='o', edgecolors='none', alpha=0.85)

# 5. Add Legend and Title
plt.title("URBAN SYMBIOSIS - Digital City Twin", color='white', fontsize=22, fontweight='bold', pad=25)

# Create compact legend
legend_items = [
    ("Metro Line 1 (Horizontal)", '#00ffff', 'line'),
    ("Metro Line 2 (Vertical)", '#ff00ff', 'line'),
    ("Metro Stations", '#00ffff', 'circle'),
    ("Hospitals", '#ff1744', 'circle'),
    ("Green Zones", '#00e676', 'circle'),
    ("Schools", '#2196f3', 'circle'),
    ("Offices", '#9e9e9e', 'circle'),
    ("Malls", '#e91e63', 'circle'),
    ("Factories", '#ff5722', 'circle'),
    ("Warehouses", '#795548', 'circle'),
    ("Community Centers", '#00bcd4', 'circle'),
    ("Government", '#9c27b0', 'circle'),
]

legend_elements = []
for label, color, shape_type in legend_items:
    if shape_type == 'line':
        legend_elements.append(Line2D([0], [0], color=color, linewidth=3, label=label))
    else:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label, 
                                    markerfacecolor=color, markersize=9, linewidth=0))

legend = plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), 
                   fontsize=10, framealpha=0.95, facecolor='#2a2a2a', edgecolor='white', 
                   labelcolor='white', title='LEGEND', title_fontsize=12, borderpad=1.2)
legend.get_title().set_color('white')
legend.get_title().set_weight('bold')

# Zone reference
zone_text = "ZONES:  ● Industrial  ● Downtown  ● Residential  ● Suburbs"
plt.text(0.5, 0.015, zone_text, transform=fig.transFigure, 
         ha='center', color='white', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#2a2a2a', alpha=0.95, edgecolor='white', linewidth=1.5))

# Interactive controls
instructions = "🖱️  CONTROLS: Drag to Pan  |  Scroll to Zoom  |  Home to Reset"
plt.text(0.5, 0.985, instructions, transform=fig.transFigure, 
         ha='center', va='top', color='#00ffff', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', alpha=0.95, edgecolor='#00ffff', linewidth=2))

plt.axis('off')
plt.subplots_adjust(right=0.85)  # Make room for legend on the right
plt.tight_layout()

print("✨ Displaying interactive map...")
print("   📌 Drag to pan, scroll to zoom")
print("   📌 All icons are now smaller circles for clarity")
plt.ioff()
plt.show()