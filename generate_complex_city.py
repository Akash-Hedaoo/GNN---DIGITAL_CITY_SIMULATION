import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
import random
import math

# --- CONFIGURATION ---
NUM_NODES = 800  # High density for complexity
CITY_CENTER_LAT = 18.5204
CITY_CENTER_LON = 73.8567
SCALE = 0.035  # Wider area coverage for better spread
NUM_HOSPITALS = 15  # How many civic amenities to inject
NUM_GREEN_ZONES = 30  # Distributed urban green spaces
NUM_METRO_LINES = 3  # Number of metro lines
METRO_STATIONS_PER_LINE = 8  # Stations per line
METRO_SPEED_KMH = 80  # Metro average speed (faster than cars!)
METRO_CAPACITY_MULTIPLIER = 5.0  # Metro can handle 5x more passengers than road

GREEN_ZONE_ZONE_SHARE = {
    "downtown": 0.25,
    "residential": 0.4,
    "suburbs": 0.35
}
GREEN_ZONE_IDEAL_RADIAL = {
    "downtown": 0.25,
    "residential": 0.55,
    "suburbs": 0.85
}
GREEN_ZONE_ANGLE_BINS = 12
ZONE_PRIORITY = {
    "downtown": 0,
    "residential": 1,
    "suburbs": 2,
    "industrial": 3
}


def designate_green_zones(G: nx.MultiDiGraph):
    """Mark nodes as green zones spread across multiple parts of the city."""
    if G.number_of_nodes() == 0 or NUM_GREEN_ZONES <= 0:
        return

    zone_pools = {zone: [] for zone in GREEN_ZONE_ZONE_SHARE}
    flexible_pool = []

    for node, data in G.nodes(data=True):
        entry = (node, data)
        zone = data.get("zone")
        if zone in zone_pools:
            zone_pools[zone].append(entry)
        else:
            flexible_pool.append(entry)

    def compute_radial(entry):
        _, data = entry
        radial = data.get("radial_distance")
        if radial is None:
            dist_sq = (data["x"] - CITY_CENTER_LON) ** 2 + (data["y"] - CITY_CENTER_LAT) ** 2
            radial = dist_sq ** 0.5
        return radial

    def sort_by_radial(entry, zone):
        radial = compute_radial(entry)
        ideal = GREEN_ZONE_IDEAL_RADIAL.get(zone, 0.6)
        return abs(radial - ideal)

    def angle_bin(entry):
        _, data = entry
        angle = data.get("polar_angle")
        if angle is None:
            y = data.get("y", CITY_CENTER_LAT)
            x = data.get("x", CITY_CENTER_LON)
            angle = math.atan2(y - CITY_CENTER_LAT, x - CITY_CENTER_LON)
        normalized = (angle + math.pi) % (2 * math.pi)
        bin_idx = int((normalized / (2 * math.pi)) * GREEN_ZONE_ANGLE_BINS)
        return min(bin_idx, GREEN_ZONE_ANGLE_BINS - 1)

    def select_with_angle_diversity(pool, zone, quota):
        if not pool or quota <= 0:
            return []

        bins = {}
        for entry in pool:
            idx = angle_bin(entry)
            bins.setdefault(idx, []).append(entry)

        for idx, entries in bins.items():
            entries.sort(key=lambda entry, z=zone: sort_by_radial(entry, z))

        picked = []
        bin_keys = list(bins.keys())
        key_index = 0

        while bin_keys and len(picked) < quota:
            idx = bin_keys[key_index % len(bin_keys)]
            if bins[idx]:
                picked.append(bins[idx].pop(0))
            if not bins[idx]:
                bins.pop(idx)
                bin_keys = list(bins.keys())
                key_index = 0
                continue
            key_index += 1

        return picked

    selected = []
    remaining = NUM_GREEN_ZONES

    for zone, share in GREEN_ZONE_ZONE_SHARE.items():
        pool = zone_pools.get(zone, [])
        if not pool or remaining <= 0:
            continue
        quota = max(1, round(NUM_GREEN_ZONES * share))
        quota = min(quota, remaining, len(pool))
        chosen = select_with_angle_diversity(pool, zone, quota)
        selected.extend(chosen)
        zone_pools[zone] = [entry for entry in pool if entry not in chosen]
        remaining -= quota

    if remaining > 0:
        leftovers = []
        for entries in zone_pools.values():
            leftovers.extend(entries)
        leftovers.extend(flexible_pool)
        leftovers = [item for item in leftovers if item not in selected]
        chosen = select_with_angle_diversity(leftovers, "fallback", remaining)
        selected.extend(chosen)

    for idx, (node, _) in enumerate(selected[:NUM_GREEN_ZONES], start=1):
        G.nodes[node]["green_zone"] = True
        G.nodes[node]["park_name"] = f"Eco Park {idx}"
        G.nodes[node]["park_type"] = random.choice(["neighborhood", "urban_forest", "botanical", "recreation"])
        G.nodes[node]["green_area_hectares"] = round(random.uniform(0.5, 6.0), 2)
        if "amenity" not in G.nodes[node]:
            G.nodes[node]["amenity"] = "park"
        else:
            G.nodes[node]["secondary_amenity"] = "park"

    print(f"🌳  Green zones added: {min(len(selected), NUM_GREEN_ZONES)} across city.")


def designate_hospitals(G: nx.MultiDiGraph):
    """Select strategic nodes and tag them as hospitals with good spatial coverage."""
    if G.number_of_nodes() == 0 or NUM_HOSPITALS <= 0:
        return

    candidates = [
        (node, data)
        for node, data in G.nodes(data=True)
        if data.get("zone") in ("downtown", "residential", "suburbs")
    ]

    if not candidates:
        return

    bucket_defs = [
        ("core", lambda d: d < 0.35, 0.4),
        ("mid", lambda d: 0.35 <= d < 0.75, 0.35),
        ("outer", lambda d: d >= 0.75, 0.25)
    ]

    bucketed = {name: [] for name, _, _ in bucket_defs}
    fallback_bucket = []

    def sort_key(item):
        _, data = item
        zone = data.get("zone", "suburbs")
        priority = ZONE_PRIORITY.get(zone, 99)
        radial = data.get("radial_distance")
        if radial is None:
            dist_sq = (data["x"] - CITY_CENTER_LON) ** 2 + (data["y"] - CITY_CENTER_LAT) ** 2
            radial = dist_sq ** 0.5
        return priority, radial

    candidates.sort(key=sort_key)

    for item in candidates:
        _, data = item
        radial = data.get("radial_distance")
        if radial is None:
            dist_sq = (data["x"] - CITY_CENTER_LON) ** 2 + (data["y"] - CITY_CENTER_LAT) ** 2
            radial = dist_sq ** 0.5

        placed = False
        for name, predicate, _ in bucket_defs:
            if predicate(radial):
                bucketed[name].append(item)
                placed = True
                break
        if not placed:
            fallback_bucket.append(item)

    selected = []
    remaining = NUM_HOSPITALS

    for name, _, share in bucket_defs:
        if remaining <= 0:
            break
        pool = bucketed[name]
        if not pool:
            continue
        random.shuffle(pool)
        quota = max(1, round(NUM_HOSPITALS * share))
        quota = min(quota, remaining, len(pool))
        selected.extend(pool[:quota])
        remaining -= quota

    if remaining > 0:
        leftovers = [
            item for item in candidates
            if item not in selected
        ]
        leftovers.sort(key=lambda item: item[1].get("radial_distance", 0), reverse=True)
        selected.extend(leftovers[:remaining])

    for idx, (node, _) in enumerate(selected[:NUM_HOSPITALS], start=1):
        G.nodes[node]["amenity"] = "hospital"
        G.nodes[node]["facility_name"] = f"City Hospital {idx}"
        G.nodes[node]["hospital_capacity"] = random.randint(120, 500)
        G.nodes[node]["emergency_level"] = random.choice(["general", "trauma", "specialty"])

    print(f"🏥  Hospitals added: {len(selected)} (amenity='hospital').")


def designate_public_places(G: nx.MultiDiGraph):
    """Assign public place amenities based on zone probabilities."""
    if G.number_of_nodes() == 0:
        return
    
    # Probability distributions by zone
    zone_amenity_probs = {
        "residential": [
            ("school", 0.15),
            ("community_center", 0.05)
        ],
        "downtown": [
            ("office", 0.30),
            ("mall", 0.10),
            ("government", 0.05)
        ],
        "industrial": [
            ("factory", 0.20),
            ("warehouse", 0.10)
        ]
    }
    
    amenities_added = 0
    
    for node, data in G.nodes(data=True):
        # Skip if node already has an amenity assigned
        if "amenity" in data:
            continue
        
        zone = data.get("zone")
        if zone not in zone_amenity_probs:
            continue
        
        # Get probabilities for this zone
        amenity_options = zone_amenity_probs[zone]
        
        # Check if we should assign an amenity
        for amenity, probability in amenity_options:
            if random.random() < probability:
                G.nodes[node]["amenity"] = amenity
                
                # Add specific attributes based on amenity type
                if amenity == "school":
                    G.nodes[node]["facility_name"] = f"School {random.randint(1, 100)}"
                    G.nodes[node]["capacity"] = random.randint(300, 1500)
                elif amenity == "community_center":
                    G.nodes[node]["facility_name"] = f"Community Center {random.randint(1, 50)}"
                    G.nodes[node]["services"] = random.choice(["sports", "cultural", "multipurpose"])
                elif amenity == "office":
                    G.nodes[node]["building_name"] = f"Office Tower {random.randint(1, 200)}"
                    G.nodes[node]["floors"] = random.randint(5, 40)
                elif amenity == "mall":
                    G.nodes[node]["facility_name"] = f"Shopping Mall {random.randint(1, 30)}"
                    G.nodes[node]["retail_area_sqm"] = random.randint(5000, 50000)
                elif amenity == "government":
                    G.nodes[node]["facility_name"] = f"Government Office {random.randint(1, 20)}"
                    G.nodes[node]["department"] = random.choice(["municipal", "administrative", "civic"])
                elif amenity == "factory":
                    G.nodes[node]["facility_name"] = f"Factory {random.randint(1, 100)}"
                    G.nodes[node]["industry_type"] = random.choice(["manufacturing", "processing", "assembly"])
                elif amenity == "warehouse":
                    G.nodes[node]["facility_name"] = f"Warehouse {random.randint(1, 150)}"
                    G.nodes[node]["storage_capacity_sqm"] = random.randint(1000, 20000)
                
                amenities_added += 1
                break  # Only assign one amenity per node
    
    print(f"🏢  Public places added: {amenities_added} across all zones.")


def build_metro_network(G: nx.MultiDiGraph):
    """
    Create an advanced metro network with multiple lines.
    Metro acts as alternative transport reducing road congestion.
    
    Features:
    - Multiple lines with unique routes
    - Faster travel times than roads
    - Higher capacity (less congestion)
    - Strategic station placement
    """
    if G.number_of_nodes() < 12:
        print("⚠️  Not enough nodes to build metro network.")
        return
    
    all_nodes = list(G.nodes(data=True))
    all_metro_stations = []
    metro_edges_added = 0
    
    # Calculate city bounds
    avg_x = sum(data['x'] for _, data in all_nodes) / len(all_nodes)
    avg_y = sum(data['y'] for _, data in all_nodes) / len(all_nodes)
    
    x_values = [data['x'] for _, data in all_nodes]
    y_values = [data['y'] for _, data in all_nodes]
    x_range = max(x_values) - min(x_values)
    y_range = max(y_values) - min(y_values)
    
    # Metro line configurations
    metro_configs = [
        {
            'name': 'Red Line',
            'direction': 'horizontal',
            'offset': 0,  # Center
            'color': '#FF0000',
            'description': 'East-West Corridor'
        },
        {
            'name': 'Blue Line',
            'direction': 'vertical',
            'offset': 0,  # Center
            'color': '#0000FF',
            'description': 'North-South Corridor'
        },
        {
            'name': 'Green Line',
            'direction': 'diagonal',
            'angle': 45,  # Diagonal NE-SW
            'color': '#00FF00',
            'description': 'Diagonal Connector'
        }
    ]
    
    for line_idx, config in enumerate(metro_configs[:NUM_METRO_LINES], 1):
        line_name = config['name']
        direction = config['direction']
        
        # Select nodes for this metro line
        if direction == 'horizontal':
            # Horizontal line (West to East)
            target_y = avg_y + config['offset'] * y_range
            tolerance = y_range * 0.2
            
            candidate_nodes = [
                (node, data) for node, data in all_nodes
                if abs(data['y'] - target_y) <= tolerance 
                and node not in all_metro_stations
            ]
            candidate_nodes.sort(key=lambda item: item[1]['x'])
            
        elif direction == 'vertical':
            # Vertical line (South to North)
            target_x = avg_x + config['offset'] * x_range
            tolerance = x_range * 0.2
            
            candidate_nodes = [
                (node, data) for node, data in all_nodes
                if abs(data['x'] - target_x) <= tolerance
                and node not in all_metro_stations
            ]
            candidate_nodes.sort(key=lambda item: item[1]['y'])
            
        elif direction == 'diagonal':
            # Diagonal line (along a specific angle)
            angle_rad = math.radians(config['angle'])
            
            # Calculate perpendicular distance from diagonal line
            candidate_nodes = []
            for node, data in all_nodes:
                if node in all_metro_stations:
                    continue
                
                # Point relative to center
                px = data['x'] - avg_x
                py = data['y'] - avg_y
                
                # Distance from diagonal line through origin
                dist_from_line = abs(-math.sin(angle_rad) * px + math.cos(angle_rad) * py)
                
                # Position along the line (for sorting)
                along_line = math.cos(angle_rad) * px + math.sin(angle_rad) * py
                
                if dist_from_line <= max(x_range, y_range) * 0.15:
                    candidate_nodes.append((node, data, along_line))
            
            candidate_nodes.sort(key=lambda item: item[2])
            candidate_nodes = [(n, d) for n, d, _ in candidate_nodes]
        
        else:
            continue
        
        # Select evenly spaced stations
        num_stations = min(METRO_STATIONS_PER_LINE, len(candidate_nodes))
        if num_stations < 2:
            print(f"⚠️  Not enough nodes for {line_name}")
            continue
        
        line_stations = []
        for i in range(num_stations):
            if num_stations == 1:
                idx = 0
            else:
                idx = int((i / (num_stations - 1)) * (len(candidate_nodes) - 1))
            
            node_id, node_data = candidate_nodes[idx]
            line_stations.append(node_id)
            all_metro_stations.append(node_id)
            
            # Mark node as metro station
            existing_amenity = G.nodes[node_id].get("amenity", "")
            if existing_amenity and "metro" not in existing_amenity:
                G.nodes[node_id]["amenity"] = f"{existing_amenity}+metro_station"
            else:
                G.nodes[node_id]["amenity"] = "metro_station"
            
            G.nodes[node_id]["metro_station"] = True
            G.nodes[node_id]["station_name"] = f"{line_name} S{i + 1}"
            # Store metro lines as comma-separated string instead of list
            existing_lines = G.nodes[node_id].get("metro_lines_str", "")
            if existing_lines:
                G.nodes[node_id]["metro_lines_str"] = existing_lines + "," + line_name
            else:
                G.nodes[node_id]["metro_lines_str"] = line_name
            G.nodes[node_id]["station_color"] = config['color']
        
        # Connect stations with metro edges
        for i in range(len(line_stations) - 1):
            source = line_stations[i]
            target = line_stations[i + 1]
            
            x1, y1 = G.nodes[source]['x'], G.nodes[source]['y']
            x2, y2 = G.nodes[target]['x'], G.nodes[target]['y']
            dist_deg = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            length_meters = dist_deg * 111000  # Convert to meters
            
            # Metro is MUCH faster than cars
            speed_mps = METRO_SPEED_KMH / 3.6
            travel_time = length_meters / speed_mps
            
            # Add bidirectional metro edges with unique keys
            for src, tgt in [(source, target), (target, source)]:
                edge_key = f"metro_{line_name.lower().replace(' ', '_')}_{src}_{tgt}"
                
                G.add_edge(src, tgt, key=edge_key, **{
                    'osmid': edge_key,
                    'highway': 'metro_railway',
                    'maxspeed': METRO_SPEED_KMH,
                    'speed_limit': METRO_SPEED_KMH,
                    'name': line_name,
                    'length': length_meters,
                    'is_closed': 0,
                    'oneway': False,
                    'lanes': 2,
                    'base_travel_time': travel_time,
                    'current_travel_time': travel_time,
                    'transport_mode': 'metro',
                    'line_name': line_name,
                    'line_number': line_idx,
                    'line_color': config['color'],
                    'capacity_multiplier': METRO_CAPACITY_MULTIPLIER,
                    'is_metro': True,
                    'congestion_resistant': True  # Metro doesn't get stuck in traffic!
                })
                metro_edges_added += 1
        
        print(f"🚇  {line_name} ({config['description']}): {num_stations} stations, {len(line_stations)-1} segments")
    
    # Add interchange stations (stations served by multiple lines)
    interchange_count = 0
    for node in all_metro_stations:
        lines_str = G.nodes[node].get("metro_lines_str", "")
        lines = lines_str.split(",") if lines_str else []
        if len(lines) > 1:
            G.nodes[node]["interchange"] = True
            G.nodes[node]["station_name"] += " (Interchange)"
            interchange_count += 1
    
    total_stations = len(all_metro_stations)
    unique_stations = len(set(all_metro_stations))
    
    print(f"\n🎯 Metro Network Summary:")
    print(f"   Total Stations: {unique_stations}")
    print(f"   Interchange Stations: {interchange_count}")
    print(f"   Metro Edges: {metro_edges_added}")
    print(f"   Average Speed: {METRO_SPEED_KMH} km/h (vs ~40 km/h road)")
    print(f"   Capacity: {METRO_CAPACITY_MULTIPLIER}x road capacity")
    print(f"   🎉 Metro can ease traffic by providing fast alternative routes!")


def generate_organic_city():
    print(f"🏗️  Generating Organic City with {NUM_NODES} nodes...")

    # 1. Generate Random Points (Organic Intersections)
    # Use Poisson disk sampling approach for better distribution
    # This prevents clustering and ensures minimum spacing between nodes
    points = []
    min_distance = 0.12  # Minimum distance between nodes
    max_attempts = 30  # Attempts to place each point
    
    # Start with a few seed points
    for _ in range(5):
        x = np.random.uniform(-1.8, 1.8)
        y = np.random.uniform(-1.8, 1.8)
        points.append([x, y])
    
    # Generate remaining points with spacing constraint
    attempts = 0
    max_total_attempts = NUM_NODES * 50
    
    while len(points) < NUM_NODES and attempts < max_total_attempts:
        attempts += 1
        
        # Mix of uniform (spread out) and normal (slight center bias)
        if random.random() > 0.4:
            x = np.random.normal(0, 0.8)  # Slight center clustering
            y = np.random.normal(0, 0.8)
        else:
            x = np.random.uniform(-1.8, 1.8)  # More spread out
            y = np.random.uniform(-1.8, 1.8)
        
        # Check minimum distance to existing points
        new_point = np.array([x, y])
        if len(points) > 0:
            distances = np.linalg.norm(np.array(points) - new_point, axis=1)
            if np.min(distances) < min_distance:
                continue  # Too close, try again
        
        points.append([x, y])
    
    points = np.array(points)
    print(f"   ✅ Generated {len(points)} well-spaced nodes")

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
        
        # Remove very long edges (adjusted for wider spread)
        if dist > 0.6 or (dist < 0.08 and random.random() > 0.7):
            edges_to_remove.append((u, v))
            
    G_temp.remove_edges_from(edges_to_remove)
    print(f"   🔧 Pruned {len(edges_to_remove)} edges for realistic street network")
    
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
        polar_angle = math.atan2(y_rel, x_rel)

        G.add_node(
            new_id,
            x=real_x,
            y=real_y,
            zone=zone,
            color=color,
            radial_distance=dist_from_center,
            polar_angle=polar_angle
        )

    designate_green_zones(G)
    designate_hospitals(G)
    designate_public_places(G)
    build_metro_network(G)

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
    