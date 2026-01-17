import os
import sys
import sumolib
import networkx as nx
import numpy as np
import math
import random
import subprocess
import logging

# Configuration
NET_FILE = os.path.join("data", "processed", "networks", "pune.net.xml")
TRIPS_FILE = os.path.join("data", "raw", "sumo", "trips.xml")
LOG_FILE = os.path.join("logs", "sumo", "phase2_simulation.log")
NUM_VEHICLES = 1000
DURATION = 3600
SEED = 42
MAX_RETRIES_PER_VEHICLE = 50

# Hotspot (Pimpri Market / Shagun Chowk approx)
HOTSPOT_LOC = (18.627, 73.800)
HOTSPOT_SIGMA = 1000

random.seed(SEED)
np.random.seed(SEED)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_edge_weight(edge, G, max_centrality, hx, hy):
    """
    Calculate demand probability weight for an edge.
    Factors: Hierarchy, Capacity, Centrality, Hotspot.
    """
    # 1. Road Hierarchy
    edge_type = edge.getType()
    if not edge_type:
        speed = edge.getSpeed()
        if speed > 20: hierarchy_score = 1.0
        elif speed > 13: hierarchy_score = 0.6
        elif speed > 8: hierarchy_score = 0.3
        else: hierarchy_score = 0.1
    else:
        et = edge_type.lower()
        if 'motorway' in et or 'trunk' in et: hierarchy_score = 5.0
        elif 'primary' in et: hierarchy_score = 3.0
        elif 'secondary' in et: hierarchy_score = 1.5
        elif 'tertiary' in et: hierarchy_score = 0.8
        else: hierarchy_score = 0.2

    # 2. Capacity
    lanes = edge.getLaneNumber()
    capacity_score = lanes * 0.5

    # 3. Centrality (Normalized)
    u, v = edge.getFromNode().getID(), edge.getToNode().getID()
    cent_u = G.nodes.get(u, {}).get('centrality', 0)
    cent_v = G.nodes.get(v, {}).get('centrality', 0)
    centrality_score = (cent_u + cent_v) / 2.0
    centrality_factor = 1.0 + (centrality_score / max_centrality) * 5.0 if max_centrality > 0 else 1.0

    # 4. Hotspot Bias
    shape = edge.getShape()
    if shape:
        mx, my = shape[len(shape)//2]
        dist = math.sqrt((mx - hx)**2 + (my - hy)**2)
        hotspot_factor = math.exp(-(dist**2) / (2 * HOTSPOT_SIGMA**2))
    else:
        hotspot_factor = 0
    
    final_w = hierarchy_score * capacity_score * centrality_factor * (1 + hotspot_factor * 3.0)
    return max(final_w, 0.001)

def generate_traffic():
    logger.info("=== Phase 2.1: Connectivity-Aware Trip Generation ===")
    logger.info("Loading network...")
    net = sumolib.net.readNet(NET_FILE)
    
    edges = net.getEdges()
    
    # Filter valid edges
    logger.info("Pre-filtering valid edges...")
    valid_start_edges = []
    valid_end_edges = []
    
    for e in edges:
        eid = e.getID()
        # Skip internal edges
        if eid.startswith(":"):
            continue
        # Must allow passenger vehicles
        if not e.allows("passenger"):
            continue
        
        # Check outgoing connections for start edges
        outgoing = e.getOutgoing()
        if outgoing:
            valid_start_edges.append(e)
        
        # Check incoming connections for end edges
        incoming = e.getIncoming()
        if incoming:
            valid_end_edges.append(e)
    
    logger.info(f"Valid start edges: {len(valid_start_edges)}")
    logger.info(f"Valid end edges: {len(valid_end_edges)}")
    
    # Build NetworkX graph for centrality and connectivity checking
    logger.info("Building graph for centrality and connectivity...")
    G = nx.DiGraph()  # Directed graph for route checking
    for e in edges:
        if e.getID().startswith(":"):
            continue
        u = e.getFromNode().getID()
        v = e.getToNode().getID()
        G.add_edge(u, v, edge_id=e.getID(), length=e.getLength())
    
    logger.info("Computing centrality...")
    centrality = nx.degree_centrality(G)
    max_cent = max(centrality.values()) if centrality else 1
    nx.set_node_attributes(G, centrality, 'centrality')
    
    # Convert Hotspot to Net Coords
    hx, hy = net.convertLonLat2XY(HOTSPOT_LOC[1], HOTSPOT_LOC[0])
    
    # Calculate weights for start edges
    logger.info("Calibrating demand model...")
    start_weights = []
    start_ids = []
    for e in valid_start_edges:
        w = get_edge_weight(e, G, max_cent, hx, hy)
        start_weights.append(w)
        start_ids.append(e.getID())
    
    total_sw = sum(start_weights)
    start_probs = [x / total_sw for x in start_weights]
    
    # Calculate weights for end edges
    end_weights = []
    end_ids = []
    for e in valid_end_edges:
        w = get_edge_weight(e, G, max_cent, hx, hy)
        end_weights.append(w)
        end_ids.append(e.getID())
    
    total_ew = sum(end_weights)
    end_probs = [x / total_ew for x in end_weights]
    
    # Build edge ID to node mapping for connectivity check
    edge_to_dest_node = {}
    edge_to_src_node = {}
    for e in edges:
        eid = e.getID()
        edge_to_dest_node[eid] = e.getToNode().getID()
        edge_to_src_node[eid] = e.getFromNode().getID()
    
    logger.info(f"Generating {NUM_VEHICLES} valid trips with connectivity validation...")
    
    trips = []
    total_rejections = 0
    total_retries = 0
    
    while len(trips) < NUM_VEHICLES:
        retries = 0
        found = False
        
        # Select origin
        origin_id = np.random.choice(start_ids, p=start_probs)
        origin_node = edge_to_dest_node.get(origin_id) # Route starts from end of origin edge
        
        while retries < MAX_RETRIES_PER_VEHICLE and not found:
            # Select destination
            dest_id = np.random.choice(end_ids, p=end_probs)
            
            # Skip if same edge
            if origin_id == dest_id:
                retries += 1
                total_rejections += 1
                continue
            
            dest_node = edge_to_src_node.get(dest_id) # Route ends at start of dest edge
            
            # Check connectivity using NetworkX (fast, in-memory)
            if origin_node and dest_node and nx.has_path(G, origin_node, dest_node):
                # Valid path exists
                depart_time = len(trips) * (DURATION / NUM_VEHICLES) + random.uniform(0, DURATION / NUM_VEHICLES)
                trips.append({
                    "id": f"veh{len(trips)}",
                    "origin": origin_id,
                    "dest": dest_id,
                    "depart": f"{depart_time:.2f}"
                })
                found = True
            else:
                retries += 1
                total_rejections += 1
        
        total_retries += retries
        
        if not found:
            # Couldn't find valid dest for this origin, pick new origin next iteration
            logger.warning(f"Max retries reached for origin {origin_id}, resampling origin...")
    
    avg_retries = total_retries / NUM_VEHICLES if NUM_VEHICLES > 0 else 0
    
    logger.info("=== Generation Statistics ===")
    logger.info(f"Total valid trips: {len(trips)}")
    logger.info(f"Total rejected OD pairs: {total_rejections}")
    logger.info(f"Average retries per vehicle: {avg_retries:.2f}")
    
    # Sort by departure time
    trips.sort(key=lambda x: float(x['depart']))
    
    logger.info(f"Writing {TRIPS_FILE}...")
    with open(TRIPS_FILE, "w") as f:
        f.write('<routes>\n')
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="55.55" guiShape="passenger"/>\n')
        
        for t in trips:
            f.write(f'    <trip id="{t["id"]}" type="car" depart="{t["depart"]}" from="{t["origin"]}" to="{t["dest"]}" />\n')
            
        f.write('</routes>\n')
    
    logger.info(f"trips.xml saved with {len(trips)} vehicles.")
    print(f"Phase 2.1 Complete: {len(trips)} valid trips generated.")

if __name__ == "__main__":
    generate_traffic()
