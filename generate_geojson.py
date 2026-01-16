import os
import json
import sumolib
import logging

# Setup logging
logging.basicConfig(
    filename=os.path.join("logs", "dashboard.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

NET_FILE = os.path.join("outputs", "pune.net.xml")
OUTPUT_FILE = os.path.join("outputs", "road_network.geojson")

def generate_geojson():
    """Convert SUMO network to GeoJSON format."""
    logging.info(f"Loading SUMO network from {NET_FILE}...")
    
    try:
        net = sumolib.net.readNet(NET_FILE)
    except Exception as e:
        logging.error(f"Failed to load network: {e}")
        return
    
    features = []
    edges = net.getEdges()
    
    logging.info(f"Processing {len(edges)} edges...")
    
    processed = 0
    for edge in edges:
        edge_id = edge.getID()
        
        # Skip internal edges (junctions)
        if edge_id.startswith(":"):
            continue
        
        try:
            # Get edge shape (list of (x, y) coordinates)
            shape = edge.getShape()
            
            # Convert to Lon/Lat
            # SUMO stores coordinates, network has projection info
            lonlat_coords = []
            for x, y in shape:
                lon, lat = net.convertXY2LonLat(x, y)
                lonlat_coords.append([lon, lat])
            
            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": lonlat_coords
                },
                "properties": {
                    "edge_id": edge_id
                }
            }
            
            features.append(feature)
            processed += 1
            
            if processed % 1000 == 0:
                logging.info(f"Processed {processed} edges...")
                
        except Exception as e:
            logging.warning(f"Failed to process edge {edge_id}: {e}")
            continue
    
    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Write to file
    logging.info(f"Writing GeoJSON to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(geojson, f)
    
    logging.info(f"Successfully generated GeoJSON with {len(features)} road segments.")
    print(f"✓ Generated {OUTPUT_FILE}")
    print(f"  Total features: {len(features)}")
    
    # Print file size
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    generate_geojson()
