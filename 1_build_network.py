import os
import sys
import osmnx as ox
import subprocess
import logging

# Setup logging
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(filename="logs/sumo_backend.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

PLACE_NAME = "Pimpri-Chinchwad, Pune"
OSM_FILE = os.path.join("outputs", "pune.osm")
NET_FILE = os.path.join("outputs", "pune.net.xml")

def build_network():
    logging.info(f"Starting network build for {PLACE_NAME}...")
    
    # 1. Download OSM Data
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    try:
        logging.info(f"Downloading OSM data for {PLACE_NAME}...")
        
        # Configure osmnx to allow XML saving
        ox.settings.all_oneway = True
        
        try:
            # Download drive network by name
            G = ox.graph_from_place(PLACE_NAME, network_type='drive', simplify=False)
        except Exception as e:
            logging.warning(f"Name search failed: {e}. Trying generic 'Pimpri-Chinchwad'...")
            try:
                 G = ox.graph_from_place("Pimpri-Chinchwad", network_type='drive', simplify=False)
            except:
                logging.warning("Name search failed. Falling back to coordinates (18.6298, 73.7997), dist=4000m...")
                point = (18.6298, 73.7997)
                G = ox.graph_from_point(point, dist=4000, network_type='drive', simplify=False)
        
        # Save to OSM XML
        ox.save_graph_xml(G, filepath=OSM_FILE)
        logging.info(f"OSM data saved to {OSM_FILE}")
        
    except Exception as e:
        logging.error(f"Failed to download OSM data: {e}")
        sys.exit(1)

    # 2. Convert to SUMO .net.xml
    logging.info("Converting to SUMO network...")
    # netconvert command
    # Assuming netconvert is in PATH or SUMO_HOME is set. 
    # If not, we might need to find it. But user environment had it before.
    
    cmd = [
        "netconvert",
        "--osm-files", OSM_FILE,
        "-o", NET_FILE,
        "--geometry.remove", "true",
        "--ramps.guess", "true",
        "--junctions.join", "true",
        "--tls.guess-signals", "true",
        "--tls.discard-simple", "true",
        "--no-turnarounds", "true"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"Network file generated: {NET_FILE}")
    except subprocess.CalledProcessError as e:
        logging.error(f"netconvert failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("netconvert not found. Please ensure SUMO is installed and in PATH.")
        sys.exit(1)

if __name__ == "__main__":
    build_network()
