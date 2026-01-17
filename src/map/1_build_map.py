import osmnx as ox
import os
import sys
import subprocess
import logging
import xml.etree.ElementTree as ET

# Configuration
LOG_FILE = os.path.join("logs", "system", "phase1_map.log")
OSM_FILE = os.path.join("data", "raw", "osm", "pune.osm")
NET_FILE = os.path.join("data", "processed", "networks", "pune.net.xml")
LOCATION = "Nashik Phata, Pimpri-Chinchwad, India"
DIST = 3500
NETWORK_TYPE = 'drive'

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

def check_dependencies():
    """Check if netconvert is available."""
    try:
        subprocess.run(["netconvert", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("netconvert is available.")
    except FileNotFoundError:
        logger.error("netconvert not found! Please ensure SUMO is installed and in PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"netconvert check failed: {e}")
        sys.exit(1)

def build_map():
    logger.info("Starting Phase 1: Digital Twin Map Construction")
    
    # 1. Data Acquisition
    logger.info(f"Downloading OSM data for: {LOCATION} (Radius: {DIST}m)")
    try:
        # Check osmnx version to handle API changes if necessary
        logger.info(f"osmnx version: {ox.__version__}")
        
        # Geocode the location first
        try:
            point = ox.geocode(LOCATION)
            logger.info(f"Geocoded '{LOCATION}' to {point}")
        except AttributeError:
             from osmnx import geocoder
             point = geocoder.geocode(LOCATION)
        
        # IMPORTANT: Download UNSIMPLIFIED graph to allow OSM XML export.
        # netconvert will handle the simplification properly via --geometry.remove etc.
        G = ox.graph_from_point(point, dist=DIST, network_type=NETWORK_TYPE, simplify=False)
        logger.info(f"Graph downloaded. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    except Exception as e:
        logger.error(f"Failed to download graph: {e}")
        sys.exit(1)

    # 2. Graph Optimization
    # SKIPPING osmnx simplification to generate valid OSM XML.
    # netconvert's optimization flags will ensure the final network is simplified.
    logger.info("Skipping osmnx internal simplification to allow OSM XML export (netconvert will optimize).")
    G_simple = G 

    # 3. Export Raw Data
    logger.info(f"Saving to {OSM_FILE}...")
    try:
        # Use save_graph_xml which requires unsimplified graph
        if hasattr(ox, 'save_graph_xml'):
             ox.save_graph_xml(G_simple, filepath=OSM_FILE)
        elif hasattr(ox.io, 'save_graph_xml'):
             ox.io.save_graph_xml(G_simple, filepath=OSM_FILE)
        else:
             logger.error("OSMnx 2.0+ removed save_graph_xml. Cannot export to .osm directly.")
             sys.exit(1)

        logger.info("OSM file saved.")
    except Exception as e:
        logger.error(f"Failed to save OSM file: {e}")
        sys.exit(1)

    # 4. Convert to SUMO Network
    logger.info(f"Converting to SUMO network: {NET_FILE}")
    cmd = [
        "netconvert",
        "--osm-files", OSM_FILE,
        "-o", NET_FILE,
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info("netconvert output:\n" + result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"netconvert failed:\n{e.stderr}")
        sys.exit(1)

    # 5. Validation
    logger.info("Validating output network...")
    if not os.path.exists(NET_FILE):
        logger.error("Output file not found!")
        sys.exit(1)
        
    try:
        tree = ET.parse(NET_FILE)
        root = tree.getroot()
        edges = len(root.findall('edge'))
        junctions = len(root.findall('junction'))
        
        logger.info("--------------------------------------------------")
        logger.info(f"Total number of edges: {edges}")
        logger.info(f"Total number of junctions (nodes): {junctions}")
        logger.info("Phase 1 Complete: SUMO Network Ready")
        print(f"Total number of edges: {edges}")
        print(f"Total number of junctions (nodes): {junctions}")
        print("Phase 1 Complete: SUMO Network Ready")
        logger.info("--------------------------------------------------")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_dependencies()
    build_map()
