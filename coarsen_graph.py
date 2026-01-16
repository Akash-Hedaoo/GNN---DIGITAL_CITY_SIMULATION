import os
import sys
import sumolib
import pandas as pd
import logging

# Setup logging
log_file = os.path.join("logs", "ai_training.log")
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

NET_FILE = os.path.join("outputs", "metro_corridor.net.xml")
DATA_FILE = os.path.join("outputs", "sumo_training_data.csv")
COARSE_EDGES_FILE = os.path.join("outputs", "coarse_edges.txt")
COARSE_DATA_FILE = os.path.join("outputs", "coarse_training_data.csv")

def coarsen_graph():
    logging.info("Starting graph coarsening...")
    
    # 1. Load SUMO network and filter edges
    try:
        net = sumolib.net.readNet(NET_FILE)
        all_edges = net.getEdges()
        original_count = len(all_edges)
        
        filtered_edges = []
        for e in all_edges:
            eid = e.getID()
            if eid.startswith(":"):
                continue
            
            # Criteria: length >= 100m AND speed >= 13.9 m/s
            if e.getLength() >= 100 and e.getSpeed() >= 13.9:
                filtered_edges.append(eid)
                
        reduced_count = len(filtered_edges)
        logging.info(f"Edge filtering complete. Original: {original_count}, Kept: {reduced_count}")
        
        with open(COARSE_EDGES_FILE, "w") as f:
            for eid in filtered_edges:
                f.write(eid + "\n")
                
    except Exception as e:
        logging.error(f"Failed to process network: {e}")
        sys.exit(1)

    # 2. Filter dataset
    try:
        logging.info("Filtering traffic dataset...")
        # Read in chunks if file is large, but 160MB fits in memory usually. 
        # Using pandas efficiently.
        df = pd.read_csv(DATA_FILE)
        rows_before = len(df)
        
        # Filter
        df_coarse = df[df['edge_id'].isin(filtered_edges)]
        rows_after = len(df_coarse)
        
        # Aggregate duplicates if any (shouldn't be for (step, edge), but good practice)
        df_coarse = df_coarse.groupby(['step', 'edge_id'], as_index=False).mean()
        
        df_coarse.to_csv(COARSE_DATA_FILE, index=False)
        logging.info(f"Data filtering complete. Rows: {rows_before} -> {rows_after}")
        
    except Exception as e:
        logging.error(f"Failed to process data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    coarsen_graph()
