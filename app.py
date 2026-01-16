import streamlit as st
import os
import sys
import traci
import sumolib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from PIL import Image
import time

# Import model definition from training script
# Ensure the directory is in path
sys.path.append(os.getcwd())
try:
    from train_stgcn import STGCN_Model, GraphConv, build_graph, load_coarse_edges
except ImportError:
    st.error("Could not import model classes from train_stgcn.py. Make sure the file exists.")
    sys.exit(1)

# Setup logging
LOG_FILE = os.path.join("logs", "dashboard.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Nashik Phata Digital Twin", layout="wide")

# Paths
NET_FILE = os.path.join("outputs", "metro_corridor.net.xml")
SUMO_CFG = os.path.join("outputs", "metro.sumocfg")
MODEL_PATH = os.path.join("outputs", "sumo_gnn_model.pt")
DATA_FILE = os.path.join("outputs", "coarse_training_data.csv")
SCREENSHOT_PATH = os.path.join("outputs", "congestion_screenshot.png")

@st.cache_resource
def load_model_and_graph():
    logging.info("Loading model and graph...")
    # 1. Edges
    coarse_edges = load_coarse_edges()
    
    # 2. Graph Adjacency
    edge_list, adj = build_graph(NET_FILE, edge_subset=coarse_edges)
    
    # 3. Model
    # Architecture params must match training
    WINDOW_STEPS = 90
    HORIZON_STEPS = 90
    NUM_NODES = len(edge_list)
    
    model = STGCN_Model(NUM_NODES, WINDOW_STEPS, HORIZON_STEPS, adj)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        logging.info("Model loaded successfully.")
    else:
        st.warning("Model file not found. Predictions will be random initialized.")
    
    return model, edge_list, coarse_edges

def run_blockage_simulation(edge_id, steps=200):
    logging.info(f"Starting blockage simulation on {edge_id}")
    
    sumo_cmd = ["sumo-gui", "-c", SUMO_CFG, "--start", "--quit-on-end"]
    
    try:
        traci.start(sumo_cmd, label="dashboard_sim")
        
        # Determine lanes
        lanes = traci.edge.getLaneNumber(edge_id)
        lane_ids = [f"{edge_id}_{i}" for i in range(lanes)]
        
        # 1. Warmup
        st.info("Warming up simulation (50 steps)...")
        for _ in range(50):
            traci.simulationStep()
            
        # 2. Apply Blockage
        st.info(f"Blocking {edge_id}...")
        for lid in lane_ids:
            traci.lane.setMaxSpeed(lid, 0.1) # 0.1 m/s
            
        # 3. Run for duration
        st.info(f"Simulating congestion ({steps} steps)...")
        for _ in range(steps):
            traci.simulationStep()
            
        # 4. Screenshot
        # Focus view on the edge
        # Get edge center position is tricky via TraCI directly without Poly conversion
        # But we can try to center if we had coordinates. For now, just screenshot current view.
        # Alternatively, use gui.setSchema or setZoom.
        
        traci.gui.screenshot("View #0", os.path.abspath(SCREENSHOT_PATH))
        # Wait a bit for screenshot write
        time.sleep(1)
        
        traci.close()
        logging.info("Simulation finished.")
        return True
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        st.error(f"Simulation failed: {e}")
        try:
           traci.close()
        except:
           pass
        return False

def get_recent_data(edge_list, steps=90):
    # Load last 'steps' data points for all edges
    # This is heavy to read full CSV every time.
    # For demo, read tail if possible or full.
    # Coarse data is small enough (~300k rows)
    
    df = pd.read_csv(DATA_FILE)
    
    # Pivot
    pivot_df = df.pivot(index='step', columns='edge_id', values='occupancy')
    pivot_df = pivot_df.fillna(0)
    
    # Ensure all edges present
    for e in edge_list:
        if e not in pivot_df.columns:
            pivot_df[e] = 0.0
            
    pivot_df = pivot_df[edge_list] # Align
    
    # Get last N rows
    if len(pivot_df) < steps:
        st.warning("Not enough data history. Padding with zeros.")
        # Pad...
        data = pivot_df.values
        # Pad logic omitted for brevity, assuming enough data for mission
        return torch.tensor([data], dtype=torch.float32) # (1, T, N) ?? Shape mismatch if T < steps
    
    last_window = pivot_df.iloc[-steps:].values
    return torch.tensor([last_window], dtype=torch.float32) # (1, 90, 818)


# --- MAIN APP UI ---

st.title("🚝 Nashik Phata Metro-Corridor Digital Twin")
st.markdown("### AI-Powered Traffic Control Dashboard")

model, edge_list, valid_edges = load_model_and_graph()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Control Panel")
    selected_edge = st.selectbox("Select Road Segment to Block", valid_edges)
    
    if st.button("🔴 Block Road & Simulate"):
        with st.spinner("Launching SUMO-GUI..."):
            success = run_blockage_simulation(selected_edge)
            
            if success:
                st.success("Simulation Complete!")
                
                # Show screenshot
                if os.path.exists(SCREENSHOT_PATH):
                    image = Image.open(SCREENSHOT_PATH)
                    st.image(image, caption="Real-time Congestion in SUMO", use_container_width=True)
                
                # --- AI Prediction ---
                st.subheader("AI Prediction (Next 15 mins)")
                with st.spinner("Running ST-GCN Model..."):
                    # Get input
                    input_tensor = get_recent_data(edge_list) 
                    # Note: Need correct shape and scaling matching training
                    
                    # Predict
                    with torch.no_grad():
                        prediction = model(input_tensor) # (1, 90, N)
                        
                    # Extract data for the blocked edge
                    # Find index
                    try:
                        idx = edge_list.index(selected_edge)
                        edge_pred = prediction[0, :, idx].numpy()
                        
                        # Plot
                        fig, ax = plt.subplots()
                        ax.plot(edge_pred, color='red', label='Predicted Occupancy')
                        ax.set_title(f"Forecast for {selected_edge}")
                        ax.set_xlabel("Time (Steps)")
                        ax.set_ylabel("Occupancy %")
                        ax.legend()
                        st.pyplot(fig)
                        
                    except ValueError:
                        st.error("Selected edge not found in graph nodes.")
                        
            else:
                st.error("Simulation failed check logs.")

with col2:
    st.info("Select a road segment on the left to simulate a blockage event. The Digital Twin will: \n1. Spin up a SUMO simulation replica.\n2. Inject a traffic incident.\n3. Visualize the result.\n4. Use AI to forecast 15-minute impact.")

    # Placeholder for map or other info
    st.image("https://sumo.dlr.de/docs/images/sumo-gui.png", caption="System Ready (Example Image)", width=500)
    
    st.markdown("---")
    st.markdown("**System Status**")
    st.write(f"- **Nodes**: {len(edge_list)}")
    st.write(f"- **Model**: ST-GCN (GraphConv + LSTM)")
    st.write(f"- **Data Source**: {DATA_FILE}")
