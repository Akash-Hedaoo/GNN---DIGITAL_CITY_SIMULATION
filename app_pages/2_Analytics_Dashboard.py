import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import traci
import torch
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_connector import SumoManager
# Import model classes
try:
    from train_stgcn import STGCN_Model, GraphConv, build_graph, load_coarse_edges
except ImportError:
    pass # Handle below

st.set_page_config(page_title="Traffic Analytics", layout="wide")
st.title("📊 Real-Time Traffic Analytics")

# --- Constants & Setup ---
# Use pune.net.xml (from Mission 1) for the current Pimpri-Chinchwad simulation
# Note: If model exists from previous work, it may reference different network
NET_FILE = os.path.join("outputs", "pune.net.xml")
MODEL_PATH = os.path.join("outputs", "sumo_gnn_model.pt")
COARSE_EDGES_FILE = os.path.join("outputs", "coarse_edges.txt")

# --- Session State ---
if 'sumo_manager' not in st.session_state:
    st.warning("⚠️ No simulation found. Please start it from the Live Map page first.")
    st.stop()

mgr = st.session_state.sumo_manager

if 'traffic_buffer' not in st.session_state:
    st.session_state.traffic_buffer = []

if 'congestion_history' not in st.session_state:
    st.session_state.congestion_history = []

# --- Model Loading ---
@st.cache_resource
def load_ai_model():
    """Load ST-GCN model if available from previous training."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    
    if not os.path.exists(COARSE_EDGES_FILE):
        return None, None
    
    try:
        coarse_edges = load_coarse_edges()
        # Try to build graph - use metro_corridor if exists, else pune
        net_for_model = os.path.join("outputs", "metro_corridor.net.xml")
        if not os.path.exists(net_for_model):
            net_for_model = NET_FILE
            
        edge_list, adj = build_graph(net_for_model, edge_subset=coarse_edges)
        
        model = STGCN_Model(len(edge_list), 90, 90, adj)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        return model, edge_list
    except Exception as e:
        logging.error(f"Model load failed: {e}")
        return None, None

model, model_edges = load_ai_model()

# --- Initialize Metrics ---
veh_count = 0
avg_speed = 0.0
jam_count = 0
current_snapshot = {}

# --- Metrics Update ---
if mgr.running:
    # 1. System Metrics
    # Vehicles
    try:
        veh_count = traci.vehicle.getIDCount() 
    except:
        veh_count = 0
        
    # Speed & Jams
    # We poll all edges? Or just model edges? 
    # For "System Metrics", maybe all? But querying 15k edges is slow.
    # Color map used edge_list from live state.
    # Let's just use the cached state from manager if possible or query major ones.
    # Manager's get_live_state only returns colored ones.
    # We can query all edges from model_edges (818) for efficiency.
    
    total_speed = 0
    jam_count = 0
    edge_count = 0
    
    current_snapshot = {} # For model buffer
    
    # If model_edges not loaded, we can't do specific list easily without reading file.
    # Instead of using potentially outdated model_edges, get edges from active simulation
    if model_edges and len(model_edges) > 0:
        # Filter to only edges that exist in current simulation
        try:
            active_edges = traci.edge.getIDList()
            target_edges = [e for e in model_edges if e in active_edges]
        except:
            target_edges = []
    else:
        # Use edges from current simulation directly
        try:
            all_edges = traci.edge.getIDList()
            # Filter out internal edges and limit to reasonable number for performance
            target_edges = [e for e in all_edges if not e.startswith(":")][:500]
        except:
            target_edges = []

    for eid in target_edges:
        try:
            occ = traci.edge.getLastStepOccupancy(eid)
            speed = traci.edge.getLastStepMeanSpeed(eid)
            
            current_snapshot[eid] = occ
            
            total_speed += speed
            if occ >= 0.8:
                jam_count += 1
            edge_count += 1
        except:
            current_snapshot[eid] = 0.0

    avg_speed = total_speed / edge_count if edge_count > 0 else 0
    
    # Update Buffer
    st.session_state.traffic_buffer.append(current_snapshot)
    if len(st.session_state.traffic_buffer) > 90:
        st.session_state.traffic_buffer.pop(0)
else:
    st.info("⏸️ Simulation is paused. Start it from the Live Map to see live data.")

# --- UI Layout ---

# Top Cards
col1, col2, col3 = st.columns(3)
col1.metric("Active Vehicles", f"{veh_count}")
col2.metric("Avg City Speed", f"{avg_speed:.2f} m/s")
col3.metric("Jammed Roads", f"{jam_count}", delta_color="inverse")

# Graphs
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("⚠️ Top 5 Congested Roads")
    if mgr.running and current_snapshot:
        # Sort by occupancy
        sorted_edges = sorted(current_snapshot.items(), key=lambda x: x[1], reverse=True)[:5]
        df_top = pd.DataFrame(sorted_edges, columns=["Road ID", "Occupancy"])
        st.bar_chart(df_top.set_index("Road ID"))
    else:
        st.info("Simulation not running.")

with col_g2:
    st.subheader("📈 Congestion Trend (Model Input)")
    # Plot average occupancy of the network over buffer
    if st.session_state.traffic_buffer:
        # Mean occupancy per step
        history_means = [np.mean(list(step.values())) for step in st.session_state.traffic_buffer]
        st.line_chart(history_means)
    else:
        st.info("No data yet.")

# AI Prediction section
st.markdown("---")
st.subheader("🤖 ST-GCN Future Prediction (Next 15 mins)")

if model and len(st.session_state.traffic_buffer) == 90:
    # Prepare input
    # Shape: (1, 90, N)
    # Align Buffer with model_edges order
    input_data = []
    for step_data in st.session_state.traffic_buffer:
        row = [step_data.get(e, 0.0) for e in model_edges]
        input_data.append(row)
    
    input_tensor = torch.tensor([input_data], dtype=torch.float32) # (1, 90, 818)
    
    with torch.no_grad():
        prediction = model(input_tensor) # (1, 90, 818)
    
    # Visualization:
    # We can't plot 818 lines.
    # Plot "Predicted Network Congestion Level" (Mean)
    pred_means = torch.mean(prediction[0], dim=1).numpy()
    
    # Or plot specific jammed edge?
    # Let's find index of max occupancy currently
    if current_snapshot:
        max_edge = max(current_snapshot, key=current_snapshot.get)
        if max_edge in model_edges:
            idx = model_edges.index(max_edge)
            edge_pred = prediction[0, :, idx].numpy()
            
            st.write(f"Forecast for currently worst road: **{max_edge}**")
            st.line_chart(pd.DataFrame({"Predicted Occupancy": edge_pred}))
            
    # Also plot network mean trend
    # st.write("Network Mean Forecast")
    # st.line_chart(pred_means)
    
elif model:
    st.info(f"Filling Data Buffer... ({len(st.session_state.traffic_buffer)}/90 steps collected)")
else:
    st.warning("Model not loaded or simulation inactive.")

# Auto Refresh (only if simulation is running)
if mgr.running:
    time.sleep(3)
    st.rerun()
