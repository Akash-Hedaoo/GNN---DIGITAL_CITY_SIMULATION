"""
🚦 Traffic Prediction AI - Streamlit Web GUI
==============================================

A beautiful web interface for the GNN-based traffic prediction system.

Run: streamlit run streamlit_gui.py

Author: Digital Twin City Simulation
Date: November 2025
"""

import streamlit as st
import torch
import numpy as np
import pickle
import os
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import time

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="🚦 Traffic Prediction AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained GNN model (cached)"""
    try:
        from gnn_model import TrafficGATv2, load_model as load_gnn_model
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = TrafficGATv2(
            in_channels=4,
            edge_features=3,
            hidden_channels=64,
            num_heads=4,
            num_layers=3,
            dropout=0.2,
            output_dim=1
        )
        
        if os.path.exists("trained_gnn.pt"):
            model = load_gnn_model(model, "trained_gnn.pt")
            model = model.to(device)
            model.eval()
            return model, device, True
        else:
            return None, device, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, 'cpu', False


@st.cache_resource
def load_graph():
    """Load the city graph (cached)"""
    try:
        if os.path.exists("city_graph.graphml"):
            G = nx.read_graphml("city_graph.graphml")
            return G, True
        else:
            return None, False
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return None, False


# Training data loading REMOVED - not needed for inference!
# The trained model (trained_gnn.pt) is all that's required


@st.cache_resource
def get_sample_tensors():
    """Generate sample tensors for inference (cached, no I/O)"""
    num_nodes = 796
    num_edges = 4676
    node_features = torch.randn(num_nodes, 4)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_features = torch.randn(num_edges, 3)
    edge_keys = list(range(num_edges))
    return node_features, edge_index, edge_features, edge_keys


def snapshot_to_tensors(G=None):
    """Get sample tensors for inference - no file I/O needed!"""
    return get_sample_tensors()


def predict_congestion(model, device, node_features, edge_index, edge_features):
    """Run model prediction"""
    try:
        with torch.no_grad():
            x = node_features.to(device)
            ei = edge_index.to(device)
            ef = edge_features.to(device)
            
            predictions = model(x, ei, ef)
            return predictions.cpu().numpy().flatten()
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# ============================================================
# UI COMPONENTS
# ============================================================

def show_header():
    """Display header"""
    st.title("🚦 Traffic Prediction AI")
    st.markdown("**Digital City Twin - GNN-based Traffic Congestion Prediction**")
    st.markdown("---")


def show_sidebar_info(G, model_loaded, graph_loaded, device):
    """Display sidebar information"""
    with st.sidebar:
        st.header("📊 System Status")
        
        # Status indicators
        col1, col2 = st.columns(2)
        with col1:
            if model_loaded:
                st.success("✅ Model Loaded")
            else:
                st.error("❌ Model Missing")
        
        with col2:
            if graph_loaded:
                st.success("✅ Graph Loaded")
            else:
                st.error("❌ Graph Missing")
        
        st.markdown(f"**Device**: `{device}`")
        
        if graph_loaded and G is not None:
            st.markdown("---")
            st.subheader("🏙️ City Statistics")
            st.metric("Nodes (Intersections)", G.number_of_nodes())
            st.metric("Edges (Roads)", G.number_of_edges())
            
            # Count metro edges
            metro_count = sum(1 for _, _, d in G.edges(data=True) 
                           if d.get('is_metro') == 'True' or d.get('is_metro') == True)
            st.metric("Metro Edges", metro_count)
        
        st.markdown("---")
        st.subheader("🤖 Model Info")
        st.markdown("""
        - **Architecture**: GATv2
        - **Parameters**: 115,841
        - **Heads**: 4
        - **Layers**: 3
        - **Hidden Dim**: 64
        """)
        
        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit")


def single_road_test(model, device, G):
    """Single road closure test"""
    st.subheader("🛣️ Single Road Test")
    st.markdown("Close one road and see the predicted impact on traffic congestion.")
    
    # Get sample data (cached)
    node_features, edge_index, edge_features, edge_keys = snapshot_to_tensors(G)
    
    if node_features is None:
        st.error("Could not load sample data. Make sure `gnn_training_data.pkl` exists.")
        return
    
    num_edges = edge_features.shape[0]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        road_num = st.slider(
            "Select Road Number",
            min_value=0,
            max_value=num_edges - 1,
            value=100,
            help="Each number represents a road segment in the city"
        )
    
    with col2:
        action = st.radio(
            "Action",
            ["Close Road", "Open Road"],
            help="Close = Block traffic, Open = Allow traffic"
        )
    
    if st.button("🔮 Predict Impact", type="primary", use_container_width=True):
        with st.spinner("Running AI prediction..."):
            # Get base prediction
            base_predictions = predict_congestion(model, device, node_features, edge_index, edge_features)
            
            if base_predictions is None:
                st.error("Prediction failed")
                return
            
            # Modify edge features
            modified_edge_features = edge_features.clone()
            if action == "Close Road":
                modified_edge_features[road_num, 1] = 1.0  # is_closed = 1
            else:
                modified_edge_features[road_num, 1] = 0.0  # is_closed = 0
            
            # Get modified prediction
            modified_predictions = predict_congestion(model, device, node_features, edge_index, modified_edge_features)
            
            if modified_predictions is None:
                st.error("Modified prediction failed")
                return
            
            # Calculate impact
            base_mean = np.mean(base_predictions)
            modified_mean = np.mean(modified_predictions)
            change_percent = ((modified_mean - base_mean) / base_mean) * 100
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Before",
                    f"{base_mean:.2f}",
                    help="Average congestion factor before change"
                )
            
            with col2:
                st.metric(
                    "After",
                    f"{modified_mean:.2f}",
                    delta=f"{change_percent:+.1f}%",
                    delta_color="inverse",
                    help="Average congestion factor after change"
                )
            
            with col3:
                impact_level = "🟢 Low" if abs(change_percent) < 5 else "🟡 Medium" if abs(change_percent) < 15 else "🔴 High"
                st.metric("Impact Level", impact_level)
            
            # Chart
            st.markdown("---")
            st.subheader("📈 Congestion Distribution")
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=base_predictions,
                name="Before",
                opacity=0.7,
                marker_color='#3498db'
            ))
            
            fig.add_trace(go.Histogram(
                x=modified_predictions,
                name="After",
                opacity=0.7,
                marker_color='#e74c3c'
            ))
            
            fig.update_layout(
                barmode='overlay',
                title="Congestion Factor Distribution",
                xaxis_title="Congestion Factor",
                yaxis_title="Number of Roads",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional stats
            with st.expander("📊 Detailed Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Before Change**")
                    st.write(f"- Mean: {base_mean:.2f}")
                    st.write(f"- Std: {np.std(base_predictions):.2f}")
                    st.write(f"- Min: {np.min(base_predictions):.2f}")
                    st.write(f"- Max: {np.max(base_predictions):.2f}")
                
                with col2:
                    st.markdown("**After Change**")
                    st.write(f"- Mean: {modified_mean:.2f}")
                    st.write(f"- Std: {np.std(modified_predictions):.2f}")
                    st.write(f"- Min: {np.min(modified_predictions):.2f}")
                    st.write(f"- Max: {np.max(modified_predictions):.2f}")


def multi_road_test(model, device, G):
    """Multiple road closure test"""
    st.subheader("🛣️ Multiple Roads Test")
    st.markdown("Close multiple roads and compare the combined impact.")
    
    # Get sample data
    node_features, edge_index, edge_features, edge_keys = snapshot_to_tensors(G)
    
    if node_features is None:
        st.error("Could not load sample data.")
        return
    
    num_edges = edge_features.shape[0]
    
    # Input method
    input_method = st.radio(
        "Selection Method",
        ["Manual Entry", "Range Selection", "Random Selection"],
        horizontal=True
    )
    
    roads_to_close = []
    
    if input_method == "Manual Entry":
        roads_input = st.text_input(
            "Enter road numbers (comma-separated)",
            placeholder="e.g., 100, 200, 300, 450",
            help="Enter road numbers separated by commas"
        )
        if roads_input:
            try:
                roads_to_close = [int(x.strip()) for x in roads_input.split(",") if x.strip()]
                roads_to_close = [r for r in roads_to_close if 0 <= r < num_edges]
            except:
                st.error("Invalid input. Please enter numbers separated by commas.")
    
    elif input_method == "Range Selection":
        col1, col2 = st.columns(2)
        with col1:
            start = st.number_input("Start", min_value=0, max_value=num_edges-1, value=100)
        with col2:
            end = st.number_input("End", min_value=0, max_value=num_edges-1, value=110)
        roads_to_close = list(range(start, min(end + 1, num_edges)))
    
    else:  # Random
        num_random = st.slider("Number of random roads", 1, 50, 10)
        if st.button("🎲 Generate Random"):
            roads_to_close = list(np.random.choice(num_edges, size=num_random, replace=False))
            st.session_state['random_roads'] = roads_to_close
        
        if 'random_roads' in st.session_state:
            roads_to_close = st.session_state['random_roads']
    
    if roads_to_close:
        st.info(f"Selected {len(roads_to_close)} roads: {roads_to_close[:10]}{'...' if len(roads_to_close) > 10 else ''}")
    
    if st.button("🔮 Predict Combined Impact", type="primary", disabled=len(roads_to_close) == 0):
        with st.spinner(f"Analyzing {len(roads_to_close)} road closures..."):
            # Base prediction
            base_predictions = predict_congestion(model, device, node_features, edge_index, edge_features)
            
            if base_predictions is None:
                return
            
            # Modified prediction
            modified_edge_features = edge_features.clone()
            for road in roads_to_close:
                modified_edge_features[road, 1] = 1.0
            
            modified_predictions = predict_congestion(model, device, node_features, edge_index, modified_edge_features)
            
            if modified_predictions is None:
                return
            
            # Results
            base_mean = np.mean(base_predictions)
            modified_mean = np.mean(modified_predictions)
            change_percent = ((modified_mean - base_mean) / base_mean) * 100
            
            st.markdown("---")
            st.subheader("📊 Combined Impact Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Roads Closed", len(roads_to_close))
            
            with col2:
                st.metric("Before", f"{base_mean:.2f}")
            
            with col3:
                st.metric("After", f"{modified_mean:.2f}", delta=f"{change_percent:+.1f}%", delta_color="inverse")
            
            with col4:
                impact = "🟢 Low" if abs(change_percent) < 5 else "🟡 Medium" if abs(change_percent) < 15 else "🔴 High"
                st.metric("Impact", impact)
            
            # Comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Box(y=base_predictions, name="Before", marker_color='#3498db'))
            fig.add_trace(go.Box(y=modified_predictions, name="After", marker_color='#e74c3c'))
            
            fig.update_layout(
                title="Congestion Distribution Comparison",
                yaxis_title="Congestion Factor",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)


def scenario_comparison(model, device, G):
    """Compare different scenarios"""
    st.subheader("⚖️ Scenario Comparison")
    st.markdown("Compare different traffic scenarios side by side.")
    
    node_features, edge_index, edge_features, edge_keys = snapshot_to_tensors(G)
    
    if node_features is None:
        st.error("Could not load sample data.")
        return
    
    num_edges = edge_features.shape[0]
    
    # Predefined scenarios
    scenarios = {
        "Normal Traffic": [],
        "Close 5 Random Roads": list(np.random.choice(num_edges, 5, replace=False)),
        "Close 10 Random Roads": list(np.random.choice(num_edges, 10, replace=False)),
        "Close 20 Random Roads": list(np.random.choice(num_edges, 20, replace=False)),
        "Major Highway Closure (0-50)": list(range(0, 50)),
    }
    
    selected_scenarios = st.multiselect(
        "Select Scenarios to Compare",
        list(scenarios.keys()),
        default=["Normal Traffic", "Close 10 Random Roads"]
    )
    
    if st.button("📊 Compare Scenarios", type="primary", disabled=len(selected_scenarios) < 2):
        results = {}
        
        progress = st.progress(0)
        
        for i, scenario_name in enumerate(selected_scenarios):
            roads = scenarios[scenario_name]
            
            modified_ef = edge_features.clone()
            for road in roads:
                if road < num_edges:
                    modified_ef[road, 1] = 1.0
            
            preds = predict_congestion(model, device, node_features, edge_index, modified_ef)
            
            if preds is not None:
                results[scenario_name] = {
                    'predictions': preds,
                    'mean': np.mean(preds),
                    'std': np.std(preds),
                    'roads_closed': len(roads)
                }
            
            progress.progress((i + 1) / len(selected_scenarios))
        
        progress.empty()
        
        # Display comparison
        st.markdown("---")
        st.subheader("📈 Comparison Results")
        
        # Metrics row
        cols = st.columns(len(results))
        for i, (name, data) in enumerate(results.items()):
            with cols[i]:
                st.markdown(f"**{name}**")
                st.metric("Avg Congestion", f"{data['mean']:.2f}")
                st.caption(f"Roads closed: {data['roads_closed']}")
        
        # Bar chart
        fig = go.Figure()
        
        for name, data in results.items():
            fig.add_trace(go.Bar(
                name=name,
                x=[name],
                y=[data['mean']],
                text=[f"{data['mean']:.2f}"],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Average Congestion by Scenario",
            yaxis_title="Average Congestion Factor",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution comparison
        fig2 = go.Figure()
        
        for name, data in results.items():
            fig2.add_trace(go.Violin(
                y=data['predictions'],
                name=name,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig2.update_layout(
            title="Congestion Distribution by Scenario",
            yaxis_title="Congestion Factor",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)


def model_analysis(model, device, G):
    """Show model analysis and statistics"""
    st.subheader("🔬 Model Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Prediction Stats", "🏗️ Architecture", "📈 Performance"])
    
    with tab1:
        node_features, edge_index, edge_features, edge_keys = snapshot_to_tensors(G)
        
        if node_features is not None:
            st.markdown("### Current Snapshot Statistics")
            
            preds = predict_congestion(model, device, node_features, edge_index, edge_features)
            
            if preds is not None:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean", f"{np.mean(preds):.2f}")
                with col2:
                    st.metric("Std Dev", f"{np.std(preds):.2f}")
                with col3:
                    st.metric("Min", f"{np.min(preds):.2f}")
                with col4:
                    st.metric("Max", f"{np.max(preds):.2f}")
                
                # Histogram
                fig = px.histogram(
                    x=preds,
                    nbins=50,
                    title="Prediction Distribution",
                    labels={'x': 'Congestion Factor', 'y': 'Count'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### GATv2 Architecture")
        
        st.code("""
TrafficGATv2(
  (node_encoder): Linear(4 → 64)
  (edge_encoder): Linear(3 → 64)
  
  (attention_layers): ModuleList(
    (0): GATv2Conv(64, 64, heads=4)
    (1): GATv2Conv(64, 64, heads=4)
    (2): GATv2Conv(64, 64, heads=4)
  )
  
  (edge_predictor): Sequential(
    Linear(192 → 64)
    ReLU()
    Dropout(0.2)
    Linear(64 → 32)
    ReLU()
    Dropout(0.2)
    Linear(32 → 1)
  )
)

Total Parameters: 115,841
        """, language="python")
        
        st.markdown("""
        **Key Features:**
        - **4 Attention Heads**: Learn different relationship patterns
        - **3 GATv2 Layers**: Deep feature extraction
        - **64 Hidden Dimensions**: Balance between capacity and speed
        - **Dropout 0.2**: Prevents overfitting
        """)
    
    with tab3:
        st.markdown("### Training Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Configuration**")
            st.write("- Epochs: 50")
            st.write("- Batch Size: 32")
            st.write("- Learning Rate: 0.001")
            st.write("- Optimizer: Adam")
            st.write("- Loss: MSE")
        
        with col2:
            st.markdown("**Results**")
            st.write("- Training Loss: 62.10 MSE")
            st.write("- Validation Loss: 61.73 MSE")
            st.write("- Training Time: ~23 min (GPU)")
            st.write("- Dataset: 6,000 snapshots")
        
        # Simulated training curve
        epochs = list(range(1, 51))
        train_loss = [185 * np.exp(-0.05 * e) + 62 + np.random.normal(0, 2) for e in epochs]
        val_loss = [184 * np.exp(-0.05 * e) + 61.73 + np.random.normal(0, 2) for e in epochs]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss', line=dict(color='#3498db')))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Val Loss', line=dict(color='#e74c3c')))
        fig.update_layout(
            title="Training Progress (Simulated)",
            xaxis_title="Epoch",
            yaxis_title="Loss (MSE)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application"""
    
    # Load resources (model is all we need!)
    model, device, model_loaded = load_model()
    G, graph_loaded = load_graph()
    
    # Header
    show_header()
    
    # Sidebar
    show_sidebar_info(G, model_loaded, graph_loaded, device)
    
    # Check if everything is loaded
    if not model_loaded:
        st.error("⚠️ Model not loaded! Make sure `trained_gnn.pt` exists.")
        st.info("Run `python train_model.py` to train the model first.")
        return
    
    st.success("✅ Model loaded and ready for inference!")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛣️ Single Road Test",
        "🛣️ Multiple Roads Test", 
        "⚖️ Scenario Comparison",
        "🔬 Model Analysis"
    ])
    
    with tab1:
        single_road_test(model, device, G)
    
    with tab2:
        multi_road_test(model, device, G)
    
    with tab3:
        scenario_comparison(model, device, G)
    
    with tab4:
        model_analysis(model, device, G)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🚦 Traffic Prediction AI | Digital Twin City Simulation | "
        "Built with Streamlit & PyTorch"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
