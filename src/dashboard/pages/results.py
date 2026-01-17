"""
Traffic Digital Twin Dashboard - Results Page
==============================================
Analysis and interpretation of simulation results.

Features:
- Occupancy vs Time graphs
- Prediction error visualization
- Congestion heatmaps
- Edge-level analysis

Author: Traffic Digital Twin Project
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "simulations")
GEOJSON_PATH = os.path.join(PROJECT_ROOT, "outputs", "maps", "pune_edges.geojson")
MAP_CENTER_PATH = os.path.join(PROJECT_ROOT, "outputs", "maps", "map_center.json")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "failure_propagation_log.csv")
BLOCKED_PATH = os.path.join(OUTPUT_DIR, "blocked_edges.json")

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Results - Traffic Digital Twin",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data
def load_results():
    """Load simulation results."""
    if not os.path.exists(RESULTS_PATH):
        return None
    return pd.read_csv(RESULTS_PATH)


@st.cache_data
def load_blocked_edges():
    """Load blocked edges list."""
    if os.path.exists(BLOCKED_PATH):
        with open(BLOCKED_PATH, 'r') as f:
            return json.load(f)
    return []


@st.cache_resource
def load_geojson():
    """Load the preprocessed GeoJSON network."""
    if not os.path.exists(GEOJSON_PATH):
        return None
    with open(GEOJSON_PATH, 'r') as f:
        return json.load(f)


@st.cache_resource
def get_map_center():
    """Get the map center coordinates."""
    if os.path.exists(MAP_CENTER_PATH):
        with open(MAP_CENTER_PATH, 'r') as f:
            center = json.load(f)
            return center['lat'], center['lon']
    return 18.612095, 73.818654

# =============================================================================
# Visualization Functions
# =============================================================================

def create_occupancy_graph(df, edge_id=None, metric='actual'):
    """Create occupancy over time graph."""
    
    y_col = 'actual_occupancy' if metric == 'actual' else 'predicted_occupancy'
    title_metric = 'Actual' if metric == 'actual' else 'Predicted'
    
    if edge_id:
        edge_df = df[df['edge_id'] == edge_id]
        title = f"{title_metric} Occupancy - {edge_id}"
    else:
        edge_df = df.groupby('timestep')[y_col].mean().reset_index()
        title = f"Mean {title_metric} Occupancy (All Edges)"
    
    fig = px.line(
        edge_df,
        x='timestep',
        y=y_col,
        title=title,
        labels={'timestep': 'Time Step', y_col: 'Occupancy'}
    )
    
    fig.add_vline(x=500, line_dash="dash", line_color="red",
                  annotation_text="Blockage Start")
    fig.update_layout(height=400)
    return fig


def create_error_graph(df, edge_id=None):
    """Create prediction error over time graph."""
    
    df_clean = df.dropna(subset=['predicted_occupancy']).copy()
    df_clean['error'] = df_clean['actual_occupancy'] - df_clean['predicted_occupancy']
    
    if edge_id:
        plot_df = df_clean[df_clean['edge_id'] == edge_id]
        title = f"Prediction Error - {edge_id}"
    else:
        plot_df = df_clean.groupby('timestep')['error'].mean().reset_index()
        title = "Mean Prediction Error (All Edges)"
    
    fig = px.line(
        plot_df,
        x='timestep',
        y='error',
        title=title,
        labels={'timestep': 'Time Step', 'error': 'Error'}
    )
    
    fig.add_vline(x=500, line_dash="dash", line_color="red",
                  annotation_text="Blockage Start")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(height=400)
    return fig


def create_comparison_graph(df, edge_id=None):
    """Create actual vs predicted comparison graph."""
    
    df_clean = df.dropna(subset=['predicted_occupancy'])
    
    if edge_id:
        plot_df = df_clean[df_clean['edge_id'] == edge_id]
        title = f"Actual vs Predicted - {edge_id}"
    else:
        plot_df = df_clean.groupby('timestep').agg({
            'actual_occupancy': 'mean',
            'predicted_occupancy': 'mean'
        }).reset_index()
        title = "Actual vs Predicted Occupancy (Mean)"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df['timestep'],
        y=plot_df['actual_occupancy'],
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=plot_df['timestep'],
        y=plot_df['predicted_occupancy'],
        mode='lines',
        name='Predicted',
        line=dict(color='orange', dash='dash')
    ))
    
    fig.add_vline(x=500, line_dash="dash", line_color="red",
                  annotation_text="Blockage Start")
    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Occupancy",
        height=400
    )
    return fig


def create_heatmap(df, metric='actual', timestep=None):
    """Create Folium heatmap of congestion intensity."""
    
    geojson = load_geojson()
    center_lat, center_lon = get_map_center()
    
    if geojson is None:
        return None
    
    # Create edge center lookup
    edge_centers = {}
    for feature in geojson['features']:
        edge_id = feature['properties']['edge_id']
        coords = feature['geometry']['coordinates']
        if coords:
            mid_idx = len(coords) // 2
            edge_centers[edge_id] = [coords[mid_idx][1], coords[mid_idx][0]]  # [lat, lon]
    
    # Get data for specified timestep
    if timestep is None:
        timestep = int(df['timestep'].max())
    
    step_df = df[df['timestep'] == timestep]
    occ_col = 'actual_occupancy' if metric == 'actual' else 'predicted_occupancy'
    
    # Prepare heatmap data
    heat_data = []
    for _, row in step_df.iterrows():
        edge_id = row['edge_id']
        occ = row[occ_col]
        
        if pd.isna(occ) or edge_id not in edge_centers:
            continue
        
        center = edge_centers[edge_id]
        weight = float(occ) * 100
        
        if weight > 0:
            heat_data.append([center[0], center[1], weight])
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='CartoDB dark_matter'
    )
    
    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.3,
            radius=15,
            blur=10,
            max_zoom=18
        ).add_to(m)
    
    # Add blocked edge markers
    blocked_edges = load_blocked_edges()
    for edge_id in blocked_edges:
        if edge_id in edge_centers:
            center = edge_centers[edge_id]
            folium.CircleMarker(
                location=center,
                radius=8,
                color='black',
                fill=True,
                fillColor='black',
                fillOpacity=1,
                popup=f"Blocked: {edge_id}"
            ).add_to(m)
    
    return m

# =============================================================================
# Main Page
# =============================================================================

def main():
    """Main results page."""
    
    st.title("📊 Analysis Results")
    st.markdown("**Congestion Propagation Analysis**")
    
    # Load data
    df = load_results()
    
    if df is None:
        st.warning("⚠️ No analysis results found. Run an analysis from the Live Map first.")
        st.page_link("app.py", label="🗺️ Go to Live Map", icon="🚗")
        return
    
    # Sidebar
    st.sidebar.title("📊 Analysis Controls")
    st.sidebar.markdown("---")
    
    edges_list = sorted(df['edge_id'].unique().tolist())
    selected_edge = st.sidebar.selectbox(
        "Select Edge:",
        options=["All Edges"] + edges_list
    )
    edge_filter = None if selected_edge == "All Edges" else selected_edge
    
    st.sidebar.markdown("---")
    
    metric = st.sidebar.radio("Heatmap Metric:", ["Actual", "Predicted"], horizontal=True)
    metric_key = 'actual' if metric == "Actual" else 'predicted'
    
    max_step = int(df['timestep'].max())
    timestep = st.sidebar.slider("Timestep:", 0, max_step, max_step, step=10)
    
    st.sidebar.markdown("---")
    st.sidebar.page_link("app.py", label="🗺️ Back to Live Map", icon="🚗")
    
    # Show blocked edges
    blocked_edges = load_blocked_edges()
    if blocked_edges:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🚧 Blocked Edges")
        for e in blocked_edges:
            st.sidebar.text(f"• {e}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🔥 Heatmap", "📊 Comparison", "📋 Data"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_occupancy_graph(df, edge_filter, 'actual'), use_container_width=True)
        with col2:
            st.plotly_chart(create_occupancy_graph(df, edge_filter, 'predicted'), use_container_width=True)
        
        st.plotly_chart(create_error_graph(df, edge_filter), use_container_width=True)
    
    with tab2:
        st.subheader(f"Congestion Heatmap - {metric}")
        st.caption(f"Timestep: {timestep}")
        
        heatmap = create_heatmap(df, metric_key, timestep)
        if heatmap:
            st_folium(heatmap, width=1200, height=600)
        else:
            st.error("Cannot create heatmap: GeoJSON not found")
    
    with tab3:
        st.plotly_chart(create_comparison_graph(df, edge_filter), use_container_width=True)
        
        df_clean = df.dropna(subset=['predicted_occupancy'])
        col1, col2, col3 = st.columns(3)
        with col1:
            mae = np.mean(np.abs(df_clean['actual_occupancy'] - df_clean['predicted_occupancy']))
            st.metric("MAE", f"{mae:.4f}")
        with col2:
            rmse = np.sqrt(np.mean((df_clean['actual_occupancy'] - df_clean['predicted_occupancy'])**2))
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("Total Edges", len(edges_list))
    
    with tab4:
        display_df = df.copy()
        if edge_filter:
            display_df = display_df[display_df['edge_id'] == edge_filter]
        
        st.dataframe(display_df.head(1000), use_container_width=True)
        
        csv = display_df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")


if __name__ == "__main__":
    main()
