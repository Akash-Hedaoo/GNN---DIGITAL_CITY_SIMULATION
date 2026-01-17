"""
Traffic Digital Twin Dashboard - True Digital Twin Map
=======================================================
High-performance traffic control with clickable road network overlay.

Features:
- SUMO road network overlaid on OpenStreetMap
- Click roads directly to block/unblock
- Real-time visual feedback
- Backend analysis triggering

Author: Traffic Digital Twin Project
"""

import os
import sys
import json
import subprocess
import streamlit as st
import folium
from folium import GeoJson
from streamlit_folium import st_folium

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEOJSON_PATH = os.path.join(PROJECT_ROOT, "outputs", "maps", "pune_edges.geojson")
MAP_CENTER_PATH = os.path.join(PROJECT_ROOT, "outputs", "maps", "map_center.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "simulations")

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Traffic Digital Twin",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Session State Initialization
# =============================================================================

if 'blocked_edges' not in st.session_state:
    st.session_state.blocked_edges = set()

if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# =============================================================================
# Cached Resources
# =============================================================================

@st.cache_resource
def load_geojson():
    """Load the preprocessed GeoJSON network."""
    if not os.path.exists(GEOJSON_PATH):
        st.error(f"GeoJSON not found. Run prepare_geojson.py first!")
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
    # Default: Nashik Phata area
    return 18.612095, 73.818654


@st.cache_data
def get_edge_list():
    """Get list of all edge IDs."""
    geojson = load_geojson()
    if geojson:
        return sorted([f['properties']['edge_id'] for f in geojson['features']])
    return []

# =============================================================================
# Map Styling
# =============================================================================

def get_road_style(blocked_edges):
    """Create style function for road coloring."""
    
    def style_function(feature):
        edge_id = feature['properties'].get('edge_id', '')
        
        if edge_id in blocked_edges:
            # Blocked road - black and thick
            return {
                'color': '#000000',
                'weight': 6,
                'opacity': 1.0
            }
        else:
            # Normal road - green
            return {
                'color': '#22AA22',
                'weight': 3,
                'opacity': 0.8
            }
    
    return style_function


def get_highlight_style():
    """Style for highlighted (hovered) roads."""
    return {
        'color': '#0066FF',
        'weight': 6,
        'opacity': 1.0
    }

# =============================================================================
# Map Creation
# =============================================================================

def create_traffic_map(blocked_edges):
    """Create Folium map with clickable road network."""
    
    geojson = load_geojson()
    center_lat, center_lon = get_map_center()
    
    if geojson is None:
        st.error("Cannot create map: GeoJSON not loaded")
        return None
    
    # Create base map with OpenStreetMap tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Create separate layers for blocked and normal roads
    blocked_features = []
    normal_features = []
    
    for feature in geojson['features']:
        edge_id = feature['properties'].get('edge_id', '')
        if edge_id in blocked_edges:
            blocked_features.append(feature)
        else:
            normal_features.append(feature)
    
    # Add normal roads layer
    if normal_features:
        normal_geojson = {
            "type": "FeatureCollection",
            "features": normal_features
        }
        
        GeoJson(
            normal_geojson,
            name="Normal Roads",
            style_function=lambda x: {
                'color': '#22AA22',
                'weight': 3,
                'opacity': 0.8
            },
            highlight_function=lambda x: {
                'color': '#0066FF',
                'weight': 6,
                'opacity': 1.0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['edge_id'],
                aliases=['Edge ID:'],
                style='font-size: 12px;'
            ),
            popup=folium.GeoJsonPopup(
                fields=['edge_id'],
                aliases=['Edge ID:'],
                style='font-size: 14px; font-weight: bold;'
            )
        ).add_to(m)
    
    # Add blocked roads layer (on top)
    if blocked_features:
        blocked_geojson = {
            "type": "FeatureCollection",
            "features": blocked_features
        }
        
        GeoJson(
            blocked_geojson,
            name="Blocked Roads",
            style_function=lambda x: {
                'color': '#000000',
                'weight': 6,
                'opacity': 1.0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['edge_id'],
                aliases=['🚧 BLOCKED:'],
                style='font-size: 12px; color: red;'
            ),
            popup=folium.GeoJsonPopup(
                fields=['edge_id'],
                aliases=['🚧 Blocked Edge:'],
                style='font-size: 14px; font-weight: bold; color: red;'
            )
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# =============================================================================
# Sidebar Controls
# =============================================================================

def render_sidebar():
    """Render sidebar with controls."""
    
    st.sidebar.title("🚗 Traffic Control")
    st.sidebar.markdown("---")
    
    # Instructions
    st.sidebar.info("👆 Click on a road on the map to see its Edge ID, then use the controls below to block it.")
    
    # Edge selector for blocking
    st.sidebar.subheader("Select Edge to Block")
    edge_list = get_edge_list()
    
    selected_edge = st.sidebar.selectbox(
        "Choose an edge:",
        options=[""] + edge_list,
        key="edge_selector",
        help="Select an edge ID to block/unblock"
    )
    
    if selected_edge:
        col1, col2 = st.sidebar.columns(2)
        
        is_blocked = selected_edge in st.session_state.blocked_edges
        
        with col1:
            if not is_blocked:
                if st.button("🚧 Block", type="primary", use_container_width=True):
                    st.session_state.blocked_edges.add(selected_edge)
                    st.rerun()
        with col2:
            if is_blocked:
                if st.button("✅ Unblock", type="secondary", use_container_width=True):
                    st.session_state.blocked_edges.discard(selected_edge)
                    st.rerun()
        
        if is_blocked:
            st.sidebar.success(f"✅ {selected_edge} is currently BLOCKED")
        else:
            st.sidebar.info(f"ℹ️ {selected_edge} is not blocked")
    
    st.sidebar.markdown("---")
    
    # Blocked edges list
    st.sidebar.subheader(f"🚧 Blocked Edges ({len(st.session_state.blocked_edges)})")
    
    if st.session_state.blocked_edges:
        for edge in sorted(st.session_state.blocked_edges):
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                display_name = edge[:20] + "..." if len(edge) > 20 else edge
                st.sidebar.text(f"• {display_name}")
            with col2:
                if st.sidebar.button("❌", key=f"rm_{edge}", help=f"Unblock {edge}"):
                    st.session_state.blocked_edges.discard(edge)
                    st.rerun()
    else:
        st.sidebar.caption("No roads blocked")
    
    st.sidebar.markdown("---")
    
    # Action buttons
    if st.sidebar.button("🗑️ Clear All Blockages", use_container_width=True):
        st.session_state.blocked_edges = set()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Run Analysis
    if st.sidebar.button("🔬 Run Analysis", type="primary", use_container_width=True):
        if st.session_state.blocked_edges:
            run_analysis()
        else:
            st.sidebar.warning("⚠️ Block at least one road first!")
    
    # Results link
    if os.path.exists(os.path.join(OUTPUT_DIR, "failure_propagation_log.csv")):
        st.sidebar.markdown("---")
        st.sidebar.success("✅ Analysis results available!")
        st.sidebar.page_link("pages/results.py", label="📊 View Results", icon="📈")

# =============================================================================
# Analysis Runner
# =============================================================================

def run_analysis():
    """Trigger backend analysis as subprocess."""
    
    blocked_edges_str = ",".join(st.session_state.blocked_edges)
    
    # Save blocked edges for results page
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata_path = os.path.join(OUTPUT_DIR, "blocked_edges.json")
    with open(metadata_path, 'w') as f:
        json.dump(list(st.session_state.blocked_edges), f)
    
    st.session_state.analysis_running = True
    
    with st.spinner("🔄 Running congestion simulation... This may take several minutes."):
        try:
            backend_path = os.path.join(PROJECT_ROOT, "src", "dashboard", "backend.py")
            
            result = subprocess.run(
                [
                    sys.executable, backend_path,
                    "--blocked-edges", blocked_edges_str,
                    "--output", "failure_propagation_log.csv"
                ],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                st.success("✅ Analysis complete! Navigate to Results page.")
                st.session_state.analysis_running = False
            else:
                st.error(f"❌ Analysis failed: {result.stderr[:500]}")
                st.session_state.analysis_running = False
                
        except subprocess.TimeoutExpired:
            st.error("❌ Analysis timed out after 10 minutes")
            st.session_state.analysis_running = False
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.analysis_running = False

# =============================================================================
# Main Page
# =============================================================================

def main():
    """Main landing page."""
    
    # Header
    st.title("🚗 Traffic Digital Twin")
    st.markdown("**Live Traffic Control — Nashik Phata Study Area (3.5 km)**")
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    col1, col2 = st.columns([5, 1])
    
    with col1:
        st.subheader("🗺️ Road Network Map")
        st.caption("Click on any road to see its Edge ID. Use the sidebar to block/unblock roads.")
        
        # Create and display map
        traffic_map = create_traffic_map(st.session_state.blocked_edges)
        
        if traffic_map:
            # Render map and capture interactions
            map_data = st_folium(
                traffic_map,
                width=1000,
                height=650,
                returned_objects=["last_object_clicked", "last_active_drawing"]
            )
            
            # Display clicked info
            if map_data and map_data.get("last_object_clicked"):
                clicked = map_data["last_object_clicked"]
                st.info(f"📍 Clicked: {clicked}")
    
    with col2:
        st.subheader("Status")
        st.metric("Blocked Roads", len(st.session_state.blocked_edges))
        
        if st.session_state.analysis_running:
            st.warning("⏳ Running...")
        else:
            st.success("✅ Ready")
        
        # Edge count
        edges = get_edge_list()
        st.metric("Total Edges", len(edges))
    
    # Footer
    st.markdown("---")
    st.caption("🚧 Black roads = Blocked | 🟢 Green roads = Normal")
    
    # Print ready message
    print("TRUE DIGITAL TWIN MAP READY")


if __name__ == "__main__":
    main()
