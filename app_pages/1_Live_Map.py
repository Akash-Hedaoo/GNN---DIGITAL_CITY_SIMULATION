import streamlit as st
import streamlit.components.v1 as components
import folium
import json
import os
import sys
import time
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_connector import SumoManager

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Live Traffic Map", layout="wide")

st.title("🗺️ Live Traffic Map - Pimpri-Chinchwad")

# --- Constants ---
GEOJSON_FILE = os.path.join("outputs", "road_network.geojson")

# --- State Management ---
if 'sumo_manager' not in st.session_state:
    st.session_state.sumo_manager = SumoManager()
    st.session_state.simulation_active = False

if 'map_initialized' not in st.session_state:
    st.session_state.map_initialized = False

mgr = st.session_state.sumo_manager

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Simulation Control")
    
    if st.button("Start Simulation"):
        if not mgr.running:
            try:
                mgr.start(gui=False)
                st.session_state.simulation_active = True
                st.success("Simulation Started!")
                logging.info("Simulation started from Live Map")
            except Exception as e:
                st.error(f"Failed to start: {e}")
        else:
            st.warning("Simulation already running.")
            
    if st.button("Stop Simulation"):
        if mgr.running:
            mgr.close()
            st.session_state.simulation_active = False
            st.success("Simulation Stopped.")

    st.markdown("---")
    st.markdown("**Status**")
    st.write(f"Active: {mgr.running}")

# --- Main Map Logic ---

# Initialize map HTML if not done
if not st.session_state.map_initialized:
    logging.info("Initializing map with GeoJSON...")
    
    # Create Folium map centered on Pune
    m = folium.Map(
        location=[18.6298, 73.7997],
        zoom_start=13,
        tiles="CartoDB dark_matter"
    )
    
    # Load GeoJSON
    with open(GEOJSON_FILE, 'r') as f:
        geojson_data = json.load(f)
    
    # Add GeoJSON layer with custom styling
    folium.GeoJson(
        geojson_data,
        name='Road Network',
        style_function=lambda feature: {
            'color': '#00FF00',  # Default green
            'weight': 2,
            'opacity': 0.8
        },
        tooltip=folium.GeoJsonTooltip(fields=['edge_id'], aliases=['Road ID:']),
        popup=folium.GeoJsonPopup(fields=['edge_id'])
    ).add_to(m)
    
    # Get map HTML
    map_html = m._repr_html_()
    
    # Inject custom JavaScript for color updates
    custom_js = """
    <script>
    // Wait for map to load
    window.addEventListener('load', function() {
        // Store reference to map layers
        window.edgeLayers = {};
        
        // Find all GeoJSON layers and build edge_id → layer mapping
        setTimeout(function() {
            // Access Leaflet map instance
            const mapElement = document.querySelector('.folium-map');
            if (!mapElement || !mapElement._leaflet_id) return;
            
            const map = window[mapElement._leaflet_id];
            if (!map) return;
            
            map.eachLayer(function(layer) {
                if (layer.feature && layer.feature.properties && layer.feature.properties.edge_id) {
                    window.edgeLayers[layer.feature.properties.edge_id] = layer;
                }
            });
            
            console.log('Initialized', Object.keys(window.edgeLayers).length, 'road layers');
        }, 1000);
        
        // Function to update traffic colors
        window.updateTrafficColors = function(colorMap) {
            let updated = 0;
            for (const [edge_id, color] of Object.entries(colorMap)) {
                const layer = window.edgeLayers[edge_id];
                if (layer) {
                    layer.setStyle({
                        color: color,
                        weight: 3,
                        opacity: 0.9
                    });
                    updated++;
                }
            }
            console.log('Updated', updated, 'road colors');
        };
    });
    </script>
    """
    
    # Combine map HTML with custom JS
    full_html = map_html + custom_js
    
    st.session_state.map_html = full_html
    st.session_state.map_initialized = True
    logging.info(f"Map initialized with {len(geojson_data['features'])} roads")

# Display map using components
map_container = st.container()
with map_container:
    components.html(st.session_state.map_html, height=700, scrolling=False)

# --- Color Update Loop ---
if st.session_state.simulation_active:
    # Step simulation
    mgr.step()
    
    # Get live colors
    color_map = mgr.get_live_state()
    
    # Inject JavaScript call to update colors
    # Note: This approach has limitations in Streamlit
    # A better approach would use a WebSocket or server-sent events
    # For this demo, we'll serialize and inject via another iframe
    
    update_js = f"""
    <script>
    if (parent.window.updateTrafficColors) {{
        parent.window.updateTrafficColors({json.dumps(color_map)});
    }}
    </script>
    """
    
    components.html(update_js, height=0)
    
    # Wait and rerun
    time.sleep(2)
    st.rerun()
