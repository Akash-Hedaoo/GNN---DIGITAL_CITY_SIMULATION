import streamlit as st
import os
import logging

# Ensure logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure Dashboard Logging
logging.basicConfig(
    filename=os.path.join("logs", "dashboard.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Page Setup
st.set_page_config(layout="wide", page_title="Digital Twin System")

# Navigation
pg = st.navigation([
    st.Page("app_pages/1_Live_Map.py", title="Live Map", icon="🗺️"),
    st.Page("app_pages/2_Analytics_Dashboard.py", title="Analytics Dashboard", icon="📊"),
])

st.sidebar.title("Navigation")
st.sidebar.info("Select a module above.")

pg.run()
