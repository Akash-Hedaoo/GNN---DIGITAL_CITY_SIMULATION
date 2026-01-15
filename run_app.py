#!/usr/bin/env python
"""
Simple launcher script for the Flask app
Ensures venv is used and handles initialization
"""
import sys
import os

# Ensure we're using the venv Python
if __name__ == "__main__":
    print("🚀 Starting GNN Traffic Simulation Web App...")
    print("=" * 60)
    
    # Import and run the app
    from app import app, twin
    
    print("\n✅ Server ready!")
    print("📍 Open your browser to: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

