# 🚦 Digital Twin City Simulation - Streamlit GUI

## Overview
Professional dark-themed web interface for GNN-based traffic prediction system. Features real-time traffic analysis, road closure simulations, and interactive city network visualization.

## Features
- 🗺️ **Interactive City Map** - Visualize the entire road network
- 📊 **Real-time Metrics** - Track travel time, energy consumption, and network stability
- 🛣️ **Road Closure Analysis** - Test single or multiple road closures
- ⚖️ **Scenario Comparison** - Compare different traffic scenarios
- 🔬 **Model Analysis** - Deep dive into GNN model performance
- 🎨 **Dark Theme UI** - Professional interface matching modern design standards

## Requirements
- Python 3.8+
- PyTorch
- Streamlit
- NetworkX
- Plotly
- NumPy

## Installation

1. **Activate virtual environment:**
   ```bash
   # Windows PowerShell
   .\twin-city-env\Scripts\Activate.ps1
   
   # Windows CMD
   .\twin-city-env\Scripts\activate.bat
   
   # Linux/Mac
   source twin-city-env/bin/activate
   ```

2. **Install dependencies (if not already installed):**
   ```bash
   pip install streamlit plotly networkx torch numpy pandas
   ```

## Required Files
Make sure these files exist in the directory:
- ✅ `trained_gnn.pt` - Trained GNN model
- ✅ `city_graph.graphml` - City road network
- ✅ `gnn_training_data.pkl` - Training data (optional, enables more features)
- ✅ `gnn_model.py` - Model architecture definitions

## Running the Application

### Method 1: Direct Run
```bash
streamlit run streamlit_gui.py
```

### Method 2: With Custom Port
```bash
streamlit run streamlit_gui.py --server.port 8502
```

### Method 3: With Full Options
```bash
streamlit run streamlit_gui.py --server.port 8501 --server.headless false
```

## Usage Guide

### 1. Sidebar Controls
- **Run Simulation** - Start the traffic simulation
- **Search** - Find specific nodes or areas
- **Simulation Settings** - Adjust speed and toggle real-time mode
- **Node/Edge Management** - Add or remove network elements
- **Visualization Layers** - Toggle different map overlays

### 2. Main Tabs

#### 🗺️ Map View
- Interactive visualization of city network
- Zoom and pan controls
- Node and edge information on hover

#### 📊 Analytics
- **Single Road Test**: Close one road and see impact
- Real-time congestion prediction
- Distribution analysis

#### 🧪 Experiments
- **Multiple Roads Test**: Test combined closures
- Manual, range, or random selection
- Impact visualization

### 3. Right Panel

#### Metrics Tab
- Average Travel Time
- Energy Consumption
- Network Stability
- Real-time charts

#### Inspector Tab
- Node information lookup
- Network statistics
- Graph properties

#### Logs Tab
- System status messages
- Operation history

### 4. Advanced Tools (Expandable)
- **Scenario Comparison**: Compare multiple traffic scenarios
- **Model Analysis**: View GNN architecture and training stats

## Customization

### Changing Theme Colors
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor="#2196F3"
backgroundColor="#0f1419"
secondaryBackgroundColor="#1a2332"
textColor="#ffffff"
```

### Modifying Metrics
Edit `create_metrics_panel()` function in `streamlit_gui.py`

### Adding New Visualizations
Add functions similar to `create_network_stability_chart()`

## Troubleshooting

### Issue: "Model not loaded"
**Solution**: Ensure `trained_gnn.pt` exists. Run `python train_model.py` first.

### Issue: "Graph not loaded"
**Solution**: Check that `city_graph.graphml` is in the current directory.

### Issue: Port already in use
**Solution**: Change port using `streamlit run streamlit_gui.py --server.port 8502`

### Issue: Import errors
**Solution**: Activate virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Streamlit cache errors
**Solution**: Clear cache:
```bash
streamlit cache clear
```

## Performance Tips

1. **Large Graphs**: The application automatically caches graph and model loading
2. **Memory**: Close other applications if handling very large networks
3. **Speed**: Use GPU if available for faster predictions (automatically detected)

## Development

### File Structure
```
GNN_DIGITAL_CITY_SIMULATION/
├── streamlit_gui.py          # Main GUI application
├── gnn_model.py              # Model definitions
├── city_graph.graphml        # City network data
├── trained_gnn.pt            # Trained model weights
├── gnn_training_data.pkl     # Training data
├── .streamlit/
│   └── config.toml          # Theme configuration
└── twin-city-env/           # Virtual environment
```

### Adding New Features
1. Create new function in `streamlit_gui.py`
2. Add to appropriate tab in `main()` function
3. Test with sample data

## Keyboard Shortcuts
- `Ctrl + R` - Rerun the application
- `C` - Clear cache
- `M` - Toggle sidebar

## Browser Compatibility
- ✅ Chrome (Recommended)
- ✅ Firefox
- ✅ Edge
- ⚠️ Safari (Limited support)

## Credits
- **Framework**: Streamlit
- **Visualization**: Plotly
- **ML Framework**: PyTorch
- **Graph Library**: NetworkX

## Support
For issues or questions:
1. Check `PENDING_WORK.md` for known issues
2. Review error messages in the Logs tab
3. Ensure all required files are present

## Version
**v2.0** - Dark theme redesign with enhanced UX

---
Built with ❤️ for Digital Twin City Simulation
