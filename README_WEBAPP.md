# GNN Traffic Simulation Web Application

Interactive web application for visualizing and analyzing traffic patterns using Graph Neural Networks.

## Features

- 🗺️ **Interactive Map**: Real-time visualization of road network with traffic congestion
- 🖱️ **Road Selection**: Click on roads to select them for closure simulation
- 🤖 **AI-Powered Predictions**: GNN model predicts traffic impact of road closures
- 📊 **Results Dashboard**: Detailed metrics and visualization of simulation results
- 🎨 **Color-Coded Traffic**: Visual representation of congestion levels

## Quick Start

### 1. Activate Virtual Environment

```powershell
# In PowerShell
. .\activate_venv.ps1

# Or use direct Python
.venv\Scripts\python.exe run_app.py
```

### 2. Start the Server

```powershell
python run_app.py
```

Or directly:
```powershell
python app.py
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

## Usage

1. **View Baseline Traffic**: The map loads with current traffic predictions
2. **Select Roads**: Click on any road segment to select it for closure
3. **Run Simulation**: Click "Run Simulation" to analyze the impact
4. **View Results**: Results page shows detailed metrics and impacted segments

## API Endpoints

- `GET /` - Main interactive map page
- `GET /results` - Results visualization page
- `GET /api/graph-data` - Get graph nodes, edges, and baseline predictions
- `POST /api/simulate-closure` - Simulate road closures and get predictions

## Project Structure

```
GNN---DIGITAL_CITY_SIMULATION/
├── app.py                 # Flask backend
├── run_app.py            # App launcher
├── templates/
│   ├── index.html        # Main map page
│   └── results.html      # Results page
├── static/
│   ├── css/
│   │   ├── style.css     # Main styles
│   │   └── results.css   # Results page styles
│   └── js/
│       ├── main.js       # Map interaction logic
│       └── results.js    # Results visualization
└── step4_train_model.py  # GNN model definition
```

## Requirements

All dependencies are installed in `.venv`. Key packages:
- Flask & Flask-CORS
- PyTorch & PyTorch Geometric
- NetworkX & OSMnx
- Leaflet (loaded from CDN)

## Troubleshooting

### Port Already in Use
If port 5000 is busy, edit `app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Graph Not Loading
Ensure `real_city_processed.graphml` exists in the project directory.

### Model Not Found
Ensure `real_city_gnn.pt` exists. If not, run the training pipeline:
```powershell
python step4_train_model.py
```

## Notes

- The map uses OpenStreetMap tiles
- Road segments are color-coded by congestion level
- Selected roads are highlighted in red
- Results are stored in browser session storage

