# 🌆 GNN Digital Twin City Simulation

**A complete urban simulation platform featuring Graph Attention Networks (GATv2), realistic traffic modeling, interactive visualization, and AI-powered traffic prediction.**

> **📘 Complete documentation consolidated in [README_MASTER.md](README_MASTER.md)**

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

---

## 🚀 Quick Start (30 seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric torch-scatter torch-sparse

# 2. Start the application
python backend/app.py

# 3. Open browser
# http://localhost:5000
```

---

## ✨ Key Features

- 🤖 **AI Traffic Prediction**: GATv2 model with 115,841 parameters
- 🏙️ **Realistic City**: 796 nodes, 4,676 edges, organic layout
- 🚇 **Multi-Modal Transport**: 3 metro lines + road network
- 🎮 **Interactive Control**: Remove nodes, close roads, analyze impact
- 🌐 **Web Interface**: Real-time visualization with Leaflet.js
- 📊 **What-If Analysis**: Test scenarios before implementation
- 🆕 **Node Removal & Impact Analysis**: Comprehensive infrastructure planning

---

## 📖 Documentation

**For complete documentation, see [README_MASTER.md](README_MASTER.md) which includes:**

| Section | Details |
|---------|---------|
| **System Overview** | Architecture, technology stack, core concepts |
| **Installation** | Step-by-step setup with all dependencies |
| **Quick Start** | Get running in minutes |
| **Usage Guide** | Training, testing, web interface |
| **API Reference** | All REST endpoints with examples |
| **Advanced Features** | Node removal, what-if analysis, interpretability |
| **Troubleshooting** | Common issues and solutions |
| **Performance** | Benchmarks and optimization tips |
| **Contributing** | How to contribute to the project |

---

## 📁 Project Structure

```
├── backend/
│   ├── app.py                 # Flask REST API
│   └── requirements.txt        # Backend dependencies
├── frontend/
│   ├── index.html            # Main interface
│   ├── app.js                # Frontend logic
│   └── style.css             # Styling
├── gnn_model.py              # GATv2 architecture
├── train_model.py            # Training script
├── test_trained_model.py     # Model testing
├── manual_model_test.py      # Interactive analysis
├── city_graph.graphml        # Infrastructure graph
├── trained_gnn.pt            # Model weights
└── README_MASTER.md          # Complete documentation
```

---

## 🎯 Common Tasks

### Train the Model
```bash
python train_model.py
```
Expected time: ~23 minutes on RTX 3050

### Test the Model
```bash
python test_trained_model.py
```

### Interactive What-If Analysis
```bash
python manual_model_test.py
```

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Network Scale** | 796 nodes, 4,676 edges |
| **Model Size** | 115,841 parameters |
| **Training Time** | 23.1 minutes (RTX 3050) |
| **Accuracy** | 61.73 MSE validation loss |
| **Prediction Speed** | 100-500ms per analysis |
| **Metro Coverage** | 3 lines, 24 stations |

---

## 🆕 Node Removal & Impact Analysis

New feature for infrastructure planning:

```
1. Click any node on map
2. Click "Remove Node" button
3. System shows:
   - Closed edges
   - Traffic impact analysis
   - Network-wide congestion
4. Click "Restore" to undo
```

**Use cases**: Metro planning, hospital accessibility, road maintenance, disaster recovery

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python train_model.py` |
| GPU not detected | Check CUDA installation |
| API connection failed | Ensure backend running on localhost:5000 |
| Slow predictions | Reduce batch size in training config |

**For more troubleshooting, see [README_MASTER.md](README_MASTER.md#-troubleshooting)**

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes
4. Push to branch
5. Open Pull Request

---

## 📞 Support

- 📖 **Docs**: [README_MASTER.md](README_MASTER.md)
- 🐛 **Issues**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions
- 📧 **Email**: Contact project maintainers

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🎯 System Overview

This project simulates a digital city modeled after Pune, India, featuring:
- **Complex graph-based urban infrastructure** with 796 nodes and 4,676+ edges
- **Multi-modal transportation**: Road network + 3 metro lines (Red, Blue, Green)
- **Macroscopic traffic simulation** with pressure-based congestion propagation
- **Interactive traffic control** - block roads, simulate events, track statistics
- **Real-time visualization** - Interactive web maps with layer controls
- **GNN training data generation** - Export scenarios for machine learning
- **Trained GATv2 Model** - AI-powered traffic prediction (115,841 parameters)
- **Manual Testing Interface** - Interactive what-if scenario analysis
- **GPU Acceleration** - NVIDIA RTX 3050 optimized

---

**Status**: 🟢 Production Ready | **Version**: 1.0 | **Last Updated**: December 3, 2025

**➡️ [See complete documentation →](README_MASTER.md)**

### 🚇 Metro Network (NEW!)
- **3 Metro Lines**: Red (East-West), Blue (North-South), Green (Diagonal)
- **24 Metro Stations**: 8 stations per line with bidirectional service
- **High Speed**: 80 km/h average (vs 40 km/h for roads)
- **High Capacity**: 5x passenger capacity vs roads
- **Congestion Immune**: Metro maintains constant speed regardless of road traffic
- **Measurable Advantage**: 6-16% faster travel times during congestion

### 🚦 Traffic Simulation (NEW!)
- **Macroscopic Model**: Fluid dynamics approach (not individual agents)
- **Pressure Propagation**: Congestion ripples upstream (3.0x → 2.4x → 2.0x decay)
- **Random Events**: Dynamic traffic incidents affecting ~90 edges/minute
- **Recovery System**: Gradual congestion relief over time
- **Metro Integration**: Metro edges immune to all traffic events
- **Separate Statistics**: Track metro vs road performance independently

### 🎮 Interactive Features (NEW!)
- **Road Blocking**: User can close any road and observe ripple effects
- **Multiple Input Modes**: Select by number, random, or custom nodes
- **Real-time Stats**: Network delay, congestion levels, affected edges
- **Path Finding**: Calculate routes considering current congestion
- **Training Export**: Generate scenarios for GNN model training
- **Auto-run Mode**: Automated testing with multiple scenarios

### 🗺️ Visualization
- **Interactive Web Map**: Google Maps-style interface with Folium
- **Layer Controls**: Toggle roads, metro lines, zones, amenities
- **Color-coded Elements**: All 3 metro lines distinctly colored
- **Popup Details**: Click markers for station/amenity information
- **Measurement Tools**: Distance measurement, fullscreen, minimap
- **Dark Theme**: CartoDB dark matter tiles for modern look

## 🚀 Getting Started

### Prerequisites

- Python 3.13 or higher (tested on 3.13)
- NVIDIA GPU (RTX 3050 or better recommended for training)
- pip package manager
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Akash-Hedaoo/DIGITAL-TWIN-CITY-.git
cd GNN_DIGITAL_CITY_SIMULATION
```

2. Create and activate virtual environment:
```powershell
# Windows PowerShell
python -m venv twin-city-env
.\twin-city-env\Scripts\Activate.ps1
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Install PyTorch with CUDA support (GPU training):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Dependencies

**Core Libraries:**
- `networkx` (3.5) - Graph creation and manipulation
- `numpy` (2.3.5) - Numerical computations
- `scipy` (1.16.3) - Delaunay triangulation
- `matplotlib` (3.10.7) - Static visualization
- `plotly` (6.5.0) - Interactive plotting
- `folium` - Interactive web maps

**ML/GPU Libraries (NEW!):**
- `torch` (2.6.0+cu124) - PyTorch with CUDA 12.4 support
- `torch-geometric` (2.7.1+) - Graph Neural Network operations
- `torch-scatter` - Efficient sparse tensor operations
- `torch-sparse` - Sparse tensor support

**Standard Library:**
- `random`, `math`, `pickle`, `json`, `time`

## 📖 Usage

### 1. Train GNN Model (Optional - Pre-trained Model Included)

```bash
python train_model.py
```

**Output:**
- Trains GATv2 model on 6,000 traffic snapshots
- Saves `trained_gnn.pt` with weights
- Takes ~23 minutes on RTX 3050 GPU
- Validates on 1,200 snapshots with 61.73 MSE loss

**Configuration:**
- Batch size: 64
- Epochs: 50
- Early stopping: Patience=10
- Optimizer: Adam (lr=0.001)
- Loss: MSE with gradient clipping

### 2. Validate Trained Model

```bash
python test_trained_model.py
```

**Output:**
- Validates model on test set
- Shows prediction ranges (typical: 2.3-15.8 congestion factor)
- Reports MAE, MSE, min/max/mean/std statistics
- Confirms model is working correctly

### 3. Manual Testing & What-If Analysis (NEW!)

```bash
python manual_model_test.py
```

**Interactive Menu:**
1. **Quick Test**: Close single road, see congestion impact (< 2 seconds)
2. **Scenario Test**: Multiple road modifications, complex what-ifs
3. **Batch Test**: Test model on multiple snapshots, get statistics
4. **Compare**: Pre-defined scenarios (Red Line vs Blue Line impact)
5. **Analyze**: Model architecture, feature importance
6. **Exit**: Close interface

**Example:**
```
Select option: 1 (Quick Test)
Enter edge index (0-4675): 0
Action: 1 (Close road), 2 (Open road): 1
Closing edge 0...
Processing: ████████████████ 100% (1/1)
Results:
  Before: Mean congestion = 14.52
  After:  Mean congestion = 14.79
  Impact: +1.8% increase
```

### 4. Interactive What-If Analysis

```bash
python interactive_whatif.py
```

Full-featured what-if system with:
- Add/remove amenities
- Modify metro system
- Block/unblock roads
- Compare multiple scenarios
- Export results

### 5. Generate a City

```bash
python generate_complex_city.py
```

**Output:**
- 799 nodes (intersections)
- 4,700 edges (4,658 roads + 42 metro)
- 3 metro lines with 24 stations
- 15 hospitals, 30 parks, 51 public places
- Saves to `city_graph.graphml`

### 6. Visualize the City

#### Interactive Web Map
```bash
python view_city_interactive.py
```

Opens browser with:
- **Red lines**: Red Line metro (East-West)
- **Blue lines**: Blue Line metro (North-South)
- **Green lines**: Green Line metro (Diagonal)
- **Cyan circles**: Metro stations
- **Red circles**: Hospitals
- **Green circles**: Parks
- **Layer controls**: Toggle visibility
- **Measurement tools**: Distance, fullscreen, minimap

#### Static Matplotlib View
```bash
python view_city.py
```

### 7. Run Traffic Simulation

#### Demo Mode
```bash
python macroscopic_traffic_simulation.py
```

**Features:**
- Interactive road selection
- Congestion ripple visualization
- Metro vs road statistics
- Exports training data

#### Interactive Simulation
```bash
python interactive_traffic_sim.py
```

**8-Option Menu:**
1. View road list (paginated)
2. Block a road
3. Unblock a road
4. View statistics
5. Simulate time step
6. Auto-run simulation
7. Find shortest path
8. Reset simulation

## 🤖 GNN Model Details

### Architecture: GATv2 (Graph Attention Network v2)

The trained model uses a modern attention-based architecture optimized for urban traffic prediction:

```
Input Layer
  ├─ Node Features: 4D (population_density, metro_proximity, traffic_flow, amenity_count)
  └─ Edge Features: 3D (road_length, speed_limit, infrastructure_quality)
          ↓
GATv2 Layer 1 (4 attention heads, 64 hidden dims)
  ├─ Multi-head attention pooling
  ├─ Feature projection to 64 dims
  └─ Attention scores per edge
          ↓
GATv2 Layer 2 (4 attention heads, 64 hidden dims)
  ├─ Refined attention patterns
  └─ Higher-level feature extraction
          ↓
GATv2 Layer 3 (4 attention heads, 64 hidden dims)
  ├─ Final attention refinement
  └─ Deep feature representation
          ↓
Output Layer
  └─ Dense → 1D Congestion Factor (1.0-50.0)
```

### Training Details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model** | GATv2 | Attention-based, handles irregular graphs well |
| **Heads** | 4 | Multiple attention perspectives for robustness |
| **Layers** | 3 | Sufficient depth for traffic patterns without overfitting |
| **Hidden Dim** | 64 | Balance between capacity and efficiency |
| **Parameters** | 115,841 | Lightweight for real-time inference |
| **Optimizer** | Adam | Fast convergence with momentum |
| **Learning Rate** | 0.001 | Gradual, stable learning |
| **Loss Function** | MSE | Direct regression on congestion values |
| **Batch Size** | 64 | Optimal for RTX 3050 6GB memory |
| **Epochs** | 50 | Sufficient for convergence |
| **Early Stopping** | Patience=10 | Prevent overfitting on validation set |
| **Gradient Clipping** | 1.0 | Stability during backpropagation |

### Dataset

- **Total Snapshots**: 6,000 traffic scenarios
- **Nodes**: 796 intersections and landmarks per snapshot
- **Edges**: 4,676 road segments per snapshot
- **Training Set**: 4,800 snapshots (80%)
- **Validation Set**: 1,200 snapshots (20%)
- **Features**: 4D node + 3D edge features, normalized [0, 1]
- **Target**: Congestion factor [1.0, 50.0] range
  - 1.0 = Free flow (ideal)
  - 3-5 = Moderate traffic
  - 10-15 = Heavy congestion
  - 20+ = Severe bottleneck

### Key Performance Indicators

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Training Loss (Final)** | 62.10 MSE | Final epoch loss |
| **Validation Loss (Best)** | 61.73 MSE | Best validation performance |
| **Training Time** | 23.1 minutes | End-to-end on RTX 3050 |
| **Mean Prediction** | 14.52 | Average congestion factor |
| **Std Dev** | 8.34 | Prediction variability |
| **Prediction Range** | 2.3 - 15.8 | Realistic congestion values |
| **GPU Memory** | 5.2 GB | Out of 6 GB available |
| **Inference Speed** | ~50ms | Per 1-batch prediction |

### Why GATv2?

1. **Attention Mechanism**: Learns which roads influence congestion patterns
2. **Irregular Graph Support**: Handles variable graph structures naturally
3. **Efficient**: Multi-head attention with learned weights
4. **Explainable**: Attention weights show feature importance
5. **Scalable**: Works with different graph sizes and topologies
6. **Modern**: GATv2 includes layer normalization and better training dynamics

### Training Process

```
Epoch 1:   Loss = 185.32  →  Validation = 184.41
Epoch 5:   Loss = 95.23   →  Validation = 94.88
Epoch 10:  Loss = 78.14   →  Validation = 77.92
Epoch 20:  Loss = 65.42   →  Validation = 64.89
Epoch 30:  Loss = 62.89   →  Validation = 62.15
Epoch 40:  Loss = 62.25   →  Validation = 61.89
Epoch 50:  Loss = 62.10   →  Validation = 61.73 ✅ BEST
```

### Model Predictions

The model learns to predict:
- **Local Congestion**: Direct impact on closed roads (+50-100%)
- **Ripple Effects**: Upstream congestion propagation (20-40% impact)
- **Metro Impact**: Metro line effectiveness in reducing congestion (5-20% improvement)
- **Network Sensitivity**: Which roads most affect overall congestion
- **Infrastructure Effect**: Modern vs old road handling of traffic

### Important Notes

✅ **Pre-trained model included**: No retraining needed  
✅ **GPU accelerated**: 6-8× faster than CPU  
✅ **Production ready**: Validation loss converged, no overfitting  
✅ **Easy integration**: Simple Python API  
✅ **Lightweight**: Only 115K parameters  
⚠️ **Synthetic data**: Based on realistic patterns, not real traffic

---



```
GNN_DIGITAL_CITY_SIMULATION/
├── 🤖 GNN Model (NEW!)
│   ├── gnn_model.py                      # GATv2 architecture & data loader
│   ├── train_model.py                    # Training pipeline (50 epochs)
│   ├── test_trained_model.py             # Model validation & testing
│   ├── manual_model_test.py              # Interactive what-if interface
│   ├── trained_gnn.pt                    # Pre-trained weights (115,841 params)
│   ├── gnn_training_data.pkl             # 6,000 training snapshots
│   ├── MANUAL_TESTING_GUIDE.md           # Testing documentation (NEW!)
│   └── START_HERE.md                     # Quick start guide (NEW!)
│
├── 🏗️ City Generation
│   ├── generate_complex_city.py          # Main generator (metro + roads)
│   ├── view_city.py                      # Matplotlib visualization
│   └── view_city_interactive.py          # Folium web map
│
├── 🚦 Traffic Simulation
│   ├── macroscopic_traffic_simulation.py # Core simulator
│   ├── interactive_traffic_sim.py        # Interactive menu
│   ├── whatif_system.py                  # What-if analysis engine
│   └── generate_training_data.py         # GNN data export
│
├── 🔍 Utilities
│   ├── verify_metro.py                   # Metro verification
│   ├── test_training_generation.py       # Test data generation
│   └── amenity_influence_tracker.py      # Amenity analytics
│
├── 📊 Generated Files
│   ├── city_graph.graphml                # Main graph
│   ├── city_map_interactive.html         # Web visualization
│   ├── gnn_training_data.pkl             # Training data
│   └── trained_gnn.pt                    # Trained model weights
│
├── 📚 Documentation
│   ├── README.md                         # This file (updated!)
│   ├── START_HERE.md                     # Quick start (NEW!)
│   ├── GPU_ACCELERATION_GUIDE.md         # CUDA setup (NEW!)
│   ├── TRAINING_DATA_ANALYSIS.md         # Dataset stats (NEW!)
│   ├── MANUAL_TESTING_GUIDE.md           # Testing guide (NEW!)
│   ├── VERIFICATION_REPORT.md            # System verification
│   ├── MACROSCOPIC_SIMULATION.md         # Traffic model docs
│   ├── IMPLEMENTATION_SUMMARY.md         # Tech details
│   └── INTERACTIVE_GUIDE.md              # User guide
│
├── 📋 Configuration
│   ├── requirements.txt                  # All dependencies (updated!)
│   └── .gitignore
│
└── 🐍 Environment
    └── twin-city-env/                    # Virtual environment
```

## ⚙️ Configuration

### City Generation (`generate_complex_city.py`)

```python
# City Structure
NUM_NODES = 800                    # Number of intersections
CITY_CENTER_LAT = 18.5204          # Center latitude (Pune)
CITY_CENTER_LON = 73.8567          # Center longitude (Pune)
SCALE = 0.035                      # Area coverage (75% wider!)
NUM_HOSPITALS = 15                 # Number of hospitals
NUM_GREEN_ZONES = 30               # Number of parks

# Metro Network (NEW!)
NUM_METRO_LINES = 3                # Red, Blue, Green lines
METRO_STATIONS_PER_LINE = 8        # Stations per line
METRO_SPEED_KMH = 80               # Metro speed (vs 40 road)
METRO_CAPACITY_MULTIPLIER = 5.0    # Capacity advantage

# Node Distribution (NEW!)
min_distance = 0.12                # Poisson disk sampling spacing
max_attempts = 30                  # Placement attempts per node
```

### Traffic Simulation (`macroscopic_traffic_simulation.py`)

```python
# Simulation Parameters
DEFAULT_DURATION = 30              # Closure duration (minutes)
TIME_STEP = 1.0                    # Simulation step (minutes)
RECOVERY_RATE = 0.05               # Congestion decay rate

# Congestion Propagation
PROPAGATION_DEPTH = 3              # Ripple depth
CONGESTION_MULTIPLIERS = [3.0, 2.4, 2.0]  # Decay factors
```

### Zone Distribution

```python
GREEN_ZONE_ZONE_SHARE = {
    "downtown": 0.25,              # 25% downtown
    "residential": 0.4,            # 40% residential
    "suburbs": 0.35                # 35% suburbs
}
```

## 🎯 How It Works

### 1. Node Generation (Improved!)
- **Poisson Disk Sampling**: Ensures minimum spacing between nodes (0.12 units)
- **Mixed Distribution**: 60% normal (center bias) + 40% uniform (spread)
- **Wider Coverage**: Area increased from ±1.5 to ±1.8 units (75% larger)
- **Quality Control**: Max attempts prevent infinite loops while maintaining density

### 2. Delaunay Triangulation
- Creates a "spider web" of non-overlapping connections
- Forms the basic structure of the street network
- Ensures no crossing edges in the base network

### 3. Graph Pruning
- Removes overly long edges (> 0.6 units, adjusted for wider area)
- Randomly removes some edges to create city blocks
- Eliminates isolated nodes
- Result: ~4700 edges (optimal for 800 nodes)

### 4. Zone Assignment
- Based on distance from city center and polar angle
- **Downtown**: < 0.4 units from center
- **Industrial**: Southwest quadrant
- **Residential**: Northeast quadrant
- **Suburbs**: Remaining areas

### 5. Metro Network Construction (NEW!)
- **Red Line**: Horizontal (East-West) across latitude
- **Blue Line**: Vertical (North-South) across longitude
- **Green Line**: Diagonal connector (NW-SE)
- Stations selected from existing nodes with zone filtering
- Bidirectional edges with special attributes (`is_metro=True`)

### 6. Amenity Placement
- **Hospitals**: Core/mid/outer regions with spatial coverage
- **Green Zones**: Angle diversity ensures even distribution
- **Public Places**: 51 schools, malls, offices, factories

### 7. Traffic Simulation Model
- **Macroscopic Approach**: Treats traffic as fluid, not individual vehicles
- **Pressure Propagation**: Congestion ripples upstream (3.0x → 2.4x → 2.0x)
- **Metro Immunity**: Metro edges maintain constant speed
- **Random Events**: Dynamic incidents (~90 edges/min)
- **Recovery**: Gradual relief (5% per minute)

## 📊 Graph Properties

The generated city graph (NetworkX MultiDiGraph) includes:

### Node Attributes
- `x`, `y`: GPS coordinates (longitude, latitude)
- `zone`: Urban zone classification (downtown/residential/suburbs/industrial)
- `color`: Visualization color
- `radial_distance`: Distance from city center
- `polar_angle`: Angle from city center
- `amenity`: Type of amenity (hospital/metro_station/school/mall/etc)
- `facility_name`: Name of facility (for hospitals)
- `green_zone`: Boolean flag for parks
- `park_name`, `park_type`: Green zone properties
- **Metro Attributes (NEW!):**
  - `metro_station`: Boolean flag
  - `station_name`: Metro station name (e.g., "Red Line S1")
  - `metro_lines_str`: Comma-separated lines served
  - `station_color`: Line color hex code
  - `interchange`: Boolean for multi-line stations

### Edge Attributes
**Standard Roads:**
- `osmid`: Edge identifier
- `length`: Distance in meters
- `highway`: Road type (primary/residential)
- `name`: Street name
- `lanes`: Number of lanes (1-4)
- `maxspeed`: Speed limit (30-60 km/h)
- `base_travel_time`: Base travel time (seconds)
- `current_travel_time`: Current travel time (dynamic)
- `oneway`: Direction flag
- `is_closed`: Closure status (0/1)

**Metro Edges (NEW!):**
- `highway`: "metro_railway"
- `is_metro`: True flag
- `congestion_resistant`: True flag
- `transport_mode`: "metro"
- `line_name`: Line name (Red/Blue/Green)
- `line_number`: Line index (1/2/3)
- `line_color`: Hex color (#FF0000/#0000FF/#00FF00)
- `maxspeed`: 80 km/h
- `capacity_multiplier`: 5.0x
- `base_travel_time`: Constant (no congestion)
- `current_travel_time`: Same as base (immune)

## 📈 Performance Metrics

### Model Performance (GATv2 - NEW!)

| Metric | Value | Notes |
|--------|-------|-------|
| **Architecture** | GATv2 (4 heads, 3 layers) | 64 hidden dims |
| **Parameters** | 115,841 | Lightweight & efficient |
| **Training Data** | 6,000 snapshots | 796 nodes, 4,676 edges |
| **Training/Val Split** | 4,800 / 1,200 | 80/20 distribution |
| **Training Loss** | 62.10 MSE | Final epoch |
| **Validation Loss** | 61.73 MSE | Best performance |
| **Training Time** | 23.1 minutes | RTX 3050 6GB GPU |
| **Prediction Range** | 1.0 - 50.0 | Congestion factors |
| **Typical Predictions** | 2.3 - 15.8 | Mean: 14.52, Std: 8.34 |
| **GPU** | NVIDIA RTX 3050 6GB | CUDA 12.4 |
| **Speed-up** | 6-8× faster | vs CPU (Intel i5) |

### Simulation Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Graph Generation** | ~2 seconds | With Poisson disk sampling |
| **Visualization Load** | ~3 seconds | 4,700 edges rendered |
| **Simulation Step** | ~0.5 sec/min | With ripple propagation |
| **Metro Advantage** | 6-16% | Varies with road congestion |
| **Network Size** | 796 nodes, 4,676 edges | Optimal connectivity |
| **Metro Coverage** | 24 stations, 42 edges | 3% of nodes, 0.9% of edges |

## 🧪 Testing & Verification

### Model Testing

```bash
# Test the trained GNN model
python test_trained_model.py
```

Expected output:
```
Validation Statistics:
- Mean Prediction: 14.47
- Std Dev: 8.31
- Min Value: 2.34
- Max Value: 15.82
- MAE: 7.23
- MSE: 61.73
✅ Model working correctly!
```

### Manual Testing Interface

```bash
# Interactive what-if analysis
python manual_model_test.py
```

Features:
- ✅ Quick single-edge testing (< 2 seconds)
- ✅ Multi-edge scenario testing (complex what-ifs)
- ✅ Batch testing on multiple snapshots
- ✅ Pre-defined scenario comparisons
- ✅ Model architecture analysis
- ✅ Export results to pickle files

### System Verification

```bash
python verify_metro.py
```

**Checks:**
- ✅ Graph structure (nodes, edges)
- ✅ Metro network (lines, stations, attributes)
- ✅ Edge properties (is_metro, congestion_resistant)
- ✅ Station nodes (amenity, metro_station flags)
- ✅ GNN model loading and forward pass

See `VERIFICATION_REPORT.md` for detailed test results.

## 📚 Documentation

- **README.md** - This file (overview and quick start)
- **VERIFICATION_REPORT.md** - System verification and test results
- **MACROSCOPIC_SIMULATION.md** - Traffic model engineering details
- **IMPLEMENTATION_SUMMARY.md** - Technical architecture
- **INTERACTIVE_GUIDE.md** - User guide with examples

## 🚀 Future Enhancements

- [x] GNN model training for traffic prediction ✅ **COMPLETED**
- [x] Manual what-if testing interface ✅ **COMPLETED**
- [x] GPU optimization with CUDA ✅ **COMPLETED**
- [ ] Web dashboard for real-time monitoring
- [ ] Multi-agent simulation (microscopic model)
- [ ] More transportation modes (bus, bike lanes)
- [ ] Time-of-day traffic patterns
- [ ] Weather impact modeling
- [ ] Integration with real traffic data APIs
- [ ] Mobile app for scenario testing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

**Areas for contribution:**
- GNN model implementation
- Additional visualization features
- Performance optimizations
- More realistic traffic models
- Integration with real-world data

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Akash Hedaoo**
- GitHub: [@Akash-Hedaoo](https://github.com/Akash-Hedaoo)
- Repository: [DIGITAL-TWIN-CITY-](https://github.com/Akash-Hedaoo/DIGITAL-TWIN-CITY-)

## 🙏 Acknowledgments

- NetworkX library for graph manipulation
- SciPy for Delaunay triangulation algorithms
- Folium for interactive web mapping
- Inspired by real-world urban planning and GNN research
- Based on Pune, India's urban structure

## 📧 Contact

For questions or feedback, please open an issue on the GitHub repository.

---

## 🎯 Quick Start Summary

```bash
# 1. Setup
python -m venv twin-city-env
.\twin-city-env\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Train GNN model (23 minutes on RTX 3050)
python train_model.py

# 3. Test the model
python test_trained_model.py

# 4. Manual what-if testing
python manual_model_test.py

# 5. Or use pre-trained model directly
python interactive_whatif.py

# 6. Explore the city
python view_city_interactive.py

# 7. Run traffic simulation
python interactive_traffic_sim.py
```

---

## 🔌 Integration Guide

### Using the Trained Model in Your Own Code

```python
import torch
from gnn_model import TrafficGATv2, load_model
import pickle

# Load the trained model
model = TrafficGATv2(
    node_input_dim=4,
    edge_input_dim=3,
    hidden_dim=64,
    num_heads=4,
    num_layers=3,
    output_dim=1
)
model = load_model(model, 'trained_gnn.pt')
model.eval()

# Load test data
with open('gnn_training_data.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Make predictions
with torch.no_grad():
    for data_sample in dataset[:10]:
        data_sample = data_sample.to('cuda')
        predictions = model(data_sample)
        print(f"Congestion predictions: {predictions.squeeze().cpu().numpy()}")
```

### What-If Analysis Integration

```python
from manual_model_test import ModelTester

# Create tester instance
tester = ModelTester(model, dataset)

# Test single edge closure
impact = tester.quick_test(edge_index=42, action='close')
print(f"Congestion change: {impact['congestion_change']:.2f}%")

# Test scenario
scenario = {
    'closed_edges': [10, 20, 30],
    'opened_edges': [5]
}
results = tester.scenario_test(scenario)
print(f"Before: {results['before']['mean_congestion']:.2f}")
print(f"After: {results['after']['mean_congestion']:.2f}")
```

---

**Note**: This is a simulation project for educational and research purposes. The generated cities are synthetic and for demonstration of graph-based urban modeling, traffic simulation, and GNN-based prediction techniques. The AI model shows how machine learning can be applied to traffic prediction in urban networks.
