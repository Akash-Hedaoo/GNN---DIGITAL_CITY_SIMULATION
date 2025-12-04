# 🌆 GNN Digital Twin City Simulation

**A complete urban simulation platform featuring Graph Attention Networks (GATv2), realistic traffic modeling, interactive visualization, and AI-powered traffic prediction.**

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

This project creates a **digital twin of an urban city** (modeled after Pune, India) featuring:

- **Complex Urban Infrastructure**: 796 nodes, 4,676+ edges representing roads and metro lines
- **Multi-Modal Transportation**: 3 metro lines (Red, Blue, Green) + comprehensive road network
- **AI-Powered Predictions**: Trained GATv2 model predicting traffic congestion in real-time
- **Interactive Platform**: Web-based visualization with control systems
- **What-If Analysis**: Simulate scenarios before implementation
- **Production-Ready**: Fully tested with comprehensive documentation

### The Problem It Solves

Urban planners need to understand traffic impact **before** making infrastructure changes:
- ❓ What if we close this intersection?
- ❓ How will metro expansion affect congestion?
- ❓ Which roads are critical for the network?
- ❓ Where will congestion spread if a route fails?

This platform **answers these questions with AI predictions**.

---

## ✨ Key Features

### 🤖 AI Traffic Prediction
- **Model**: Graph Attention Network v2 (GATv2)
- **Size**: 115,841 parameters (highly efficient)
- **Training**: 6,000 scenarios, 50 epochs, ~23 minutes on RTX 3050
- **Accuracy**: 61.73 MSE validation loss
- **Speed**: Real-time predictions (100-500ms per analysis)
- **Features**: 
  - ✅ Population density
  - ✅ Metro station proximity
  - ✅ Node coordinates
  - ✅ Edge closure status
  - ✅ Travel time baseline

### 🏙️ Realistic City Generation
- **Organic Street Layout**: Delaunay triangulation + Poisson disk sampling
- **Multi-Zone Structure**: Downtown, residential, suburbs, industrial
- **Amenities**: 15 hospitals, 30 parks, distributed strategically
- **GPS Coordinates**: Real-world Pune coordinates
- **Smart Distribution**: Minimum spacing prevents unrealistic clustering

### 🚇 Multi-Modal Transportation
- **3 Metro Lines**: Red (East-West), Blue (North-South), Green (Diagonal)
- **24 Metro Stations**: 8 per line with bidirectional service
- **Performance**: 
  - Metro: 80 km/h, 5x capacity, congestion-immune
  - Roads: 40 km/h base, congestion-sensitive
  - Benefit: 6-16% faster travel time with metro

### 🚦 Advanced Traffic Simulation
- **Macroscopic Modeling**: Fluid dynamics approach (not agent-based)
- **Pressure Propagation**: Congestion ripples upstream with decay (3.0x → 2.4x → 2.0x)
- **Dynamic Events**: ~90 traffic incidents per minute
- **Recovery System**: Gradual congestion relief over time
- **Network Statistics**: Separate road vs. metro metrics

### 🎮 Interactive Features
- **Road Blocking**: Close any road and observe ripple effects
- **Real-Time Stats**: Network delay, affected edges, congestion levels
- **Route Finding**: Calculate paths considering current congestion
- **Multiple Modes**: Quick test, scenario test, batch testing
- **Export**: Save scenarios for GNN training
- **Dark Theme**: Modern CartoDB interface

### 🆕 Node Removal & Impact Analysis
- **Interactive Removal**: Click nodes to simulate infrastructure removal
- **Automatic Edge Closure**: All connected edges automatically close
- **Impact Analysis**: Detailed statistics on traffic congestion
- **Visual Feedback**: Pink dashed lines for affected edges
- **Full Reversibility**: Restore nodes at any time
- **Sidebar Tracking**: See all removed nodes and their impact

### 🎨 Rich Visualization
- **Interactive Web Map**: Leaflet.js powered
- **Layer Controls**: Toggle roads, metro, zones, amenities
- **Color Coding**: All metro lines distinctly colored
- **Real-Time Updates**: Instant visual feedback
- **Dark/Light Themes**: Full theme support
- **Responsive Design**: Works on desktop and mobile

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DIGITAL TWIN CITY SYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Frontend (Browser)          Backend (Flask)          ML Model    │
│  ┌──────────────────┐        ┌─────────────┐         ┌─────────┐ │
│  │ Interactive Map  │ ◄────► │ REST API    │ ◄─────► │ GATv2   │ │
│  │ (Leaflet.js)     │  JSON  │ (Python)    │ Query   │ Network │ │
│  └──────────────────┘        └─────────────┘         └─────────┘ │
│        │                            │                      │      │
│        └────────────────────────────┴──────────────────────┘      │
│                                                                     │
│  Data Layer:                                                        │
│  • city_graph.graphml (796 nodes, 4,676+ edges)                   │
│  • trained_gnn.pt (115,841 parameters)                            │
│  • Training snapshots (6,000 scenarios)                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.13+ |
| **Backend** | Flask | Latest |
| **Graphs** | NetworkX | 3.5 |
| **ML Framework** | PyTorch | 2.6+ |
| **ML Ops** | PyTorch Geometric | 2.7.1+ |
| **GPU** | CUDA | 12.4 |
| **Frontend** | HTML5/CSS3/JS | Latest |
| **Visualization** | Leaflet.js | Latest |

---

## 📦 Installation

### Prerequisites
- Python 3.13 or higher
- pip package manager
- Virtual environment tool
- NVIDIA GPU (RTX 3050+ recommended for training)

### Step 1: Clone Repository
```bash
git clone https://github.com/Akash-Hedaoo/GNN---DIGITAL_CITY_SIMULATION.git
cd GNN---DIGITAL_CITY_SIMULATION
```

### Step 2: Create Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# PyTorch with GPU support (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch_geometric; print('PyTorch Geometric installed')"
python -c "import networkx; print('NetworkX installed')"
```

---

## 🚀 Quick Start

### 1. Start the Application
```bash
python backend/app.py
```

**Output:**
```
============================================================
🚦 Digital Twin City Simulation - Flask Backend
============================================================
[INIT] Using device: cuda
[OK] Model loaded successfully
[OK] Graph loaded: 796 nodes, 4676 edges
📡 Starting server...
   Frontend: http://localhost:5000
   API Docs: http://localhost:5000/api/status
============================================================
```

### 2. Open in Browser
Navigate to: **http://localhost:5000**

### 3. Try the Features
- 🗺️ **Explore Map**: Navigate, zoom, see infrastructure
- 🔴 **Click Nodes**: View node details and remove/restore
- 🚧 **Close Roads**: Click edges to simulate closures
- 📊 **Run Prediction**: Predict traffic with current configuration
- 📈 **View Analysis**: Export and analyze scenarios
- ⏰ **Change Time**: Adjust hour to see time-dependent impacts

---

## 📖 Usage Guide

### Train Model (Optional - Pre-trained Model Included)
```bash
python train_model.py
```

**Configuration:**
- Batch size: 64
- Epochs: 50
- Learning rate: 0.001
- Early stopping: Patience=10
- Training time: ~23 minutes on RTX 3050

**Output:**
- `trained_gnn.pt` - Model weights
- Training/validation loss curves

### Test Model
```bash
python test_trained_model.py
```

**Shows:**
- Prediction ranges
- Statistics (MAE, MSE, min/max/mean/std)
- Model correctness verification

### Manual What-If Analysis
```bash
python manual_model_test.py
```

**Options:**
1. **Quick Test**: Single road closure
2. **Scenario Test**: Multiple modifications
3. **Batch Test**: Test on multiple snapshots
4. **Compare**: Pre-defined scenarios
5. **Analyze**: Model architecture details

### Interactive Web Interface

#### Node Operations
```
1. Click any node on map
2. Info panel appears with node details
3. Click "Remove Node" button
4. System analyzes impact:
   - Finds all connected edges
   - Runs GNN prediction
   - Shows impact statistics
5. Review affected edges (pink dashed lines)
6. Click "Restore" to undo
```

#### Road Closure
```
1. Click any edge (road) on map
2. Edge turns pink and dashed
3. Appears in "Road Closure" panel
4. Click "Run Prediction" to analyze
5. View congestion impact
6. Click × to remove from closure list
```

#### Time-Based Analysis
```
1. Use time slider (0-23 hours)
2. Different congestion at different times
3. Peak hours: 8-9 AM, 5-7 PM
4. Off-peak: 2-4 AM
5. Analysis updates automatically
```

#### Search & Navigate
```
1. Use search bar at top
2. Search by:
   - Node ID (e.g., "0")
   - Amenity type (e.g., "hospital")
   - Zone name (e.g., "downtown")
3. Click result to navigate
```

---

## 🔌 API Reference

### Status Endpoint
```
GET /api/status
```

**Response:**
```json
{
  "status": "online",
  "model_loaded": true,
  "graph_loaded": true,
  "device": "cuda",
  "nodes": 796,
  "edges": 4676
}
```

### Get Graph Data
```
GET /api/graph
```

**Response:**
```json
{
  "nodes": [
    {"id": "0", "x": 0.0, "y": 0.0, "zone": "downtown", "population": 50000, "amenity": "none", "is_metro": false},
    ...
  ],
  "edges": [
    {"source": "0", "target": "1", "is_metro": false, "travel_time": 1.5, "metro_line": null},
    ...
  ],
  "node_count": 796,
  "edge_count": 4676
}
```

### Predict Traffic
```
POST /api/predict
```

**Request:**
```json
{
  "closed_roads": ["0-1", "5-6"],
  "hour": 9
}
```

**Response:**
```json
{
  "predictions": [
    {"source": "0", "target": "1", "is_metro": false, "congestion": 0.62},
    ...
  ],
  "stats": {
    "mean_congestion": 0.412,
    "max_congestion": 0.91,
    "road_mean": 0.398,
    "metro_mean": 0.521,
    "closed_roads": 2
  }
}
```

### Analyze Node Removal
```
POST /api/analyze-node-removal
```

**Request:**
```json
{
  "node_id": "156",
  "hour": 9
}
```

**Response:**
```json
{
  "impact_analysis": {
    "removed_node": "156",
    "node_details": {
      "id": "156",
      "zone": "downtown",
      "population": 12345,
      "amenity": "hospital",
      "x": 18.52,
      "y": 73.87
    },
    "closed_edges_count": 3,
    "mean_closed_edge_congestion": 0.688,
    "max_closed_edge_congestion": 0.875,
    "mean_congestion": 0.412,
    "max_congestion": 0.91,
    "road_mean": 0.398,
    "metro_mean": 0.521
  },
  "affected_edges": ["156-42", "42-156", "156-201"],
  "predictions": [...]
}
```

### Get Metro Lines
```
GET /api/metro-lines
```

**Response:**
```json
{
  "lines": {
    "Line 1": {
      "edges": [{"source": "0", "target": "1"}, ...],
      "stations": ["0", "2", "5", ...]
    },
    ...
  },
  "stations": [
    {"id": "0", "x": 18.52, "y": 73.87, "name": "Central Station"},
    ...
  ],
  "total_stations": 24
}
```

### Get Amenities
```
GET /api/amenities
```

**Response:**
```json
{
  "amenities": {
    "hospital": [{"id": "156", "x": 18.52, "y": 73.87, "zone": "downtown"}, ...],
    "park": [...],
    ...
  }
}
```

### Search Nodes
```
GET /api/search?q=hospital
```

**Response:**
```json
{
  "results": [
    {"id": "156", "x": 18.52, "y": 73.87, "zone": "downtown", "population": 12345, "amenity": "hospital"},
    ...
  ]
}
```

### Find Shortest Path
```
POST /api/shortest-path
```

**Request:**
```json
{
  "source": "0",
  "target": "100"
}
```

**Response:**
```json
{
  "path": ["0", "5", "10", "20", "100"],
  "length": 12.5,
  "hops": 4
}
```

---

## 📁 Project Structure

```
GNN---DIGITAL_CITY_SIMULATION/
├── backend/
│   ├── app.py                 # Flask REST API server
│   └── requirements.txt        # Backend dependencies
├── frontend/
│   ├── index.html            # Main web interface
│   ├── analysis.html         # Analysis page
│   ├── app.js                # Main JavaScript logic
│   ├── analysis.js           # Analysis page logic
│   ├── style.css             # Styling
│   └── analysis.css          # Analysis page styling
├── gnn_model.py              # GATv2 model architecture
├── train_model.py            # Model training script
├── test_trained_model.py     # Model testing
├── manual_model_test.py      # Interactive what-if analysis
├── macroscopic_traffic_simulation.py  # Traffic simulation engine
├── whatif_system.py          # Scenario analysis
├── city_graph.graphml        # City infrastructure graph
├── trained_gnn.pt            # Pre-trained model weights
├── requirements.txt          # Project dependencies
└── README_MASTER.md          # This file
```

---

## 🎓 Advanced Features

### Node Removal & Impact Analysis
The most recent feature allows comprehensive infrastructure impact analysis:

1. **Click any node** on the map
2. **Click "Remove Node"** button
3. System automatically:
   - Finds all connected edges
   - Marks them as closed
   - Runs GNN model prediction
   - Calculates impact statistics
4. **View detailed analysis**:
   - Number of closed edges
   - Mean/max congestion on affected edges
   - Network-wide impact
   - Road vs. metro breakdown
5. **Restore node** to revert changes

**Use Cases:**
- Metro station expansion planning
- Hospital accessibility analysis
- Critical intersection identification
- Disaster recovery planning
- Road maintenance scheduling

### What-If Scenario Analysis
- **Quick Test**: Single road closure, instant impact
- **Scenario Test**: Multiple closures, complex configurations
- **Batch Test**: Test 100+ configurations, get statistics
- **Comparison**: Pre-defined scenarios (metro line impact, etc.)
- **Export**: Save results for reporting

### Model Interpretability
- **Feature Importance**: Which features affect predictions most?
- **Edge Analysis**: Identify critical/bottleneck roads
- **Zone Analytics**: Impact by urban zone
- **Time Analysis**: Peak vs. off-peak patterns
- **Network Resilience**: Which nodes are most critical?

---

## 🐛 Troubleshooting

### "Model not loaded" Error
```bash
# Solution: Ensure trained_gnn.pt exists in root directory
# If not, train the model:
python train_model.py

# Or download pre-trained weights from repository
```

### "Graph not loaded" Error
```bash
# Solution: Ensure city_graph.graphml exists
# If not, regenerate:
python -c "from train_model import generate_city; generate_city()"
```

### GPU Memory Error
```bash
# Solution: Reduce batch size in training
# In train_model.py, change:
# batch_size = 64  # Reduce to 32 or 16
```

### API Connection Failed
```bash
# Solution: Ensure backend is running
python backend/app.py

# Check if running on correct port:
# http://localhost:5000

# For external access:
# In app.py: app.run(host='0.0.0.0', port=5000)
```

### Slow Predictions
```bash
# Solution: Check GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Port Already in Use
```bash
# Solution: Use different port in app.py
# app.run(host='0.0.0.0', port=5001)  # Use 5001 instead
```

---

## 📊 Performance Metrics

### Model Performance
| Metric | Value |
|--------|-------|
| Parameters | 115,841 |
| Training Time | 23.1 min (RTX 3050) |
| Validation Loss | 61.73 MSE |
| Prediction Speed | 100-500ms |
| Model Size | ~5 MB |

### System Performance
| Operation | Time |
|-----------|------|
| API Response | <100ms (avg) |
| Graph Load | <500ms |
| Model Inference | 100-500ms |
| Prediction Run | 1-2 seconds |
| Node Removal | 2-3 seconds |

### Network Scale
| Metric | Value |
|--------|-------|
| Nodes | 796 |
| Edges | 4,676 |
| Metro Stations | 24 |
| Amenities | 45 |
| Coverage | ~25 sq km equivalent |

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing`)
7. Open a Pull Request

### Areas for Contribution
- 🎨 UI/UX improvements
- 📊 Additional analytics
- 🔧 Performance optimization
- 📚 Documentation enhancement
- 🧪 Testing coverage
- 🐛 Bug fixes

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **NetworkX**: Graph operations and algorithms
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural network support
- **Folium/Leaflet**: Interactive mapping
- **Pune, India**: City inspiration and coordinates
- **Contributors**: All developers and testers

---

## 📞 Support

For questions, issues, or suggestions:

1. **GitHub Issues**: Report bugs and request features
2. **Documentation**: Check README_MASTER.md and feature docs
3. **Discussions**: Join project discussions
4. **Email**: Contact project maintainers

---

## 📚 Additional Resources

### Documentation
- `NODE_REMOVAL_FEATURE.md` - Node removal detailed guide
- `QUICK_START_NODE_REMOVAL.md` - Quick start for node removal
- `VISUAL_GUIDE_NODE_REMOVAL.md` - Visual walkthrough
- `CODE_REFERENCE_NODE_REMOVAL.md` - Implementation details
- `IMPLEMENTATION_VERIFICATION.md` - QA verification

### Quick Links
- 🌐 [Website](http://localhost:5000)
- 📖 [API Documentation](#api-reference)
- 🎓 [Usage Guide](#usage-guide)
- 🚀 [Getting Started](#quick-start)

---

## 🎯 Roadmap

### Version 1.1 (Next)
- [ ] Multi-node removal
- [ ] Restoration timeline
- [ ] Alternative route suggestions
- [ ] Export impact reports (PDF)

### Version 1.2 (Future)
- [ ] Historical comparison
- [ ] Predictive maintenance alerts
- [ ] Network resilience scoring
- [ ] Real-time weather integration

### Version 2.0 (Long-term)
- [ ] 3D city visualization
- [ ] Ride-sharing optimization
- [ ] Autonomous vehicle integration
- [ ] Real-time data integration

---

**Status**: 🟢 **Production Ready**  
**Last Updated**: December 3, 2025  
**Version**: 1.0

---

© 2025 GNN Digital Twin City Simulation. All rights reserved.
