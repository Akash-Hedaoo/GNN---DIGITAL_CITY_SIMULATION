# GNN - Digital City Simulation with Traffic Modeling

A Graph Neural Network (GNN) based digital city simulation featuring realistic urban environments, multi-modal transportation networks, and advanced macroscopic traffic simulation. The project creates organic city structures with integrated metro systems, traffic congestion modeling, and interactive visualization.

## 🌆 Overview

This project simulates a digital city modeled after Pune, India, featuring:
- **Complex graph-based urban infrastructure** with 800 nodes and 4700+ edges
- **Multi-modal transportation**: Road network + 3 metro lines (Red, Blue, Green)
- **Macroscopic traffic simulation** with pressure-based congestion propagation
- **Interactive traffic control** - block roads, simulate events, track statistics
- **Real-time visualization** - Interactive web maps with layer controls
- **GNN training data generation** - Export scenarios for machine learning

The simulation uses advanced graph algorithms and Poisson disk sampling to create realistic street networks that mimic organic city growth patterns, while the traffic model demonstrates how metro systems can ease urban congestion.

## ✨ Features

### 🏙️ City Generation
- **Organic Network Structure**: Uses Delaunay triangulation + Poisson disk sampling
- **Better Node Distribution**: Minimum spacing constraints prevent clustering (75% wider coverage)
- **Multi-Zone Urban Structure**: 
  - 🏙️ Downtown (city center)
  - 🏡 Residential areas
  - 🏘️ Suburbs
  - 🏭 Industrial zones
- **Civic Amenities**: 15 hospitals strategically placed
- **Green Zones**: 30 parks distributed with angle diversity
- **GPS Coordinates**: Real-world coordinates based on Pune, India (18.5204°N, 73.8567°E)

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
pip install networkx numpy scipy matplotlib plotly folium
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### Dependencies

- `networkx` (3.5) - Graph creation and manipulation
- `numpy` (2.3.5) - Numerical computations
- `scipy` (1.16.3) - Delaunay triangulation
- `matplotlib` (3.10.7) - Static visualization
- `plotly` (6.5.0) - Interactive plotting
- `folium` - Interactive web maps
- Standard library: `random`, `math`, `pickle`

## 📖 Usage

### 1. Generate a City

Run the city generation script to create a new synthetic city:

```bash
python generate_complex_city.py
```

**Output:**
- 799 nodes (intersections) with Poisson disk sampling
- 4700 edges (4658 roads + 42 metro)
- 3 metro lines with 24 stations
- 15 hospitals, 30 parks, 51 public places
- Saves to `city_graph.graphml`

### 2. Visualize the City

#### Interactive Web Map (Recommended)
```bash
python view_city_interactive.py
```

Opens in browser with:
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

Shows traditional colored node visualization.

### 3. Run Traffic Simulation

#### Demo Mode (Recommended for first run)
```bash
python macroscopic_traffic_simulation.py
```

**Features:**
- Interactive road selection menu
- Random or custom road blocking
- Congestion ripple visualization
- Metro vs road statistics
- Exports training data to `.pkl`

**Sample Output:**
```
🚧 Closing road: 283 -> 286
  ↑ Backlog at depth 0: 7 edges (×3.0 congestion)
  ↑ Backlog at depth 1: 6 edges (×2.4 congestion)
  ↑ Backlog at depth 2: 6 edges (×2.0 congestion)

--- Minute 30 ---
📊 TRAFFIC STATISTICS
Road Congestion:     1.08x
Metro Congestion:    1.00x (Constant)
Metro Advantage:     7.6% faster! 🎉
```

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

### 4. Verify Metro Network

```bash
python verify_metro.py
```

Displays:
- Graph summary (nodes, edges)
- Metro network details (lines, stations)
- Sample edge attributes
- Station node properties

### 5. Generate Training Data

```bash
python generate_training_data.py
```

Creates 100+ traffic scenarios with:
- Random road closures
- Different event frequencies
- Various durations
- Exports to `training_scenarios_*.pkl`

## 🏗️ Project Structure

```
GNN_DIGITAL_CITY_SIMULATION/
├── 🏗️ City Generation
│   ├── generate_complex_city.py          # Main generator (metro + roads)
│   ├── view_city.py                      # Matplotlib visualization
│   └── view_city_interactive.py          # Folium web map (NEW!)
│
├── 🚦 Traffic Simulation
│   ├── macroscopic_traffic_simulation.py # Core simulator (NEW!)
│   ├── interactive_traffic_sim.py        # Interactive menu (NEW!)
│   └── generate_training_data.py         # GNN data export (NEW!)
│
├── 🔍 Utilities
│   ├── verify_metro.py                   # Metro verification (NEW!)
│   └── test_training_generation.py       # Test data gen (NEW!)
│
├── 📊 Generated Files
│   ├── city_graph.graphml                # Main graph
│   ├── city_map_interactive.html         # Web visualization
│   ├── demo_traffic_data.pkl             # Training data
│   └── training_scenarios_*.pkl          # Batch scenarios
│
├── 📚 Documentation
│   ├── README.md                         # This file
│   ├── VERIFICATION_REPORT.md            # System verification (NEW!)
│   ├── MACROSCOPIC_SIMULATION.md         # Traffic model docs (NEW!)
│   ├── IMPLEMENTATION_SUMMARY.md         # Tech details (NEW!)
│   └── INTERACTIVE_GUIDE.md              # User guide (NEW!)
│
└── 🐍 Environment
    ├── twin-city-env/                    # Virtual environment
    └── requirements.txt                  # Dependencies
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

| Metric | Value | Notes |
|--------|-------|-------|
| **Graph Generation** | ~2 seconds | With Poisson disk sampling |
| **Visualization Load** | ~3 seconds | 4700 edges rendered |
| **Simulation Step** | ~0.5 sec/min | With ripple propagation |
| **Metro Advantage** | 6-16% | Varies with road congestion |
| **Network Size** | 799 nodes, 4700 edges | Optimal connectivity |
| **Metro Coverage** | 24 stations, 42 edges | 3% of nodes, 0.9% of edges |

## 🧪 Testing & Verification

Run the verification script to check all systems:

```bash
python verify_metro.py
```

**Checks:**
- ✅ Graph structure (nodes, edges)
- ✅ Metro network (lines, stations, attributes)
- ✅ Edge properties (is_metro, congestion_resistant)
- ✅ Station nodes (amenity, metro_station flags)

See `VERIFICATION_REPORT.md` for detailed test results.

## 📚 Documentation

- **README.md** - This file (overview and quick start)
- **VERIFICATION_REPORT.md** - System verification and test results
- **MACROSCOPIC_SIMULATION.md** - Traffic model engineering details
- **IMPLEMENTATION_SUMMARY.md** - Technical architecture
- **INTERACTIVE_GUIDE.md** - User guide with examples

## 🚀 Future Enhancements

- [ ] GNN model training for traffic prediction
- [ ] Web dashboard for real-time monitoring
- [ ] Multi-agent simulation (microscopic model)
- [ ] More transportation modes (bus, bike lanes)
- [ ] Time-of-day traffic patterns
- [ ] Weather impact modeling
- [ ] Integration with real traffic data APIs

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
pip install networkx numpy scipy matplotlib plotly folium

# 2. Generate city with metro
python generate_complex_city.py

# 3. View interactive map
python view_city_interactive.py

# 4. Run traffic simulation
python macroscopic_traffic_simulation.py

# 5. Verify everything works
python verify_metro.py
```

---

**Note**: This is a simulation project for educational and research purposes. The generated cities are synthetic and for demonstration of graph-based urban modeling and traffic simulation techniques. Metro network demonstrates how multi-modal transportation can ease urban congestion.
