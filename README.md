# GNN - Digital City Simulation

A Graph Neural Network (GNN) based digital city simulation that generates realistic synthetic urban environments using Delaunay triangulation and graph theory. The project creates organic, complex city structures with multiple zones, amenities, and realistic street networks.

## 🌆 Overview

This project simulates a digital city modeled after Pune, India, generating a complex graph-based representation of urban infrastructure. The city includes various zones (downtown, residential, suburbs, and industrial areas), civic amenities (hospitals), and green spaces (parks) distributed strategically across the urban landscape.

The simulation uses advanced graph algorithms to create realistic street networks that mimic organic city growth patterns, avoiding the artificial grid-like structures of traditional city generation methods.

## ✨ Features

- **Organic City Generation**: Uses Delaunay triangulation to create realistic, non-grid street networks
- **Multi-Zone Urban Structure**: 
  - 🏙️ Downtown (city center)
  - 🏡 Residential areas
  - 🏘️ Suburbs
  - 🏭 Industrial zones
- **Civic Amenities**: Strategic placement of 15 hospitals across the city for optimal coverage
- **Green Zones**: 30 parks and green spaces distributed across different city zones
- **Realistic Road Networks**: 
  - Primary highways (ring roads)
  - Residential streets
  - Variable lanes and speed limits
- **GPS Coordinates**: Real-world coordinates based on Pune, India (18.5204°N, 73.8567°E)
- **Graph-based Structure**: Uses NetworkX MultiDiGraph for complex urban modeling
- **Visualization**: Beautiful matplotlib-based visualization with color-coded zones and amenities

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Akash-Hedaoo/GNN---DIGITAL_CITY_SIMULATION.git
cd GNN---DIGITAL_CITY_SIMULATION
```

2. Install required dependencies:
```bash
pip install networkx numpy scipy matplotlib
```

Or create a requirements.txt:
```bash
pip install -r requirements.txt
```

### Dependencies

- `networkx` - Graph creation and manipulation
- `numpy` - Numerical computations
- `scipy` - Delaunay triangulation
- `matplotlib` - Visualization
- Standard library: `random`, `math`

## 📖 Usage

### Generate a City

Run the city generation script to create a new synthetic city:

```bash
python generate_complex_city.py
```

This will:
- Generate 800 nodes (intersections) with organic distribution
- Create realistic street connections using Delaunay triangulation
- Assign zones based on distance from city center
- Place 15 hospitals strategically across zones
- Distribute 30 green zones/parks throughout the city
- Save the graph to `city_graph.graphml`

### Visualize the City

After generating the city, visualize it with:

```bash
python view_city.py
```

This will display an interactive map showing:
- **Blue nodes**: Downtown area
- **Green nodes**: Residential zones
- **Gray nodes**: Suburbs
- **Red nodes**: Industrial zones
- **Yellow stars**: Hospitals
- **Bright green squares**: Parks/green zones
- **Orange edges**: Primary highways (ring roads)
- **Dark edges**: Residential streets

## 🏗️ Project Structure

```
GNN---DIGITAL_CITY_SIMULATION/
├── generate_complex_city.py    # Main city generation script
├── view_city.py                 # Visualization script
├── city_graph.graphml          # Generated city graph (auto-generated)
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## ⚙️ Configuration

You can customize the city generation by modifying the configuration constants in `generate_complex_city.py`:

```python
NUM_NODES = 800              # Number of intersections
CITY_CENTER_LAT = 18.5204    # Center latitude (Pune)
CITY_CENTER_LON = 73.8567    # Center longitude (Pune)
SCALE = 0.02                 # Area coverage scale
NUM_HOSPITALS = 15           # Number of hospitals
NUM_GREEN_ZONES = 30         # Number of parks/green zones
```

### Zone Distribution for Green Spaces

```python
GREEN_ZONE_ZONE_SHARE = {
    "downtown": 0.25,        # 25% in downtown
    "residential": 0.4,      # 40% in residential
    "suburbs": 0.35          # 35% in suburbs
}
```

## 🎯 How It Works

### 1. Node Generation
- Generates random points using a combination of normal and uniform distributions
- Normal distribution creates clustering near the city center
- Uniform distribution adds spread-out suburban areas

### 2. Delaunay Triangulation
- Creates a "spider web" of non-overlapping connections
- Forms the basic structure of the street network

### 3. Graph Pruning
- Removes overly long edges (outliers)
- Randomly removes some edges to create city blocks
- Eliminates isolated nodes

### 4. Zone Assignment
- Assigns zones based on distance from city center and direction
- Downtown: < 0.4 units from center
- Industrial: Southwest quadrant
- Residential: Northeast quadrant
- Suburbs: Remaining areas

### 5. Amenity Placement
- **Hospitals**: Distributed across core, mid, and outer city regions with strategic spatial coverage
- **Green Zones**: Spread across zones with angle diversity to ensure even distribution

### 6. Road Network Creation
- Creates bidirectional edges (two-way streets)
- Ring roads identified at radius ~0.7-0.9
- Calculates realistic travel times based on distance and speed limits

## 📊 Graph Properties

The generated city graph includes:

**Node Attributes:**
- `x`, `y`: GPS coordinates (longitude, latitude)
- `zone`: Urban zone classification
- `color`: Visualization color
- `radial_distance`: Distance from city center
- `polar_angle`: Angle from city center
- `amenity`: Type of amenity (if applicable)
- `facility_name`: Name of facility (for hospitals)
- `green_zone`: Boolean flag for parks
- `park_name`, `park_type`: Green zone properties

**Edge Attributes:**
- `osmid`: Edge identifier
- `length`: Distance in meters
- `highway`: Road type (primary/residential)
- `name`: Street name
- `lanes`: Number of lanes
- `maxspeed`: Speed limit (km/h)
- `base_travel_time`: Base travel time
- `current_travel_time`: Current travel time
- `oneway`: Direction flag

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Akash Hedaoo**
- GitHub: [@Akash-Hedaoo](https://github.com/Akash-Hedaoo)

## 🙏 Acknowledgments

- NetworkX library for graph manipulation
- SciPy for Delaunay triangulation algorithms
- Inspired by real-world urban planning and GNN research

## 📧 Contact

For questions or feedback, please open an issue on the GitHub repository.

---

**Note**: This is a simulation project for educational and research purposes. The generated cities are synthetic and for demonstration of graph-based urban modeling techniques.
