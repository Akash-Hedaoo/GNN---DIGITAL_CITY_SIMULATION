# Macroscopic Traffic Simulation - Digital Twin

## 🎯 The Core Concept: "Faking Agents" Mathematically

**The Critical Question**: *If there are no cars, how do we know where the traffic goes?*

**The Answer**: We use **Macroscopic Simulation (Fluid Dynamics)** instead of **Microscopic Simulation (Individual Cars)**.

## 🔬 Engineering Breakdown

### 1. The "Pressure Model" - How Traffic is Simulated

Think of roads as **pipes** and traffic as **water pressure**.

**Real World Analogy**:
- When a pipe bursts → water backs up into feeding pipes
- When a road closes → traffic backs up onto feeding roads

**We don't need to track every water molecule to know the pipes will overflow!**

### Implementation in Code

```python
# Logic: Identify the closed edge
closed_edge = (u, v, key)

# Backlog: Find all edges pointing to the blocked node
predecessors = G.predecessors(u)

# Math: Multiply their travel time by a Congestion Factor
for pred in predecessors:
    current_travel_time *= congestion_multiplier  # 3x or 5x

# Ripple: Repeat for neighbors of neighbors
propagate_congestion(pred, depth + 1, decayed_multiplier)
```

**Result**: The dataset records that "Closing Road A causes massive delays on Roads B and C" - which is exactly what 1,000 agents rerouting would have caused!

### 2. Shortest Path Calculation (Dijkstra's Algorithm)

For the **Green Wave (Ambulance)** and **Demo Dashboard**, we calculate actual paths using **Dijkstra's Algorithm** (built into NetworkX).

```python
# Fast path calculation based on current traffic conditions
fast_path = nx.shortest_path(
    G, 
    source=start_node, 
    target=end_node, 
    weight='current_travel_time'
)
```

**How it works**:
- Looks at `current_travel_time` attribute of every road
- If GNN says a road takes 999 minutes (congested), Dijkstra automatically skips it
- Finds longer but faster route around the city

### 3. How Rerouting Information is Provided

**This is the beauty of the Digital Twin loop!**

Rerouting isn't a "message" sent to a driver - it's a **natural consequence** of the graph updating.

**Event Flow**:
1. **Event**: You close "Ring Road" in the Dashboard
2. **Update**: GNN updates edge weights: `Ring_Road.weight = Infinity`
3. **Consequence**: Next time we ask for a path, algorithm sees infinite weight and automatically reroutes through "Suburbs"

## 📊 The Mathematical "Trick"

| Phase | Method | Purpose |
|-------|--------|---------|
| **Training** | Probability Math (Ripple Effect) | Generate data FAST |
| **Demo** | Graph Algorithms (Dijkstra) | Show routes ACCURATELY for single vehicles |

## 🚀 Usage

### Step 1: Generate City Graph

```bash
python generate_complex_city.py
```

This creates `city_graph.graphml` with 800 nodes and ~4000 edges.

### Step 2: Run Demo Simulation

```bash
python macroscopic_traffic_simulation.py
```

**What happens**:
- Loads city graph
- Randomly closes a road
- Propagates congestion upstream (ripple effect)
- Simulates 30 minutes with road closed
- Reopens road and simulates 20 more minutes
- Exports training data to `demo_traffic_data.pkl`

### Step 3: Generate Training Data

```bash
python generate_training_data.py
```

**What happens**:
- Generates 100 different traffic scenarios
- Each scenario: 60 minutes of simulation
- Random road closures and reopenings
- Varied congestion patterns
- Exports comprehensive dataset to `gnn_training_data.pkl`

**Output**: 6000+ snapshots of traffic conditions for GNN training!

## 📈 Example Output

```
🚧 Closing road: 144 -> 14 (key=0)
  ↑ Backlog at depth 0: 4 upstream edges
     Edge 1->144: 509.3m -> 1528.0m (×3.0)
     Edge 14->144: 391.8m -> 1175.5m (×3.0)
     Edge 64->144: 552.7m -> 1658.0m (×3.0)
     Edge 297->144: 625.9m -> 1877.6m (×3.0)
    ↑ Backlog at depth 1: 8 upstream edges
       Edge 0->1: 753.9m -> 1809.3m (×2.4)
       Edge 2->1: 996.2m -> 2390.8m (×2.4)
       ...
```

**Statistics After 30 Minutes**:
```
📊 TRAFFIC STATISTICS
Simulation Time:     30.0 minutes
Total Network Delay: 138839.9 minutes
Closed Roads:        1
Average Congestion:  1.07x
Max Congestion:      2.96x
Congested Edges:     127/4030
```

## 🎓 Key Concepts

### Congestion Propagation

```python
# Depth 0: Directly blocked edges
multiplier = 3.0x

# Depth 1: One hop upstream
multiplier = 2.4x (3.0 * 0.7 decay)

# Depth 2: Two hops upstream
multiplier = 2.0x (2.4 * 0.7 decay)
```

### Recovery Model

```python
# Traffic gradually returns to normal
new_time = base_time + (current_time - base_time) * recovery_rate

# Recovery rate: 0.85 means 15% improvement each time step
```

### Random Events

```python
# Simulate accidents, construction, etc.
if random.random() < event_probability:
    apply_random_slowdown(1.2x to 2.0x)
```

## 🧠 Training a GNN

The generated data contains:

**Input Features (per edge)**:
- Base travel time
- Road type (highway, residential, etc.)
- Number of lanes
- Speed limit
- Connectivity (in-degree, out-degree)

**Output Labels**:
- Current travel time (congested)
- Congestion factor
- Whether edge is in ripple effect zone

**GNN Task**: Learn the function `f(closed_edges, graph_structure) → congestion_pattern`

## 🎯 Why This Works

1. **Mathematical Validity**: The ripple effect captures real traffic physics
2. **Computational Efficiency**: No need to simulate 100,000 individual vehicles
3. **Training Data Quality**: Creates diverse, realistic scenarios
4. **Real-time Prediction**: Trained GNN can predict in milliseconds

## 🔄 Digital Twin Loop

```
┌─────────────────────────────────────┐
│  1. Real Event (Road Closure)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. GNN Predicts Congestion         │
│     (using trained model)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Dashboard Shows Impact          │
│     (updated graph weights)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Dijkstra Finds New Routes       │
│     (automatic rerouting)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. Users See Green Wave Path       │
└─────────────────────────────────────┘
```

## 📦 Output Files

| File | Description | Size |
|------|-------------|------|
| `demo_traffic_data.pkl` | Single scenario demo | ~5 MB |
| `gnn_training_data.pkl` | 100 scenarios for training | ~500 MB |

## 🎨 Visualization

The simulation includes real-time statistics:
- Network-wide delay
- Average congestion factor
- Number of congested edges
- Closed roads count

## 🚦 Next Steps

1. **Train GNN Model**: Use PyTorch Geometric with the generated data
2. **Build Dashboard**: Flask/React app for real-time visualization
3. **Deploy Green Wave**: Ambulance routing using live predictions
4. **Expand Scenarios**: Add weather, events, time-of-day patterns

## 📚 References

- **Macroscopic Traffic Flow**: Lighthill-Whitham-Richards (LWR) Model
- **Graph Algorithms**: Dijkstra's Shortest Path
- **GNN Architecture**: Graph Convolutional Networks (GCN)
- **Digital Twin**: Cyber-physical systems integration

## 🎉 Summary

**No individual agents needed!** The macroscopic model captures traffic behavior through:
- Graph connectivity analysis
- Probability-based ripple effects
- Mathematical congestion propagation
- Algorithmic pathfinding

This is the foundation for your Digital Twin City - realistic traffic simulation without the computational overhead of tracking thousands of vehicles!
