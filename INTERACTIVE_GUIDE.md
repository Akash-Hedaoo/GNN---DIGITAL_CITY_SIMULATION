# Interactive Traffic Simulation - User Guide

## 🎮 Overview

You now have **two ways** to run the traffic simulation with user input:

### 1. Demo Mode (Simple)
**File**: `macroscopic_traffic_simulation.py`

Run with: `python macroscopic_traffic_simulation.py`

**Features**:
- Interactive road selection at start
- Choose from list, random, or custom node input
- Runs single scenario with 30-minute simulation
- Exports demo training data

### 2. Interactive Mode (Full Control)
**File**: `interactive_traffic_sim.py`

Run with: `python interactive_traffic_sim.py`

**Features**:
- Full menu-driven interface
- Block/unblock multiple roads dynamically
- View real-time statistics
- Step-by-step or auto-run simulation
- Path finding with current traffic
- Save training data anytime

---

## 🚀 Quick Start

### Demo Mode

```bash
# Activate environment
.\twin-city-env\Scripts\Activate.ps1

# Run demo
python macroscopic_traffic_simulation.py
```

**When prompted:**

```
🛣️  ROAD SELECTION MENU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sample roads (showing first 20):
#     From       To         Length(km)   Speed(km/h)
1     144        14         0.50         50.0
2     297        64         0.75         40.0
...

💡 Options:
   1. Enter road number (1-20) from the list above
   2. Enter 'r' or 'random' for random road closure
   3. Enter 'custom' to specify source and target nodes
   4. Enter 'q' or 'quit' to skip road closure

👉 Your choice: 
```

**Input Options:**
- `5` - Block road #5 from the list
- `r` - Random road selection
- `custom` - Enter specific nodes
- `q` - Skip blocking (run with no closures)

---

### Interactive Mode

```bash
python interactive_traffic_sim.py
```

**Main Menu:**

```
╔════════════════════════════════════════════════════════════════════╗
║                         📋 MAIN MENU                               ║
╚════════════════════════════════════════════════════════════════════╝

1️⃣  View Roads (list all roads)
2️⃣  Block Road (close a road)
3️⃣  Unblock Road (reopen a road)
4️⃣  Show Statistics (current traffic state)
5️⃣  Run Simulation (advance time)
6️⃣  Auto-Run (simulate 30 minutes)
7️⃣  Find Path (calculate shortest route)
8️⃣  Reset Simulation
0️⃣  Exit

👉 Select option:
```

---

## 📋 Feature Details

### 1. View Roads
Shows paginated list with:
- Road number
- Source and destination nodes
- Length and speed limit
- Current status (🟢 Clear, 🟡 Medium, 🔴 Heavy, 🚧 Closed)

Navigate with: `n` (next), `p` (previous), `q` (quit)

### 2. Block Road
Three ways to block:

**A) By Number:**
```
👉 Enter choice: 5
✅ Road 144 → 14 blocked!
```

**B) Random:**
```
👉 Enter choice: random
🎲 Random: 297 → 64
✅ Road blocked!
```

**C) Custom Nodes:**
```
👉 Enter choice: custom
   Source node: 100
   Target node: 200
✅ Road 100 → 200 blocked!
```

### 3. Unblock Road
View currently blocked roads and reopen:

```
🚧 Currently Blocked Roads:
1. Road 144 → 14
2. Road 297 → 64

👉 Enter road # to unblock (or 'all'): 1
✅ Road 144 → 14 reopened!
```

### 4. Show Statistics
Displays current traffic state:

```
📊 TRAFFIC STATISTICS
Simulation Time:     15.0 minutes
Total Network Delay: 85,432.3 minutes
Closed Roads:        2
Average Congestion:  1.15x
Max Congestion:      4.50x
Congested Edges:     89/4030
```

### 5. Run Simulation
Advance time step-by-step:

```
⏱️  Minutes to simulate (default=1): 5
✅ Simulated 5 minute(s)
```

### 6. Auto-Run
Run for extended period:

```
⏱️  Duration in minutes (default=30): 60
🚀 Running simulation for 60 minutes...

--- Minute 10/60 ---
Network delay: 125,432.1m
Congested edges: 145/4030
...
```

### 7. Find Path
Calculate shortest route considering traffic:

```
🗺️  Path Finder
   Start node: 100
   End node: 500

✅ Path found!
   Nodes: 23
   Route: 100 → 144 → 297 → 64 → 68...
   Travel time: 45.7 minutes
```

### 8. Reset Simulation
Clears all blocks and starts fresh:

```
⚠️  Reset simulation? (yes/no): yes
✅ Simulation reset!
```

---

## 🎯 Example Usage Scenarios

### Scenario 1: Test Single Road Closure

1. Run `python macroscopic_traffic_simulation.py`
2. Enter `5` to select road #5
3. Watch congestion propagate
4. See statistics after 30 minutes

### Scenario 2: Multiple Road Closures

1. Run `python interactive_traffic_sim.py`
2. Select option `2` (Block Road)
3. Block 3-4 different roads
4. Select option `4` to see statistics
5. Select option `6` to auto-run 30 minutes
6. Select option `3` to unblock roads one by one

### Scenario 3: Emergency Routing

1. Run interactive mode
2. Block several major roads
3. Use option `7` (Find Path) to calculate ambulance route
4. System automatically avoids blocked/congested roads
5. Compare with original route before closure

---

## 💡 Tips & Tricks

### Road Selection
- **Major roads**: Usually have higher node IDs (e.g., 400-600)
- **Ring roads**: Look for roads with high speed limits
- **Downtown**: Nodes near city center (check x, y coordinates)

### Congestion Patterns
- **Immediate effect**: 3x increase on upstream roads (depth 0)
- **Ripple effect**: 2.4x at depth 1, 2.0x at depth 2
- **Recovery**: Gradual (85% per time step by default)

### Path Finding
- Dijkstra algorithm automatically avoids closed roads
- Uses `current_travel_time` as edge weight
- Closed roads have weight = 999 (virtually infinite)

### Training Data
- More closures = more training scenarios
- Block important roads for higher impact
- Run longer simulations for temporal patterns

---

## 🎨 Visualization Tips

**Status Indicators:**
- 🟢 Clear: Congestion < 1.5x
- 🟡 Medium: Congestion 1.5-2.0x
- 🔴 Heavy: Congestion > 2.0x
- 🚧 Closed: Road blocked

**Congestion Factor:**
- 1.0x = Normal (base travel time)
- 2.0x = Double the normal time
- 3.0x = Triple (heavy congestion)

---

## 🔧 Advanced Options

### Custom Configuration

Edit the `SimulationConfig` in either script:

```python
config = SimulationConfig(
    base_congestion_multiplier=3.0,   # Initial impact (try 2.0-5.0)
    ripple_decay=0.7,                 # Decay rate (try 0.5-0.9)
    ripple_depth=3,                   # Propagation hops (try 2-5)
    random_event_probability=0.01     # Random slowdowns (try 0.0-0.1)
)
```

### Export Training Data

In interactive mode:
1. Press `0` to exit
2. Answer `yes` to save
3. Enter filename (e.g., `my_scenario.pkl`)

---

## 📊 Output Files

| File | Created By | Content |
|------|------------|---------|
| `demo_traffic_data.pkl` | Demo mode | Single scenario |
| `interactive_data.pkl` | Interactive mode | Custom scenarios |
| `test_training_data.pkl` | Test script | 5 scenarios |
| `gnn_training_data.pkl` | Full generator | 100 scenarios |

---

## 🐛 Troubleshooting

**Problem**: "city_graph.graphml not found"
```bash
# Solution: Generate the city first
python generate_complex_city.py
```

**Problem**: "No road exists from X to Y"
```
# Solution: View available roads first
# Option 1 in interactive mode, or check with:
python -c "import networkx as nx; G=nx.read_graphml('city_graph.graphml'); print(list(G.edges())[:20])"
```

**Problem**: Cannot enter node IDs
```
# Solution: Nodes might be strings in GraphML
# Try without quotes: 100
# Or with quotes: "100"
```

---

## 🎓 Understanding the Simulation

### When you block a road:

1. **Immediate**: Road weight → 999 minutes (virtually closed)
2. **Depth 0**: Direct upstream roads × 3.0
3. **Depth 1**: Second-level upstream × 2.4 (3.0 × 0.7 decay)
4. **Depth 2**: Third-level upstream × 2.0 (2.4 × 0.7 decay)

### Over time:

- **Random events**: Small slowdowns occur (accidents, etc.)
- **Recovery**: Congestion gradually clears (×0.85 each step)
- **Equilibrium**: System reaches steady state

### Path finding:

- Always uses **current** traffic conditions
- Automatically reroutes around problems
- Returns **realistic** travel times

---

## 🚀 Next Steps

1. **Test the system**: Run both modes to understand behavior
2. **Generate data**: Create diverse scenarios for GNN training
3. **Train GNN**: Use the `.pkl` files with PyTorch Geometric
4. **Build dashboard**: Integrate with web interface
5. **Deploy**: Real-time traffic prediction!

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Demo with user input | `python macroscopic_traffic_simulation.py` |
| Full interactive | `python interactive_traffic_sim.py` |
| Generate training data | `python generate_training_data.py` |
| Quick test | `python test_training_generation.py` |

**Enjoy the Interactive Traffic Simulator!** 🚦🌆
