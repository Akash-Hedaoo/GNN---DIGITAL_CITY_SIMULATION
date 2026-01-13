# Cell Transmission Model (CTM) Implementation

## Overview

The training data generation (`step3_generate_training_data.py`) has been refactored from a static BFS ripple heuristic to a physics-based **Cell Transmission Model (CTM)** simulation.

## Key Changes

### 1. **Physics-Based Traffic Flow**
- **Old**: Static multiplier-based congestion propagation
- **New**: Time-stepped simulation with supply/demand flow dynamics

### 2. **Core Components**

#### **CTMLink Class**
- Tracks vehicle density (`n_vehicles`) on each road segment
- Calculates **Sending Flow** (vehicles wanting to leave)
- Calculates **Receiving Flow** (vehicles that can enter)
- Enforces physical capacity limits (Jam Density)

#### **Fundamental Diagram**
- **Free Flow**: Flow increases linearly with density
- **Capacity**: Maximum flow at critical density
- **Congestion**: Flow decreases as density approaches jam density

#### **BPR Formula Conversion**
- Converts vehicle density to congestion factor
- Formula: `congestion_factor = 1.0 + 0.15 * (density/capacity)^4`
- Ensures output format matches GNN training expectations

### 3. **Simulation Parameters**

```python
JAM_DENSITY = 150.0          # veh/km/lane
SATURATION_FLOW = 1800.0     # veh/hr/lane
SIMULATION_TIME = 45 minutes  # Total duration
TIME_STEP = 10 seconds        # Resolution
SIMULATION_STEPS = 270        # Total iterations per scenario
```

### 4. **Memory Optimizations for RTX 3050 6GB**

- **Reduced simulation time**: 45 minutes (was 60)
- **Larger time steps**: 10 seconds (was 6) → fewer iterations
- **Periodic garbage collection**: Every 250 scenarios
- **Efficient data structures**: Minimal intermediate objects

## Output Format

The CTM simulation produces the **same output format** as before:

```python
SimpleNamespace(
    edge_travel_times: dict[(u,v,k)] -> float,
    edge_congestion: dict[(u,v,k)] -> float,
    closed_edges: list[(u,v,k)],
    node_populations: dict[node] -> float
)
```

This ensures **100% compatibility** with `step4_train_model.py`.

## Advantages Over Ripple Heuristic

1. **Physical Realism**: Enforces actual capacity constraints
2. **Spillback**: Queues build backward naturally when roads fill up
3. **Time Evolution**: Traffic patterns evolve over time, not instant
4. **No Absurd Values**: Cannot exceed physical road capacity
5. **Metro Immunity**: Metro system remains unaffected by road traffic

## Usage

```bash
# Generate training data with CTM
python step3_generate_training_data.py

# Validate implementation
python validate_ctm.py

# Train model (unchanged)
python step4_train_model.py
```

## Performance Notes

- **Generation Time**: ~2-3x slower than ripple heuristic (due to time-stepping)
- **Memory Usage**: Optimized for 6GB GPU systems
- **Quality**: Significantly more realistic traffic patterns

## Technical Details

### Flow Distribution Logic

1. **Initialize**: Create CTMLink for each graph edge
2. **Apply Closures**: Set capacity to 0 for closed edges
3. **Time Loop** (270 iterations):
   - Calculate sending/receiving flows at each node
   - Distribute flow proportionally to receiving capacity
   - Update vehicle counts on links
4. **Convert**: Map final density to congestion factor (BPR)

### Node Processing

For each intersection node:
- **Incoming edges**: Calculate total sending flow
- **Outgoing edges**: Calculate total receiving capacity
- **Actual flow**: `min(sending, receiving)` (CTM rule)
- **Distribution**: Proportional to receiving capacity

## Validation

Run `validate_ctm.py` to verify:
- ✅ Output structure matches expected format
- ✅ Congestion factors are reasonable (1.0 - 100.0)
- ✅ All edges have valid travel times
- ✅ Closed edges are properly tracked

---

**Status**: ✅ Ready for training
**Compatibility**: ✅ Fully compatible with existing training pipeline
**Memory**: ✅ Optimized for RTX 3050 6GB


