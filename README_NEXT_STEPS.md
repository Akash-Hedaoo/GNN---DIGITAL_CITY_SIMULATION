# 📋 Next Steps Guide - CTM Implementation

## ✅ Implementation Status

**Cell Transmission Model (CTM) has been successfully implemented!**

The traffic simulation has been upgraded from a static ripple heuristic to a physics-based Cell Transmission Model. This provides:
- ✅ Realistic traffic flow dynamics
- ✅ Physical capacity constraints
- ✅ Natural spillback effects
- ✅ Time-stepped simulation
- ✅ BPR formula for congestion calculation

---

## 🚀 Step-by-Step Execution Guide

### **Prerequisites Check**

Before proceeding, ensure you have:

1. **Python Environment**: Virtual environment activated
   ```powershell
   . .\activate_venv.ps1
   ```

2. **Required Files**:
   - ✅ `real_city_processed.graphml` (from step2)
   - ✅ All dependencies installed (from requirements.txt)

3. **GPU Setup** (Optional but Recommended):
   - RTX 3050 6GB or compatible GPU
   - CUDA installed and working
   - Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 📊 Step 1: Generate Training Data with CTM

**Purpose**: Generate 2000 realistic traffic scenarios using Cell Transmission Model.

**Command**:
```powershell
python step3_generate_training_data.py
```

**What Happens**:
1. Loads the processed graph (`real_city_processed.graphml`)
2. Creates CTM links for each road segment
3. Generates 2000 scenarios with random road closures
4. Runs time-stepped simulations (45 minutes, 270 iterations each)
5. Converts vehicle density to congestion factors using BPR formula
6. Saves output to `gnn_training_data.pkl`

**Expected Output**:
```
🔄 Loading graph: real_city_processed.graphml...
   Caching static graph data...
   Generating 2000 CTM-based traffic scenarios...
   Simulation: 45 min, 10s steps, 270 iterations per scenario
   Memory optimization: Clearing cache every 250 scenarios
   
   [Progress bar showing generation progress]
   
💾 Saving dataset to gnn_training_data.pkl...
✅ Done! CTM dataset ready for training.
   Generated 2000 scenarios with physics-based traffic flow.
```

**Estimated Time**: 
- **With CPU**: 2-4 hours (depending on graph size)
- **With GPU**: Not used in this step (CPU-only operation)

**Memory Usage**: 
- Optimized for 6GB GPU systems
- Periodic garbage collection every 250 scenarios
- Peak RAM usage: ~2-4 GB (depends on graph size)

**Output File**: 
- `gnn_training_data.pkl` (~50-200 MB, depends on graph size)

---

## 🧠 Step 2: Train the GNN Model

**Purpose**: Train the Graph Attention Network (GATv2) model on the CTM-generated data.

**Command**:
```powershell
python step4_train_model.py
```

**What Happens**:
1. Loads the training dataset (`gnn_training_data.pkl`)
2. Creates TrafficGATv2 model architecture
3. Splits data: 80% training, 20% validation
4. Trains for 30 epochs with batch size 8 (optimized for 6GB GPU)
5. Validates after each epoch
6. Saves trained model to `real_city_gnn.pt`

**Expected Output**:
```
🔄 Loading Graph & Data to System RAM...
📦 Processing 2000 snapshots...
✅ GPU DETECTED: NVIDIA GeForce RTX 3050 (6.0 GB VRAM)

🔥 Starting Memory-Safe Training (30 Epochs)...

   Epoch 01 | Train Loss: 0.XXXX | Val Loss: 0.XXXX
   Epoch 02 | Train Loss: 0.XXXX | Val Loss: 0.XXXX
   ...
   Epoch 30 | Train Loss: 0.XXXX | Val Loss: 0.XXXX

✅ Training Complete in XXX.X seconds!
💾 Model saved to real_city_gnn.pt
```

**Estimated Time**: 
- **With RTX 3050 6GB**: 15-30 minutes
- **With CPU**: 2-4 hours (not recommended)

**Memory Usage**:
- GPU VRAM: ~4-5 GB (with batch size 8)
- System RAM: ~2-3 GB

**Output File**: 
- `real_city_gnn.pt` (~2-5 MB)

**Training Parameters** (from step4_train_model.py):
- Epochs: 30
- Batch Size: 8 (optimized for 6GB GPU)
- Learning Rate: 0.002
- Optimizer: Adam
- Loss Function: MSE (Mean Squared Error)

---

## 🔬 Step 3: Run What-If Analysis (Optional)

**Purpose**: Test the trained model with specific scenarios.

**Command**:
```powershell
python step5_whatif_analysis.py
```

**What Happens**:
1. Loads the trained model
2. Runs pre-defined scenarios:
   - Road closure simulation
   - Mall addition (Phoenix Market City)
   - Metro station addition
3. Generates impact reports

**Expected Output**:
```
🔄 SYSTEM: Initializing Digital Twin Engine...
   📂 Loading map from real_city_processed.graphml...
   🧠 Loading AI Brain from real_city_gnn.pt...
   ✅ AI Model Loaded.
   📊 Establishing Baseline Traffic State...

============================================================
🚦 RUNNING FINAL DEMO SCENARIOS
============================================================

📝 GENERATING REPORT: Closure: ...
   📉 Net Traffic Change: +X.XXXX%
   🔥 Impacted Road Segments: XXX
   ⚠️ Peak Bottleneck Spike: +X.XX

...
✅ SIMULATION COMPLETE.
```

---

## 🎨 Step 4: Visualize Results (Optional)

**Purpose**: Generate traffic heatmap visualization.

**Command**:
```powershell
python step6_visualize_results.py
```

**What Happens**:
1. Creates a scenario (e.g., new metro station)
2. Runs simulation
3. Generates congestion heatmap
4. Saves as PNG image

**Output File**: 
- `final_traffic_heatmap.png`

---

## 🌐 Step 5: Launch Web Application

**Purpose**: Interactive web interface for traffic simulation.

**Command**:
```powershell
python app.py
```

**Or using the launcher**:
```powershell
python run_app.py
```

**What Happens**:
1. Loads graph and trained model
2. Establishes baseline predictions
3. Starts Flask server on port 5000
4. Opens interactive map interface

**Access**: 
- Open browser: `http://localhost:5000`

**Features**:
- 🗺️ Interactive map with traffic visualization
- 🖱️ Click roads to select for closure
- 🤖 Run AI-powered simulations
- 📊 View detailed results and metrics
- 📍 Side-by-side map comparisons

---

## 📝 Complete Pipeline Summary

```powershell
# 1. Activate virtual environment
. .\activate_venv.ps1

# 2. Generate CTM training data (2-4 hours)
python step3_generate_training_data.py

# 3. Train GNN model (15-30 minutes)
python step4_train_model.py

# 4. (Optional) Run what-if analysis
python step5_whatif_analysis.py

# 5. (Optional) Generate visualization
python step6_visualize_results.py

# 6. Launch web application
python app.py
```

---

## ⚠️ Troubleshooting

### **Issue: Out of Memory (OOM) during training**

**Symptoms**: 
- CUDA out of memory error
- Process killed during training

**Solutions**:
1. **Reduce batch size** in `step4_train_model.py`:
   ```python
   BATCH_SIZE = 4  # Reduce from 8 to 4
   ```

2. **Reduce training data** in `step3_generate_training_data.py`:
   ```python
   NUM_SNAPSHOTS = 1000  # Reduce from 2000 to 1000
   ```

3. **Close other GPU applications**

4. **Use CPU training** (slower but works):
   - Model will automatically fall back to CPU if GPU unavailable

### **Issue: Training data generation is too slow**

**Solutions**:
1. **Reduce simulation time** in `step3_generate_training_data.py`:
   ```python
   SIMULATION_TIME_MINUTES = 30  # Reduce from 45
   TIME_STEP_SECONDS = 15  # Increase from 10 (fewer iterations)
   ```

2. **Reduce number of snapshots**:
   ```python
   NUM_SNAPSHOTS = 1000  # Reduce from 2000
   ```

### **Issue: Model loss not decreasing**

**Symptoms**:
- Training loss stays high
- Validation loss doesn't improve

**Solutions**:
1. **Check training data quality**:
   - Ensure step3 completed successfully
   - Verify `gnn_training_data.pkl` exists and has data

2. **Adjust learning rate** in `step4_train_model.py`:
   ```python
   optimizer = optim.Adam(model.parameters(), lr=0.001)  # Reduce from 0.002
   ```

3. **Increase epochs**:
   ```python
   EPOCHS = 50  # Increase from 30
   ```

### **Issue: Graph file not found**

**Error**: `FileNotFoundError: real_city_processed.graphml`

**Solution**:
1. Run step2 first: `python step2_feature_engineering.py`
2. Ensure you're in the correct directory
3. Check file exists: `ls real_city_processed.graphml`

### **Issue: Web app not loading**

**Error**: Model file not found or graph loading fails

**Solutions**:
1. Ensure `real_city_gnn.pt` exists (run step4 first)
2. Ensure `real_city_processed.graphml` exists (run step2 first)
3. Check Flask server logs for specific errors
4. Verify port 5000 is not in use:
   ```powershell
   netstat -ano | findstr :5000
   ```

---

## 📊 Expected File Sizes & Times

| File/Step | Size | Time (RTX 3050) | Time (CPU) |
|-----------|------|----------------|------------|
| `real_city_processed.graphml` | 5-20 MB | - | - |
| `gnn_training_data.pkl` | 50-200 MB | 2-4 hours | 3-5 hours |
| `real_city_gnn.pt` | 2-5 MB | 15-30 min | 2-4 hours |
| Step 3 (Generation) | - | N/A (CPU only) | 2-4 hours |
| Step 4 (Training) | - | 15-30 min | 2-4 hours |

---

## 🎯 Success Criteria

After completing all steps, you should have:

1. ✅ **Training Data**: `gnn_training_data.pkl` (2000 scenarios)
2. ✅ **Trained Model**: `real_city_gnn.pt` (GATv2 model weights)
3. ✅ **Working Web App**: Flask server running on port 5000
4. ✅ **Traffic Predictions**: Model generating realistic congestion factors

---

## 🔄 Next Development Steps (Future Enhancements)

After successful implementation, consider:

1. **Model Optimization**:
   - Hyperparameter tuning (learning rate, batch size, epochs)
   - Architecture modifications (hidden channels, attention heads)
   - Loss function improvements

2. **Data Enhancement**:
   - Increase training scenarios (3000-5000)
   - Add more diverse scenarios (weather, events, time-of-day)
   - Real-world traffic data integration

3. **Performance**:
   - Model quantization for faster inference
   - Batch inference optimization
   - Multi-GPU training support

4. **Features**:
   - Real-time traffic updates
   - Multiple scenario comparison
   - Export results (PDF, CSV)
   - Historical analysis
   - 3D visualization

5. **Production**:
   - Docker containerization
   - API deployment
   - Database integration
   - User authentication

---

## 📚 Additional Resources

- **CTM Theory**: See `CTM_IMPLEMENTATION_NOTES.md` for detailed implementation details
- **Web App Guide**: See `README_WEBAPP.md` for web interface documentation
- **Environment Setup**: See `README_VENV.md` for virtual environment instructions

---

## ✅ Verification Checklist

Before proceeding to production use, verify:

- [ ] Step 3 completed without errors
- [ ] Training data file exists and has reasonable size
- [ ] Step 4 training completed with decreasing loss
- [ ] Model file exists and loads correctly
- [ ] Web application starts without errors
- [ ] Traffic predictions are reasonable (congestion factors 1.0-10.0)
- [ ] Map visualization works correctly
- [ ] Simulation API responds correctly

---

**Status**: ✅ CTM Implementation Complete - Ready for Training

**Branch**: `EDI2025`

**Last Updated**: 2025-01-XX

