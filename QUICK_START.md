# 🚀 Quick Start Guide - Digital Twin City Simulation GUI

## 5-Minute Setup

### Step 1: Open Terminal
Navigate to the project folder:
```bash
cd E:\sem-3_subjects\EDI\GNN_DIGITAL_CITY_SIMULATION
```

### Step 2: Activate Virtual Environment
**Windows PowerShell:**
```powershell
.\twin-city-env\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
.\twin-city-env\Scripts\activate.bat
```

### Step 3: Launch Application
**Option A - Double-click:**
- Double-click `launch_gui.bat` (Windows)

**Option B - Command line:**
```bash
streamlit run streamlit_gui.py
```

### Step 4: Open Browser
The app will automatically open, or go to: **http://localhost:8501**

---

## What You'll See

### 📍 Left Sidebar
- **▶️ Run Simulation** button at top
- **Simulation Settings** - adjust speed
- **Visualization Layers** - toggle overlays

### 🗺️ Main Area (Center)
- **Map View Tab** - Interactive city network
- **Analytics Tab** - Single road closure testing
- **Experiments Tab** - Multiple road testing

### 📊 Right Panel
- **Metrics** - Real-time statistics
- **Inspector** - Node details
- **Logs** - System messages

---

## Quick Tests

### Test 1: Close a Single Road
1. Click **Analytics** tab
2. Use slider to select road number (e.g., 100)
3. Select "Close Road"
4. Click **🔮 Predict Impact**
5. View results and charts

### Test 2: Close Multiple Roads
1. Click **Experiments** tab
2. Choose "Manual Entry" or "Range Selection"
3. Enter road numbers (e.g., 100, 200, 300)
4. Click **🔮 Predict Combined Impact**
5. See comparative analysis

### Test 3: Compare Scenarios
1. Scroll down to **Advanced Analysis Tools**
2. Expand the section
3. Click **Scenario Comparison** tab
4. Select scenarios to compare
5. Click **📊 Compare Scenarios**

---

## Tips

### 💡 Navigation
- Use tabs to switch between different tools
- Expand/collapse sections with ▶️ arrows
- Hover over metrics for more info

### ⚡ Performance
- First load may take 10-20 seconds (model loading)
- Subsequent predictions are instant
- GPU automatically used if available

### 🎨 Customization
- Dark theme enabled by default
- Adjust in `.streamlit/config.toml`
- Modify colors, fonts, layout

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | Run: `streamlit run streamlit_gui.py --server.port 8502` |
| Model not found | Ensure `trained_gnn.pt` exists |
| Graph not loaded | Check `city_graph.graphml` is present |
| Import errors | Run: `pip install streamlit plotly networkx torch` |

---

## Keyboard Shortcuts
- **R** - Rerun app
- **C** - Clear cache
- **M** - Toggle sidebar
- **Ctrl+C** - Stop server (in terminal)

---

## Next Steps

1. ✅ **Explore** different tabs and features
2. ✅ **Test** various road closure scenarios
3. ✅ **Analyze** traffic patterns
4. ✅ **Compare** before/after states
5. ✅ **Customize** settings to your needs

---

## Getting Help

- Check **Logs** tab for error messages
- Review `STREAMLIT_GUI_README.md` for detailed docs
- Look at `PENDING_WORK.md` for known issues

---

**Happy Exploring! 🚦**

*The application saves your preferences and remembers your last session.*
