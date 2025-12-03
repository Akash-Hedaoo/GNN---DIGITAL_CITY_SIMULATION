# 🚀 How to Run

## Quick Start (3 Steps)

```powershell
# Step 1: Open terminal in project folder
cd F:\SEM3\EDI\GNN---DIGITAL_CITY_SIMULATION

# Step 2: Activate environment
.\nv\Scripts\Activate.ps1

# Step 3: Run server
python backend\app.py
```

Then open: **http://localhost:5000**

---

## One-Liner

```powershell
cd F:\SEM3\EDI\GNN---DIGITAL_CITY_SIMULATION; .\nv\Scripts\Activate.ps1; python backend\app.py
```

---

## Stop Server

Press `Ctrl+C` in terminal

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | Run: `pip install flask flask-cors torch torch-geometric networkx` |
| Port in use | Close other terminals or use: `python backend\app.py --port 5001` |
| Model not loaded | Check `trained_gnn.pt` exists |

---

## Project Structure

```
├── backend/app.py      ← Run this
├── frontend/           ← Web UI (auto-served)
├── gnn_model.py        ← Model architecture
├── trained_gnn.pt      ← Trained weights
└── city_graph.graphml  ← City data
```
