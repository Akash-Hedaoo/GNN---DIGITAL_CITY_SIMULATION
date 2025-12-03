# GNN City Simulator - Modern UI Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     GNN City Simulator Platform                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────┬──────────────────────────────┐   │
│  │     WEB BROWSER (Frontend)       │    FLASK SERVER (Backend)    │   │
│  ├──────────────────────────────────┼──────────────────────────────┤   │
│  │                                  │                              │   │
│  │  ┌────────────────────────────┐  │  ┌──────────────────────┐   │   │
│  │  │   HTML5 (index.html)       │  │  │  Flask App (app.py)  │   │   │
│  │  │  - Semantic Structure      │  │  │  - Route Handlers    │   │   │
│  │  │  - 25/75 Layout            │  │  │  - CORS Support      │   │   │
│  │  │  - Modal dialogs           │  │  │  - JSON Responses    │   │   │
│  │  └────────────────────────────┘  │  └──────────────────────┘   │   │
│  │           ↓                       │           ↓                 │   │
│  │  ┌────────────────────────────┐  │  ┌──────────────────────┐   │   │
│  │  │  CSS3 (styles.css)         │  │  │ ModelWrapper         │   │   │
│  │  │  - Dark Theme              │  │  │  (model.py)          │   │   │
│  │  │  - Flexbox Grid            │  │  │ - Load torch model   │   │   │
│  │  │  - Animations              │  │  │ - Fallback predict   │   │   │
│  │  │  - Responsive              │  │  │ - Normalize features │   │   │
│  │  └────────────────────────────┘  │  └──────────────────────┘   │   │
│  │           ↓                       │           ↓                 │   │
│  │  ┌────────────────────────────┐  │  ┌──────────────────────┐   │   │
│  │  │ JavaScript (app.js)        │  │  │  GNN Model           │   │   │
│  │  │ - Map initialization       │  │  │  (trained_gnn.pt)    │   │   │
│  │  │ - Event handlers           │  │  │ - Graph Neural Net   │   │   │
│  │  │ - Data visualization       │  │  │ - Traffic prediction │   │   │
│  │  │ - API calls                │  │  └──────────────────────┘   │   │
│  │  └────────────────────────────┘  │                              │   │
│  │           ↓                       │                              │   │
│  │  ┌────────────────────────────┐  │  ┌──────────────────────┐   │   │
│  │  │ Utils (utils.js)           │  │  │  Dependencies        │   │   │
│  │  │ - Helper functions         │  │  │ - PyTorch            │   │   │
│  │  │ - Data processing          │  │  │ - torch-geometric    │   │   │
│  │  │ - Geographic calculations  │  │  │ - numpy/networkx     │   │   │
│  │  │ - Storage utilities        │  │  └──────────────────────┘   │   │
│  │  └────────────────────────────┘  │                              │   │
│  │                                  │                              │   │
│  │  EXTERNAL LIBRARIES:             │                              │   │
│  │  - Leaflet.js (mapping)          │                              │   │
│  │  - Chart.js (analytics)          │                              │   │
│  │                                  │                              │   │
│  └──────────────────────────────────┴──────────────────────────────┘   │
│                              ↕↕ HTTP API                               │
│         http://localhost:8000 (Frontend)                               │
│         http://localhost:5000 (Backend)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Frontend Component Hierarchy

```
app-container (flex layout)
│
├─ control-panel (25%)
│  ├─ panel-header
│  │  ├─ h1 (City Simulator)
│  │  └─ .subtitle (What-if Analysis Engine)
│  │
│  └─ panel-content (scrollable)
│     ├─ control-section (Scenario Control)
│     │  ├─ control-group
│     │  │  ├─ label (Scenario Type)
│     │  │  └─ select#scenarioType
│     │  ├─ control-group
│     │  │  ├─ label (Severity Level)
│     │  │  ├─ input#severity (range)
│     │  │  └─ span#severityValue
│     │  └─ control-group
│     │     ├─ label (Duration)
│     │     └─ input#duration (number)
│     │
│     ├─ control-section (Selection)
│     │  └─ .selection-info (display selected node)
│     │
│     ├─ control-section (Actions)
│     │  ├─ button#runScenario
│     │  └─ button#resetScenario
│     │
│     └─ control-section (Results)
│        ├─ .quick-stats
│        │  ├─ .stat-item
│        │  └─ .stat-item
│        └─ button#detailedAnalysis
│
├─ map-container (75%)
│  ├─ div#map (Leaflet container)
│  └─ .map-legend
│     ├─ .legend-item (Roads)
│     ├─ .legend-item (Metro)
│     ├─ .legend-item (Amenities)
│     └─ .legend-item (Congestion)
│
└─ modal#analysisModal
   ├─ .modal-header
   │  ├─ h2 (Statistical Analysis)
   │  └─ button#closeAnalysis
   │
   └─ .modal-body
      └─ .analysis-charts
         ├─ .chart-container
         │  └─ canvas#comparisonChart
         ├─ .chart-container
         │  └─ canvas#distributionChart
         └─ .stats-table
            └─ <table with metrics>
```

## Data Flow Diagram

```
User Interaction
      ↓
  [Scenario Form]
  [Severity Slider]
  [Duration Input]
      ↓
    Click "Run"
      ↓
┌─────────────────────────────┐
│ POST /predict (baseline)    │
│ {features: [...]}           │
└──────────────┬──────────────┘
               ↓
        predictions_before: [...]
               ↓
┌─────────────────────────────┐
│ POST /whatif (scenario)      │
│ {features: [...],           │
│  scenario: {...}}           │
└──────────────┬──────────────┘
               ↓
      {before: [...],
       after: [...]}
               ↓
      [Calculate Stats]
               ↓
    ┌─────────────┬──────────────┐
    ↓             ↓              ↓
[Update Map]  [Quick Stats]  [Modal Data]
[Colors]      [Numbers]     [Charts/Table]
    ↓             ↓              ↓
 Rendered     Displayed      Popup
```

## API Endpoint Contract

### 1. GET /health
```
Request:  GET /health
Response: {
  "status": "ok"
}
Status:   200 OK
```

### 2. GET /city-data
```
Request:  GET /city-data
Response: {
  "nodes": [
    {"id": 1, "lat": 40.7128, "lng": -74.0060, "name": "..."}
  ],
  "edges": [
    {"id": 1, "source": 1, "target": 2, "flow": 450, "congestion": 0.3}
  ],
  "amenities": [
    {"id": "h1", "lat": 40.714, "lng": -74.003, "name": "...", "type": "hospital"}
  ],
  "metros": [
    {"id": "m1", "lat": 40.7128, "lng": -74.006, "name": "...", "line": "Red"}
  ]
}
Status:   200 OK
```

### 3. POST /predict
```
Request:  {
  "features": [0.3, 0.4, 0.2, 0.5, 0.1]
}
Response: {
  "predictions": [8.44, 7.58, 4.21, ...]
}
Status:   200 OK
```

### 4. POST /whatif
```
Request:  {
  "features": [0.3, 0.4, 0.2, 0.5, 0.1],
  "scenario": {
    "type": "road_closure",
    "severity": 0.7,
    "duration": 30
  }
}
Response: {
  "before": [8.44, 7.58, 4.21, ...],
  "after": [9.21, 8.12, 5.43, ...]
}
Status:   200 OK
```

## File Structure

```
GNN---DIGITAL_CITY_SIMULATION/
│
├── backend/
│   ├── app.py                 (Flask server)
│   ├── model.py               (Model wrapper)
│   ├── __init__.py            (Package marker)
│   └── requirements.txt        (Dependencies)
│
├── frontend/
│   ├── index.html             (HTML structure)
│   ├── styles.css             (Styling - 750 lines)
│   ├── app.js                 (Logic - 550 lines)
│   └── utils.js               (Utilities - 25+ functions)
│
├── Documentation/
│   ├── MODERN_UI_README.md            (Features & integration)
│   ├── MODERN_UI_QUICKSTART.md        (Quick start guide)
│   ├── IMPLEMENTATION_SUMMARY.md      (This implementation)
│   ├── ARCHITECTURE.md                (This file)
│   └── start-ui.bat                   (Startup script)
│
├── Testing/
│   └── test_integration.py    (7 integration tests)
│
├── Configuration/
│   ├── .gitignore             (Git ignore rules)
│   ├── .venv/                 (Virtual environment)
│   └── trained_gnn.pt         (Model weights)
│
└── city_graph.graphml         (City graph data)
```

## State Management

```
state = {
  map: <Leaflet Map>,
  currentScenario: {type, severity, duration},
  scenarioResults: {before, after},
  selectedNode: {id, type},
  layers: {
    roads: <FeatureGroup>,
    metros: <FeatureGroup>,
    amenities: <FeatureGroup>,
    heatmap: null
  },
  cityData: {nodes, edges, amenities, metros},
  predictionsBefore: [...]
  predictionsAfter: [...]
}
```

## Execution Flow

```
1. Page Load
   → DOMContentLoaded event
   → initializeMap() - Create Leaflet map
   → loadCityData() - Fetch from /city-data or use mock
   → renderCityLayers() - Draw roads, metros, amenities
   → setupEventListeners() - Attach handlers
   → checkBackendHealth() - Verify /health endpoint

2. User Input
   → User adjusts scenario form inputs
   → severityValue label updates in real-time

3. Run Scenario
   → Collect inputs from form
   → Collect features from current map state
   → POST /predict → Get baseline predictions
   → POST /whatif → Get impact predictions
   → Calculate statistics
   → updateMapVisualization() - Change road colors
   → updateQuickStats() - Update statistics display

4. View Analysis
   → User clicks "Detailed Analysis"
   → Modal becomes visible
   → renderAnalysisCharts() - Create Chart.js charts
   → renderStatsTable() - Create metrics table
   → Display before/after metrics

5. Reset
   → Clear scenario state
   → Reset map to original state
   → Clear quick statistics
```

## Performance Optimization Strategy

```
Memory:
  - Feature groups cleared before re-rendering
  - Chart instances destroyed when modal closes
  - Event listeners debounced (300ms)

Network:
  - City data cached after first load
  - API responses are JSON (compact)
  - No unnecessary polling

Rendering:
  - Leaflet handles efficient map rendering
  - Chart.js optimized for canvas
  - CSS animations use GPU acceleration
  - Flexbox for efficient layout

Code:
  - Vanilla JavaScript (no framework overhead)
  - Minimal library usage (Leaflet + Chart.js)
  - Debounced event handlers
  - Efficient DOM queries
```

## Browser Rendering Pipeline

```
1. Parse HTML → DOM Tree
2. Parse CSS → CSSOM
3. Combine → Render Tree
4. Layout → Calculate positions
5. Paint → Rasterize pixels
6. Composite → Layers

Leaflet & Chart.js:
- Use canvas for efficient rendering
- Avoid layout thrashing
- Batch DOM updates
```

## Security Considerations

```
✓ CORS enabled (frontend accessible)
✓ No authentication (local development)
✓ Input validation on backend
✓ JSON serialization prevents XSS
✓ No sensitive data exposed

For Production:
- Add authentication (JWT/OAuth)
- Implement rate limiting
- Add input sanitization
- Use HTTPS/TLS
- Set strict CSP headers
```

## Accessibility Features

```
✓ Semantic HTML (nav, main, aside, section)
✓ Color contrast (WCAG AA standard)
✓ Font sizes readable
✓ Interactive elements keyboard accessible
✓ Clear labels on form inputs
✓ Modal has proper role and focus management
```

## Mobile Responsive Breakpoints

```
Desktop (1024px+):
  - 25% sidebar | 75% map
  - All controls visible
  - Full feature set

Tablet (768-1024px):
  - 30% sidebar | 70% map
  - Adjusted spacing

Mobile (<768px):
  - Stacked layout
  - 40% control panel
  - 60% map
  - Touch-optimized buttons
```

## Deployment Architecture (Recommended)

```
Production:
  ┌──────────────────────────────────────┐
  │           CloudFlare CDN             │
  │     (Frontend assets + caching)      │
  └──────────────┬───────────────────────┘
                 │
  ┌──────────────▼───────────────────────┐
  │         Nginx (Reverse Proxy)        │
  │  - Load balancing                    │
  │  - HTTPS/TLS termination             │
  │  - Gzip compression                  │
  └──────────────┬───────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼─────────────┐  ┌────────▼────────┐
│  Flask Backend  │  │  Flask Backend   │
│  (Instance 1)   │  │  (Instance 2)    │
│  (Port 5000)    │  │  (Port 5001)     │
└────────┬────────┘  └────────┬─────────┘
         │                    │
    ┌────▼────────────────────▼───┐
    │   PostgreSQL Database       │
    │   (Redis Cache)             │
    │   (ML Model Storage)        │
    └─────────────────────────────┘
```

## Summary

The **Modern UI Architecture** provides:

✅ **Clean Separation**: Frontend/Backend clearly separated  
✅ **Scalable**: Easy to add features and endpoints  
✅ **Maintainable**: Well-organized, documented code  
✅ **Performant**: Optimized rendering and network  
✅ **Secure**: CORS enabled, input validation  
✅ **Accessible**: Semantic HTML, WCAG compliant  
✅ **Responsive**: Works on all devices  
✅ **Production-Ready**: Can scale to production  

---

**Created**: 2024  
**Status**: ✅ Complete  
**Commits**: 4  
**Branch**: akash
