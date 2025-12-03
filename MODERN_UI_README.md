# Modern UI - GNN City Simulator

## Overview
This is the redesigned, modern web-based interface for the GNN Digital City Simulator. It features a professional, interactive dashboard with real-time traffic prediction, dynamic map visualization, and advanced scenario analysis tools.

## Features

### 🗺️ Interactive Map (75% of screen)
- **Dynamic City Visualization**: Real-time rendering of road networks, metro lines, and amenities
- **Congestion Heatmap**: Color-coded roads showing traffic intensity
  - 🟢 Green: Light traffic (< 20%)
  - 🟡 Yellow: Moderate (40-60%)
  - 🔴 Red: Heavy congestion (> 80%)
- **Layered Visualization**:
  - Road networks with dynamic width and color
  - Metro stations and lines
  - Public amenities (hospitals, schools, parks, restaurants)
- **Interactive Selection**: Click roads, metros, or amenities to view details
- **Zoom & Pan**: Smooth navigation with Leaflet.js

### 🎮 Control Panel (25% of screen)
- **Scenario Control Section**
  - Scenario type selector (Road Closure, Metro Impact, Traffic Event, Infrastructure Change)
  - Severity slider (1-10)
  - Duration input (1-240 minutes)
- **Selection Information**
  - View details of selected map elements
  - Real-time updates
- **Action Buttons**
  - Run Scenario: Execute what-if analysis
  - Reset: Return to baseline state
- **Quick Statistics**
  - Average congestion before/after
  - Congestion change indicator
  - Updates after scenario execution

### 📊 Analysis Modal
- **Comparison Charts**
  - Before/After congestion comparison (bar chart)
  - Distribution changes across road segments (line chart)
- **Impact Summary Table**
  - Peak congestion changes
  - Minimum congestion levels
  - Detailed metrics breakdown
- **Color-Coded Changes**
  - Red for negative impacts
  - Green for improvements

## Architecture

### Frontend Files

#### `index.html`
- Semantic HTML5 structure
- 25/75 flex layout (sidebar + map)
- Leaflet.js integration
- Chart.js integration
- Modal dialog for analysis

#### `styles.css`
- Modern dark theme with gradients
- CSS custom properties (variables)
- Flexbox layout system
- Smooth animations and transitions
- Responsive design (mobile-friendly)
- Custom scrollbar styling
- Leaflet control customization

#### `app.js`
Main application logic:
- **Map Management**: Initialize Leaflet, load tile layers, manage feature groups
- **Data Loading**: Fetch city graph from backend or use mock data
- **Rendering**: Draw roads, metros, amenities with interactive popups
- **Scenario Execution**: Collect inputs, call `/predict` and `/whatif` endpoints
- **Visualization Updates**: Dynamically update road colors based on predictions
- **Analysis Rendering**: Populate charts and tables using Chart.js

#### `utils.js`
Utility functions and helpers:
- Formatting utilities (percentage, numbers, time)
- Geographic calculations (distance, bounds)
- Data processing (normalization, aggregation)
- Local storage management
- API helper for common HTTP operations
- Array manipulation and grouping

### Backend Endpoints

#### GET `/health`
Status check endpoint
```bash
curl http://localhost:5000/health
# Response: {"status": "ok"}
```

#### GET `/city-data`
Returns city graph for map visualization
```bash
curl http://localhost:5000/city-data
# Response: {
#   "nodes": [...],
#   "edges": [...],
#   "amenities": [...],
#   "metros": [...]
# }
```

#### POST `/predict`
ML inference on traffic features
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.3, 0.4, 0.2, 0.5, 0.1]}'
# Response: {"predictions": [8.4442, 7.5795, 4.2057, ...]}
```

#### POST `/whatif`
What-if scenario analysis (before/after)
```bash
curl -X POST http://localhost:5000/whatif \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.3, 0.4, 0.2, 0.5, 0.1],
    "scenario": {
      "type": "road_closure",
      "severity": 0.7,
      "duration": 30
    }
  }'
# Response: {
#   "before": [8.4442, 7.5795, ...],
#   "after": [9.2103, 8.1234, ...]
# }
```

## How to Use

### Starting the Application

1. **Ensure Backend is Running**
   ```powershell
   # Navigate to project root
   cd "c:\Users\Akash\Desktop\EDI\Edi proj\GNN---DIGITAL_CITY_SIMULATION"
   
   # Activate virtual environment
   .\.venv\Scripts\Activate.ps1
   
   # Start Flask server
   cd backend
   python -m flask run --host 0.0.0.0 --port 5000
   ```

2. **Start Frontend Server**
   ```powershell
   # In another terminal, navigate to frontend
   cd frontend
   
   # Use Python's built-in server
   python -m http.server 8000
   
   # OR use Node.js http-server if installed
   # npx http-server -p 8000
   ```

3. **Open in Browser**
   - Navigate to: `http://localhost:8000`

### Running a Scenario

1. **Select Scenario Type** from dropdown
2. **Adjust Severity** using slider (1-10)
3. **Set Duration** in minutes
4. **Optional**: Click on map to select specific roads/amenities
5. **Click "Run Scenario"** to execute
6. **Watch Map Update** in real-time with new congestion values
7. **View Quick Stats** in control panel
8. **Click "Detailed Analysis"** for comprehensive charts and metrics

## Design Specifications

### Color Scheme
- **Primary**: `#1e88e5` (Blue) - Interactive elements
- **Accent**: `#ff6b6b` (Red) - Warnings and after-scenario
- **Success**: `#4CAF50` (Green) - Positive changes
- **Warning**: `#FFC107` (Yellow) - Medium congestion
- **Danger**: `#FF5722` (Orange) - High congestion
- **Background**: `#0f1419` (Dark) - Main background
- **Surface**: `#1a1f2e` (Darker Blue) - Cards and panels

### Typography
- **Font Family**: System UI (-apple-system, BlinkMacSystemFont, Segoe UI, Roboto)
- **Headings**: Bold, uppercase, letter-spaced
- **Body**: Regular weight, 13-14px
- **Labels**: Small, uppercase, letter-spaced

### Animations
- **Transitions**: 0.2-0.3s ease
- **Button Hover**: -2px Y translation, shadow increase
- **Fade In**: 0.3s fade in for sections
- **Smooth Scrolling**: Enabled

### Responsive Behavior
- **Desktop** (1024px+): Full 25/75 layout
- **Tablet** (768-1024px): 30/70 layout
- **Mobile** (<768px): Stacked layout (40vh + 60vh)

## Integration with GNN Model

The UI seamlessly integrates with the trained GNN model:

1. **Feature Extraction**: Road network states (congestion, flow) are used as input
2. **Baseline Prediction**: Initial state predicted by `/predict` endpoint
3. **Scenario Modification**: User inputs modify feature values
4. **What-If Analysis**: `/whatif` endpoint computes impact
5. **Visualization**: Predictions displayed as road colors and statistics

## Dependencies

### Frontend Libraries
- **Leaflet.js** 1.9.4 - Interactive mapping
- **Chart.js** 3.9.1 - Statistical charts
- **Vanilla JavaScript** - No jQuery or frameworks

### Backend Requirements
- Flask 3.1.2
- torch 2.6.0+cpu
- torch-geometric 2.7.0
- numpy
- networkx
- flask-cors
- requests

## Performance Considerations

- **Lazy Loading**: City data loaded on startup
- **Debounced Events**: Map interactions throttled
- **Chart Pooling**: Chart instances reused/destroyed
- **Memory Management**: Feature groups cleared before re-rendering
- **Network Optimization**: Compressed API responses

## Future Enhancements

- [ ] Real city graph loading from GraphML files
- [ ] Time-series predictions (hour-by-hour)
- [ ] Multi-scenario comparison
- [ ] Export analysis reports (PDF/PNG)
- [ ] Real-time collaboration features
- [ ] Advanced filtering and search
- [ ] Historical scenario library
- [ ] Performance optimization (web workers for heavy computations)

## Troubleshooting

### Map Not Loading
- Check backend is running: `curl http://localhost:5000/health`
- Verify Leaflet.js and CSS loaded in browser console
- Check browser console for CORS errors

### Scenario Not Running
- Ensure `/predict` endpoint returns valid predictions
- Check network tab for API call responses
- Verify feature array has correct format

### Charts Not Rendering
- Ensure Chart.js library is loaded
- Verify canvas elements exist with correct IDs
- Check Chart.js has data before rendering

### Styling Issues
- Clear browser cache (Ctrl+Shift+Delete)
- Verify styles.css linked in HTML
- Check CSS custom properties supported (modern browsers only)

## Performance Tips

1. **Reduce Map Complexity**: Filter amenities for better performance
2. **Optimize Features**: Use normalized feature values
3. **Batch Updates**: Group map changes together
4. **Memory Management**: Reset scenarios when not needed

## License & Credits

- Built with Leaflet.js (OpenStreetMap)
- Styled with Chart.js
- Backend: Flask + PyTorch + PyG

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready
