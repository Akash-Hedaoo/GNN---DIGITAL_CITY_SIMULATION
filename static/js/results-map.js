// Global variables for dual-map view
let baselineMap;
let predictionMap;
let graphData = null;
let simulationResults = null;
let baselinePredictions = {};
let predictionPredictions = {};
let baselineRoadLayers = new Map();
let predictionRoadLayers = new Map();
let isSyncEnabled = true;
let isUserInteracting = false;

// Initialize dual maps
function initDualMaps() {
    // Create baseline map
    baselineMap = L.map('baselineMap').setView([18.6, 73.8], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(baselineMap);

    // Create prediction map
    predictionMap = L.map('predictionMap').setView([18.6, 73.8], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(predictionMap);

    // Show loading overlay
    document.getElementById('loadingOverlay').classList.remove('hidden');

    // Load data
    loadMapData();

    // Setup synchronization
    setupMapSync();

    // Setup sync toggle
    document.getElementById('syncToggle').addEventListener('change', function() {
        isSyncEnabled = this.checked;
    });
}

// Load graph and simulation results
async function loadMapData() {
    try {
        // Get simulation results from session storage
        const resultsData = sessionStorage.getItem('simulationResults');
        if (!resultsData) {
            throw new Error('No simulation results found. Please run a simulation first.');
        }
        simulationResults = JSON.parse(resultsData);

        // Fetch graph data
        const graphResponse = await fetch('/api/graph-data');
        graphData = await graphResponse.json();

        // Setup predictions
        baselinePredictions = graphData.baseline_predictions;
        predictionPredictions = simulationResults.predictions;

        // Update map views
        const bounds = graphData.bounds;
        baselineMap.setView([bounds.center.lat, bounds.center.lon], 13);
        predictionMap.setView([bounds.center.lat, bounds.center.lon], 13);

        // Render graphs on both maps
        renderBaselineGraph();
        renderPredictionGraph();

        // Hide loading overlay
        document.getElementById('loadingOverlay').classList.add('hidden');

        console.log('Dual maps loaded successfully');
    } catch (error) {
        console.error('Error loading map data:', error);
        alert('Failed to load map data: ' + error.message);
        document.getElementById('loadingOverlay').classList.add('hidden');
    }
}

// Render baseline traffic on first map
function renderBaselineGraph() {
    // Clear existing layers
    baselineRoadLayers.forEach(layer => baselineMap.removeLayer(layer));
    baselineRoadLayers.clear();

    // Render edges
    graphData.edges.forEach(edge => {
        if (edge.is_metro) return; // Skip metro edges

        const edgeId = edge.id;
        const congestion = baselinePredictions[edgeId] || 1.0;
        const color = getCongestionColor(congestion);

        const polyline = L.polyline(edge.geometry, {
            color: color,
            weight: 3,
            opacity: 0.7,
            lineCap: 'round',
            lineJoin: 'round'
        });

        // Add hover effect and tooltip
        polyline.on('mouseover', function() {
            this.setStyle({
                weight: 5,
                opacity: 0.9
            });
            this.bindTooltip(
                `<strong>Road: ${edgeId.substring(0, 20)}...</strong><br>` +
                `Baseline Congestion: <strong>${congestion.toFixed(2)}x</strong>`,
                { permanent: false, direction: 'top', className: 'map-tooltip' }
            ).openTooltip();
        });

        polyline.on('mouseout', function() {
            this.setStyle({
                weight: 3,
                opacity: 0.7
            });
            this.closeTooltip();
        });

        polyline.addTo(baselineMap);
        baselineRoadLayers.set(edgeId, polyline);
    });
}

// Render prediction traffic on second map
function renderPredictionGraph() {
    // Clear existing layers
    predictionRoadLayers.forEach(layer => predictionMap.removeLayer(layer));
    predictionRoadLayers.clear();

    // Render edges
    graphData.edges.forEach(edge => {
        if (edge.is_metro) return; // Skip metro edges

        const edgeId = edge.id;
        const congestion = predictionPredictions[edgeId] || 1.0;
        const color = getCongestionColor(congestion);

        const polyline = L.polyline(edge.geometry, {
            color: color,
            weight: 3,
            opacity: 0.7,
            lineCap: 'round',
            lineJoin: 'round'
        });

        // Add hover effect and tooltip
        polyline.on('mouseover', function() {
            this.setStyle({
                weight: 5,
                opacity: 0.9
            });
            
            const baseline = baselinePredictions[edgeId] || 1.0;
            const change = congestion - baseline;
            const pctChange = (change / baseline) * 100;
            
            this.bindTooltip(
                `<strong>Road: ${edgeId.substring(0, 20)}...</strong><br>` +
                `Predicted Congestion: <strong>${congestion.toFixed(2)}x</strong><br>` +
                `Baseline: ${baseline.toFixed(2)}x<br>` +
                `Change: <span style="color: ${change > 0 ? '#e74c3c' : '#27ae60'}"><strong>${change > 0 ? '+' : ''}${pctChange.toFixed(1)}%</strong></span>`,
                { permanent: false, direction: 'top', className: 'map-tooltip' }
            ).openTooltip();
        });

        polyline.on('mouseout', function() {
            this.setStyle({
                weight: 3,
                opacity: 0.7
            });
            this.closeTooltip();
        });

        polyline.addTo(predictionMap);
        predictionRoadLayers.set(edgeId, polyline);
    });
}

// Setup map synchronization
function setupMapSync() {
    // Sync baseline map events to prediction map
    baselineMap.on('move', function() {
        if (isSyncEnabled && !isUserInteracting) {
            isUserInteracting = true;
            predictionMap.setView(baselineMap.getCenter(), baselineMap.getZoom(), {
                animate: false
            });
            isUserInteracting = false;
        }
    });

    // Sync prediction map events to baseline map
    predictionMap.on('move', function() {
        if (isSyncEnabled && !isUserInteracting) {
            isUserInteracting = true;
            baselineMap.setView(predictionMap.getCenter(), predictionMap.getZoom(), {
                animate: false
            });
            isUserInteracting = false;
        }
    });
}

// Get color based on congestion level
function getCongestionColor(congestion) {
    if (congestion < 2.0) return '#00ff00'; // Green - Low
    if (congestion < 3.0) return '#ffff00'; // Yellow - Moderate
    if (congestion < 5.0) return '#ff8800'; // Orange - High
    return '#ff0000'; // Red - Severe
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initDualMaps();
});
