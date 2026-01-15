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
let closedEdges = new Set();

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
        const selectedEdgesData = sessionStorage.getItem('selectedEdges');
        
        if (!resultsData) {
            throw new Error('No simulation results found. Please run a simulation first.');
        }
        
        simulationResults = JSON.parse(resultsData);
        
        // Get closed edges
        if (selectedEdgesData) {
            const edges = JSON.parse(selectedEdgesData);
            closedEdges = new Set(edges);
        }

        // Fetch graph data
        const graphResponse = await fetch('/api/graph-data');
        graphData = await graphResponse.json();

        // Setup predictions
        baselinePredictions = graphData.baseline_predictions || {};
        predictionPredictions = simulationResults.predictions || {};
        
        // Ensure ALL edges have predictions - if missing, use baseline
        // This is critical: every road must have a predicted value
        graphData.edges.forEach(edge => {
            if (edge.is_metro) return;
            const edgeId = edge.id;
            
            // If prediction is missing, use baseline (shouldn't happen, but ensure coverage)
            if (predictionPredictions[edgeId] === undefined || predictionPredictions[edgeId] === null) {
                predictionPredictions[edgeId] = baselinePredictions[edgeId] || 1.0;
            }
        });
        
        // Debug: Log to ensure we have predictions for all roads
        console.log('Baseline predictions count:', Object.keys(baselinePredictions).length);
        console.log('Prediction predictions count:', Object.keys(predictionPredictions).length);
        console.log('Graph edges count:', graphData.edges.length);
        console.log('Closed edges:', Array.from(closedEdges));

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
                `<strong>Road: ${edgeId.substring(0, 30)}...</strong><br>` +
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

// Render prediction traffic on second map - SHOW ALL AFFECTED ROADS
function renderPredictionGraph() {
    // Clear existing layers
    predictionRoadLayers.forEach(layer => predictionMap.removeLayer(layer));
    predictionRoadLayers.clear();

    // Render edges with prediction data - ALL ROADS MUST SHOW PREDICTED COLORS
    graphData.edges.forEach(edge => {
        if (edge.is_metro) return; // Skip metro edges

        const edgeId = edge.id;
        const baselineCongestion = baselinePredictions[edgeId] || 1.0;
        
        // CRITICAL: Get predicted congestion - EVERY road must show its predicted state
        // The prediction map should show ALL roads with their predicted colors
        let predictedCongestion = predictionPredictions[edgeId];
        
        // Fallback: if prediction missing (shouldn't happen after initialization), use baseline
        if (predictedCongestion === undefined || predictedCongestion === null) {
            predictedCongestion = baselineCongestion;
            console.warn(`Missing prediction for edge ${edgeId}, using baseline`);
        }
        
        // Check if this is a closed road
        const isClosed = closedEdges.has(edgeId);
        
        // Calculate change to determine visibility
        const change = Math.abs(predictedCongestion - baselineCongestion);
        const hasSignificantChange = change > 0.05; // Even small changes should be visible
        
        // Determine color and styling
        let color, weight, opacity;
        
        if (isClosed) {
            // Closed roads: dark red, thicker, fully opaque - clearly marked
            color = '#991b1b';
            weight = 7;
            opacity = 1.0;
        } else {
            // ALL OTHER ROADS: MUST use predicted congestion color
            // This is the key: every road shows its PREDICTED congestion level
            // Not comparing to baseline - showing actual predicted state
            color = getCongestionColor(predictedCongestion);
            
            // Base styling for all roads
            weight = 3.5;
            opacity = 0.8;
            
            // Make roads with changes more visible
            if (hasSignificantChange) {
                weight = 4.5;
                opacity = 0.9;
                
                // Major changes get even more prominence
                if (change > 0.3) {
                    weight = 5.5;
                    opacity = 0.95;
                }
                
                // Very large changes
                if (change > 1.0) {
                    weight = 6;
                    opacity = 1.0;
                }
            }
        }

        const polyline = L.polyline(edge.geometry, {
            color: color,
            weight: weight,
            opacity: opacity,
            lineCap: 'round',
            lineJoin: 'round'
        });

        // Add hover effect and tooltip
        polyline.on('mouseover', function() {
            this.setStyle({
                weight: weight + 2,
                opacity: 1.0
            });
            
            const change = predictedCongestion - baselineCongestion;
            const pctChange = baselineCongestion > 0 ? (change / baselineCongestion) * 100 : 0;
            
            let tooltipContent = `<strong>Road: ${edgeId.substring(0, 30)}...</strong><br>`;
            
            if (isClosed) {
                tooltipContent += `<span style="color: #991b1b; font-weight: bold;">CLOSED ROAD</span><br>`;
            }
            
            tooltipContent += `Predicted Congestion: <strong>${predictedCongestion.toFixed(2)}x</strong><br>`;
            tooltipContent += `Baseline: ${baselineCongestion.toFixed(2)}x<br>`;
            
            if (Math.abs(pctChange) > 1) {
                const changeColor = change > 0 ? '#ef4444' : '#10b981';
                tooltipContent += `Change: <span style="color: ${changeColor}; font-weight: bold;">${change > 0 ? '+' : ''}${pctChange.toFixed(1)}%</span>`;
            } else {
                tooltipContent += `Change: <span style="color: #6b7280;">Minimal</span>`;
            }
            
            this.bindTooltip(
                tooltipContent,
                { permanent: false, direction: 'top', className: 'map-tooltip' }
            ).openTooltip();
        });

        polyline.on('mouseout', function() {
            this.setStyle({
                weight: weight,
                opacity: opacity
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

    baselineMap.on('zoom', function() {
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

    predictionMap.on('zoom', function() {
        if (isSyncEnabled && !isUserInteracting) {
            isUserInteracting = true;
            baselineMap.setView(predictionMap.getCenter(), predictionMap.getZoom(), {
                animate: false
            });
            isUserInteracting = false;
        }
    });
}

// Get color based on congestion level - Premium color palette
function getCongestionColor(congestion) {
    if (congestion < 2.0) return '#10b981'; // Green - Low (premium green)
    if (congestion < 3.0) return '#f59e0b'; // Amber - Moderate
    if (congestion < 5.0) return '#f97316'; // Orange - High
    return '#ef4444'; // Red - Severe
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initDualMaps();
});
