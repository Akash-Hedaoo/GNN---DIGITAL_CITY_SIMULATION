// Global variables
let map;
let graphData = null;
let baselinePredictions = {};
let currentPredictions = {};
let selectedEdges = new Set();
let roadLayers = new Map(); // Map of edge_id -> Leaflet layer
let viewMode = 'baseline';

// Initialize map
function initMap() {
    // Default center (will be updated when graph loads)
    map = L.map('map').setView([18.6, 73.8], 13);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);
    
    // Show loading overlay
    document.getElementById('loadingOverlay').classList.remove('hidden');
    
    // Load graph data
    loadGraphData();
}

// Load graph data from API
async function loadGraphData() {
    try {
        const response = await fetch('/api/graph-data');
        graphData = await response.json();
        
        // Update map view to fit graph bounds
        const bounds = graphData.bounds;
        map.setView([bounds.center.lat, bounds.center.lon], 13);
        
        // Store baseline predictions
        baselinePredictions = graphData.baseline_predictions;
        currentPredictions = baselinePredictions;
        
        // Render graph on map
        renderGraph();
        
        // Hide loading overlay
        document.getElementById('loadingOverlay').classList.add('hidden');
        
        console.log('Graph loaded:', graphData.nodes.length, 'nodes,', graphData.edges.length, 'edges');
    } catch (error) {
        console.error('Error loading graph data:', error);
        alert('Failed to load graph data. Please refresh the page.');
    }
}

// Render graph on map
function renderGraph() {
    // Clear existing layers
    roadLayers.forEach(layer => map.removeLayer(layer));
    roadLayers.clear();
    
    // Render edges (roads)
    graphData.edges.forEach(edge => {
        // Skip metro edges for road visualization
        if (edge.is_metro) return;
        
        const edgeId = edge.id;
        const congestion = currentPredictions[edgeId] || 1.0;
        
        // Get color based on congestion
        const color = getCongestionColor(congestion);
        
        // Create polyline from geometry
        const polyline = L.polyline(edge.geometry, {
            color: color,
            weight: 3,
            opacity: 0.7,
            lineCap: 'round',
            lineJoin: 'round'
        });
        
        // Add click handler for selection
        polyline.on('click', function(e) {
            toggleEdgeSelection(edgeId, polyline);
        });
        
        // Add hover effect
        polyline.on('mouseover', function(e) {
            if (!selectedEdges.has(edgeId)) {
            this.setStyle({
                weight: 5,
                opacity: 0.9
            });
            }
        });
        
        polyline.on('mouseout', function(e) {
            if (!selectedEdges.has(edgeId)) {
            this.setStyle({
                weight: 3,
                opacity: 0.7
            });
            }
        });
        
        // Add tooltip
        polyline.bindTooltip(
            `Road: ${edgeId.substring(0, 20)}...<br>Congestion: ${congestion.toFixed(2)}<br>Click to select`,
            { permanent: false, direction: 'top', className: 'road-tooltip' }
        );
        
        // Add to map
        polyline.addTo(map);
        roadLayers.set(edgeId, polyline);
    });
    
    // Update selected edges styling
    updateSelectionStyling();
}

// Get color based on congestion level
function getCongestionColor(congestion) {
    if (congestion < 2.0) return '#00ff00'; // Green - Low
    if (congestion < 3.0) return '#ffff00'; // Yellow - Moderate
    if (congestion < 5.0) return '#ff8800'; // Orange - High
    return '#ff0000'; // Red - Severe
}

// Toggle edge selection
function toggleEdgeSelection(edgeId, layer) {
    if (selectedEdges.has(edgeId)) {
        selectedEdges.delete(edgeId);
    } else {
        selectedEdges.add(edgeId);
    }
    
    updateSelectionStyling();
    updateSelectedCount();
}

// Update styling for selected edges
function updateSelectionStyling() {
    roadLayers.forEach((layer, edgeId) => {
        if (selectedEdges.has(edgeId)) {
            layer.setStyle({
                color: '#e74c3c',
                weight: 5,
                opacity: 1,
                className: 'road-selected'
            });
        } else {
            const congestion = currentPredictions[edgeId] || 1.0;
            layer.setStyle({
                color: getCongestionColor(congestion),
                weight: 3,
                opacity: 0.7,
                className: ''
            });
        }
    });
}

// Update selected count badge
function updateSelectedCount() {
    const count = selectedEdges.size;
    document.getElementById('selectedCount').textContent = `${count} road${count !== 1 ? 's' : ''} selected`;
}

// Clear selection
function clearSelection() {
    selectedEdges.clear();
    updateSelectionStyling();
    updateSelectedCount();
    
    // Reset to baseline view
    currentPredictions = baselinePredictions;
    viewMode = 'baseline';
    document.getElementById('viewMode').value = 'baseline';
    renderGraph();
}

// Run simulation
async function runSimulation() {
    if (selectedEdges.size === 0) {
        alert('Please select at least one road to block.');
        return;
    }
    
    // Show loading
    document.getElementById('loadingOverlay').classList.remove('hidden');
    
    try {
        const response = await fetch('/api/simulate-closure', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                closed_edges: Array.from(selectedEdges)
            })
        });
        
        const results = await response.json();
        
        // Store results in sessionStorage for results page
        sessionStorage.setItem('simulationResults', JSON.stringify(results));
        sessionStorage.setItem('selectedEdges', JSON.stringify(Array.from(selectedEdges)));
        
        // Update current predictions for visualization
        currentPredictions = results.predictions;
        
        // Update view mode
        viewMode = 'simulation';
        
        // Re-render graph with new predictions
        renderGraph();
        
        // Hide loading
        document.getElementById('loadingOverlay').classList.add('hidden');
        
        // Navigate to results page
        window.location.href = '/results';
        
    } catch (error) {
        console.error('Error running simulation:', error);
        alert('Failed to run simulation. Please try again.');
        document.getElementById('loadingOverlay').classList.add('hidden');
    }
}

// Handle view mode change
function handleViewModeChange() {
    const mode = document.getElementById('viewMode').value;
    viewMode = mode;
    
    if (mode === 'baseline') {
        currentPredictions = baselinePredictions;
    } else if (mode === 'selection') {
        // Highlight selected edges
        // Keep current predictions but emphasize selection
    }
    
    renderGraph();
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    initMap();
    
    document.getElementById('clearSelection').addEventListener('click', clearSelection);
    document.getElementById('runSimulation').addEventListener('click', runSimulation);
    document.getElementById('viewMode').addEventListener('change', handleViewModeChange);
});

