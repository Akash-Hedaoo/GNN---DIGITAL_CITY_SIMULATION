/**
 * 🚦 Digital Twin City Simulation - Frontend Application
 * ========================================================
 * 
 * Interactive map visualization with GNN traffic prediction
 * 
 * Author: Digital Twin City Simulation Team
 * Date: December 2025
 */

// ============================================================
// CONFIGURATION
// ============================================================

const CONFIG = {
    API_BASE: 'http://localhost:5000/api',
    // Pune, India - center coordinates
    MAP_CENTER: [18.5204, 73.8567],
    MAP_ZOOM: 13,
    // Set to true to overlay on real map, false for abstract view
    USE_REAL_MAP: true,
    // Scale factor to convert graph coordinates to real-world offset
    COORD_SCALE: 0.001,  // Adjust this to fit city size
    COLORS: {
        road: '#3498db',
        roadHover: '#2980b9',
        // Metro line colors
        metroRed: '#FF0000',
        metroBlue: '#0000FF', 
        metroGreen: '#00FF00',
        highCongestion: '#e74c3c',  // Red for high congestion
        lowCongestion: '#2ecc71',   // Green for low congestion
        closed: '#ff69b4',          // Pink for blocked roads,
        // Node colors
        node: '#95a5a6',
        nodeMetro: '#9b59b6',
        // Amenity colors
        hospital: '#e74c3c',
        park: '#27ae60',
        school: '#f39c12',
        mall: '#e91e63',
        factory: '#795548',
        warehouse: '#607d8b',
        office: '#3498db',
        community_center: '#9c27b0'
    }
};

// ============================================================
// STATE
// ============================================================

const state = {
    map: null,
    graphData: null,
    predictions: null,
    baselinePredictions: null,  // Store baseline for comparison
    closedRoads: new Set(),
    layers: {
        roads: null,
        metro: null,
        nodes: null,
        amenities: null
    },
    visible: {
        roads: true,
        metro: true,
        nodes: true,
        amenities: true
    }
};

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', init);

async function init() {
    showLoading('Initializing...');
    
    // Initialize map
    initMap();
    
    // Setup event listeners
    setupEventListeners();
    
    // Check API status
    await checkStatus();
    
    // Load graph data
    await loadGraphData();
    
    hideLoading();
}

function initMap() {
    if (CONFIG.USE_REAL_MAP) {
        // Real map with OpenStreetMap tiles
        state.map = L.map('map', {
            center: CONFIG.MAP_CENTER,
            zoom: CONFIG.MAP_ZOOM,
            zoomControl: true,
            zoomDelta: 0.5,          // Smaller zoom steps
            zoomSnap: 0.5,           // Snap to 0.5 increments
            wheelPxPerZoomLevel: 120 // More pixels needed to zoom (slower scroll zoom)
        });

        // Add OpenStreetMap tile layer (dark theme)
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(state.map);
    } else {
        // Abstract view with simple coordinates
        state.map = L.map('map', {
            center: [0, 0],
            zoom: 2,
            crs: L.CRS.Simple,
            zoomControl: true,
            zoomDelta: 0.5,
            zoomSnap: 0.5,
            wheelPxPerZoomLevel: 120
        });
    }

    // Add zoom control to top-right
    state.map.zoomControl.setPosition('topright');
}

function setupEventListeners() {
    // Layer toggles
    document.getElementById('layer-roads').addEventListener('change', (e) => {
        state.visible.roads = e.target.checked;
        updateLayerVisibility();
    });
    
    document.getElementById('layer-metro').addEventListener('change', (e) => {
        state.visible.metro = e.target.checked;
        updateLayerVisibility();
    });
    
    document.getElementById('layer-nodes').addEventListener('change', (e) => {
        state.visible.nodes = e.target.checked;
        updateLayerVisibility();
    });
    
    document.getElementById('layer-amenities').addEventListener('change', (e) => {
        state.visible.amenities = e.target.checked;
        updateLayerVisibility();
    });
    
    // Buttons
    document.getElementById('btn-predict').addEventListener('click', runPrediction);
    document.getElementById('btn-reset').addEventListener('click', resetView);
    document.getElementById('btn-clear-closures').addEventListener('click', clearClosures);
    
    // Info panel close
    document.getElementById('close-info').addEventListener('click', () => {
        document.getElementById('info-panel').classList.remove('visible');
    });
    
    // Analysis button
    document.getElementById('btn-analysis').addEventListener('click', goToAnalysis);
}

// Navigate to analysis page with current state
function goToAnalysis() {
    if (state.closedRoads.size === 0) {
        showToast('Block some roads first before viewing analysis!', 'error');
        return;
    }
    
    // Save current state to localStorage
    const analysisData = {
        closedRoads: Array.from(state.closedRoads),
        timestamp: Date.now()
    };
    localStorage.setItem('cityAnalysisData', JSON.stringify(analysisData));
    
    showToast('Opening analysis page...', 'info');
    
    // Navigate to analysis page
    setTimeout(() => {
        window.location.href = 'analysis.html';
    }, 300);
}

// ============================================================
// API CALLS
// ============================================================

async function checkStatus() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/status`);
        const data = await response.json();
        
        updateStatusUI(data);
        return data;
    } catch (error) {
        console.error('Status check failed:', error);
        updateStatusUI({ status: 'offline' });
        showToast('Failed to connect to backend', 'error');
        return null;
    }
}

async function loadGraphData() {
    showLoading('Loading city graph...');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/graph`);
        state.graphData = await response.json();
        
        // Debug: count metro edges
        const metroEdges = state.graphData.edges.filter(e => e.is_metro);
        console.log(`Loaded ${state.graphData.node_count} nodes, ${state.graphData.edge_count} edges`);
        console.log(`Metro edges: ${metroEdges.length}`);
        if (metroEdges.length > 0) {
            console.log('Sample metro edge:', metroEdges[0]);
        }
        
        // Render the graph
        renderGraph();
        
        showToast(`Loaded ${state.graphData.node_count} nodes`, 'success');
    } catch (error) {
        console.error('Failed to load graph:', error);
        showToast('Failed to load graph data', 'error');
    }
}

async function runPrediction() {
    showLoading('Running prediction...');
    
    try {
        // FIRST: Get baseline prediction (no closures) for comparison
        const baselineResponse = await fetch(`${CONFIG.API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ closed_roads: [] })
        });
        state.baselinePredictions = await baselineResponse.json();
        
        // THEN: Get prediction with current road closures
        const response = await fetch(`${CONFIG.API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                closed_roads: Array.from(state.closedRoads)
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        state.predictions = data;
        
        // Update visualization
        updatePredictionVisualization();
        updateStatsUI(data.stats);
        
        // Show result message
        if (state.closedRoads.size > 0) {
            console.log('=== PREDICTION WITH ROAD CLOSURES ===');
            console.log(`Closed roads: ${Array.from(state.closedRoads).join(', ')}`);
            showToast(`Prediction done! Showing traffic impact of ${state.closedRoads.size} blocked road(s)`, 'success');
        } else {
            showToast('Prediction done! Showing current traffic levels', 'success');
        }
    } catch (error) {
        console.error('Prediction failed:', error);
        showToast('Prediction failed: ' + error.message, 'error');
    }
    
    hideLoading();
}

// ============================================================
// RENDERING
// ============================================================

function renderGraph() {
    if (!state.graphData) return;
    
    const { nodes, edges } = state.graphData;
    
    // Calculate bounds of graph coordinates
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    const nodeMap = {};
    nodes.forEach(node => {
        nodeMap[node.id] = node;
        minX = Math.min(minX, node.x);
        maxX = Math.max(maxX, node.x);
        minY = Math.min(minY, node.y);
        maxY = Math.max(maxY, node.y);
    });
    
    // Calculate center and range of graph
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    
    // Transform function: maps graph coords to real-world lat/lng
    let transform;
    if (CONFIG.USE_REAL_MAP) {
        // Map to real Pune coordinates
        // Scale to cover roughly 5km x 5km area (0.045 degrees ~ 5km)
        const scaleFactor = 0.045 / Math.max(rangeX, rangeY);
        transform = (x, y) => [
            CONFIG.MAP_CENTER[0] + (y - centerY) * scaleFactor,
            CONFIG.MAP_CENTER[1] + (x - centerX) * scaleFactor
        ];
    } else {
        // Simple scaling for abstract view
        const scale = 10;
        transform = (x, y) => [y * scale, x * scale];
    }
    
    // Clear existing layers
    Object.values(state.layers).forEach(layer => {
        if (layer) state.map.removeLayer(layer);
    });
    
    // Create layer groups
    state.layers.roads = L.layerGroup();
    state.layers.metro = L.layerGroup();
    state.layers.nodes = L.layerGroup();
    state.layers.amenities = L.layerGroup();
    
    // Render edges
    edges.forEach(edge => {
        const source = nodeMap[edge.source];
        const target = nodeMap[edge.target];
        
        if (!source || !target) return;
        
        const latlngs = [
            transform(source.x, source.y),
            transform(target.x, target.y)
        ];
        
        const isMetro = edge.is_metro === true;  // Strict boolean check
        const isClosed = state.closedRoads.has(`${edge.source}-${edge.target}`);
        
        let color = CONFIG.COLORS.road;  // Default: blue (#3498db)
        let weight = 2;
        let dashArray = null;
        
        if (isMetro) {
            // Metro lines - use their specific colors
            if (edge.line_color) {
                color = edge.line_color;
            } else {
                const line = (edge.metro_line || '').toLowerCase();
                if (line.includes('red')) color = CONFIG.COLORS.metroRed;
                else if (line.includes('blue')) color = CONFIG.COLORS.metroBlue;
                else if (line.includes('green')) color = CONFIG.COLORS.metroGreen;
                else color = '#9b59b6';  // Default purple
            }
            weight = 5;
        }
        
        if (isClosed) {
            color = CONFIG.COLORS.closed;
            dashArray = '8, 8';
        }
        
        const polyline = L.polyline(latlngs, {
            color: color,
            weight: weight,
            opacity: isMetro ? 0.9 : 0.7,
            dashArray: dashArray
        });
        
        // Add click handler for road closure (not for metro)
        if (!isMetro) {
            polyline.on('click', () => toggleRoadClosure(edge));
            polyline.on('mouseover', function() {
                this.setStyle({ weight: weight + 2, opacity: 1 });
            });
            polyline.on('mouseout', function() {
                this.setStyle({ weight: weight, opacity: 0.7 });
            });
        }
        
        // Add to appropriate layer
        if (isMetro) {
            state.layers.metro.addLayer(polyline);
        } else {
            state.layers.roads.addLayer(polyline);
        }
        
        // Store reference for updates
        edge._polyline = polyline;
    });
    
    // Render nodes
    nodes.forEach(node => {
        const pos = transform(node.x, node.y);
        
        let color = CONFIG.COLORS.node;
        let radius = 3;
        let isAmenity = false;
        let emoji = null;
        const amenity = (node.amenity || 'none').toLowerCase();
        
        // Determine emoji and properties based on amenity type
        if (amenity.includes('hospital')) {
            emoji = '🏥';
            isAmenity = true;
        } else if (amenity.includes('park')) {
            emoji = '🌳';
            isAmenity = true;
        } else if (amenity.includes('school')) {
            emoji = '🏫';
            isAmenity = true;
        } else if (amenity.includes('mall')) {
            emoji = '🛒';
            isAmenity = true;
        } else if (amenity.includes('factory')) {
            emoji = '🏭';
            isAmenity = true;
        } else if (amenity.includes('warehouse')) {
            emoji = '📦';
            isAmenity = true;
        } else if (amenity.includes('office')) {
            emoji = '🏢';
            isAmenity = true;
        } else if (amenity.includes('community')) {
            emoji = '🏛️';
            isAmenity = true;
        }
        
        // Metro stations get special treatment
        if (amenity.includes('metro_station') || node.is_metro) {
            emoji = '🚇';
            isAmenity = true;
        }
        
        let marker;
        
        if (emoji) {
            // Use emoji marker for amenities
            const emojiIcon = L.divIcon({
                html: `<div class="emoji-marker">${emoji}</div>`,
                className: 'emoji-icon',
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            });
            marker = L.marker(pos, { icon: emojiIcon });
            
            // Add tooltip for amenities
            marker.bindTooltip(`${amenity.replace('_', ' ').toUpperCase()}`, {
                permanent: false,
                direction: 'top'
            });
        } else {
            // Use circle for regular nodes
            marker = L.circleMarker(pos, {
                radius: radius,
                fillColor: color,
                fillOpacity: 0.5,
                color: color,
                weight: 1
            });
        }
        
        marker.on('click', () => showNodeInfo(node));
        
        if (isAmenity) {
            state.layers.amenities.addLayer(marker);
        }
        state.layers.nodes.addLayer(marker);
    });
    
    // Add visible layers to map (amenities ON by default now)
    if (state.visible.roads) state.layers.roads.addTo(state.map);
    if (state.visible.metro) state.layers.metro.addTo(state.map);
    if (state.visible.nodes) state.layers.nodes.addTo(state.map);
    state.layers.amenities.addTo(state.map);  // Always show amenities
    
    // Fit bounds to show all nodes
    if (CONFIG.USE_REAL_MAP) {
        // Create bounds from transformed node positions
        const allPositions = nodes.map(n => transform(n.x, n.y));
        const bounds = L.latLngBounds(allPositions);
        state.map.fitBounds(bounds, { padding: [50, 50] });
    } else {
        const scale = 10;
        const bounds = [
            [minY * scale, minX * scale],
            [maxY * scale, maxX * scale]
        ];
        state.map.fitBounds(bounds, { padding: [50, 50] });
    }
}

function updateLayerVisibility() {
    // Roads
    if (state.visible.roads && state.layers.roads) {
        state.layers.roads.addTo(state.map);
    } else if (state.layers.roads) {
        state.map.removeLayer(state.layers.roads);
    }
    
    // Metro
    if (state.visible.metro && state.layers.metro) {
        state.layers.metro.addTo(state.map);
    } else if (state.layers.metro) {
        state.map.removeLayer(state.layers.metro);
    }
    
    // Nodes
    if (state.visible.nodes && state.layers.nodes) {
        state.layers.nodes.addTo(state.map);
    } else if (state.layers.nodes) {
        state.map.removeLayer(state.layers.nodes);
    }
    
    // Amenities
    if (state.visible.amenities && state.layers.amenities) {
        state.layers.amenities.addTo(state.map);
    } else if (state.layers.amenities) {
        state.map.removeLayer(state.layers.amenities);
    }
}

function updatePredictionVisualization() {
    if (!state.predictions || !state.graphData) return;
    
    const predictions = state.predictions.predictions;
    const predMap = {};
    
    predictions.forEach(p => {
        predMap[`${p.source}-${p.target}`] = p.congestion;
    });
    
    // Build baseline map for comparison (only if we have closed roads)
    const baselineMap = {};
    const hasClosures = state.closedRoads.size > 0;
    
    if (state.baselinePredictions && hasClosures) {
        state.baselinePredictions.predictions.forEach(p => {
            baselineMap[`${p.source}-${p.target}`] = p.congestion;
        });
    }
    
    // Calculate percentiles for congestion levels
    const congestionValues = predictions.map(p => p.congestion);
    const sorted = [...congestionValues].sort((a, b) => a - b);
    const p20 = sorted[Math.floor(sorted.length * 0.20)];
    const p40 = sorted[Math.floor(sorted.length * 0.40)];
    const p60 = sorted[Math.floor(sorted.length * 0.60)];
    const p80 = sorted[Math.floor(sorted.length * 0.80)];
    
    // Track changed edges
    let increasedCount = 0;
    let decreasedCount = 0;
    
    state.graphData.edges.forEach(edge => {
        if (!edge._polyline) return;
        if (edge.is_metro) return;  // Don't color metro
        
        const edgeKey = `${edge.source}-${edge.target}`;
        const congestion = predMap[edgeKey];
        const baseline = baselineMap[edgeKey];
        const isClosed = state.closedRoads.has(edgeKey);
        
        let color = '#3498db';  // Default blue
        let weight = 2;
        let dashArray = null;
        
        if (isClosed) {
            // Blocked road - PINK dashed
            color = '#ff69b4';
            weight = 5;
            dashArray = '12, 6';
        } else if (congestion !== undefined) {
            // Color based on absolute congestion level (5 levels)
            if (congestion >= p80) {
                // Very High - Dark Red
                color = '#c0392b';
                weight = 4;
            } else if (congestion >= p60) {
                // High - Orange
                color = '#e67e22';
                weight = 3;
            } else if (congestion >= p40) {
                // Medium - Yellow
                color = '#f1c40f';
                weight = 2.5;
            } else if (congestion >= p20) {
                // Low - Light Green
                color = '#2ecc71';
                weight = 2;
            } else {
                // Very Low - Dark Green
                color = '#1e8449';
                weight = 2;
            }
            
            // Track changes from baseline
            if (hasClosures && baseline !== undefined) {
                const diff = congestion - baseline;
                if (diff > 0.01) increasedCount++;
                else if (diff < -0.01) decreasedCount++;
            }
        }
        
        edge._polyline.setStyle({ color: color, weight: weight, dashArray: dashArray });
    });
    
    if (hasClosures) {
        console.log(`Traffic redistribution: ${increasedCount} roads got busier, ${decreasedCount} roads got less busy`);
    } else {
        console.log(`Showing current traffic levels (no road closures)`);
    }
}

// ============================================================
// ROAD CLOSURE
// ============================================================

function toggleRoadClosure(edge) {
    const roadId = `${edge.source}-${edge.target}`;
    
    if (state.closedRoads.has(roadId)) {
        state.closedRoads.delete(roadId);
        edge._polyline.setStyle({
            color: CONFIG.COLORS.road,
            weight: 2,
            dashArray: null
        });
        showToast(`Opened road: ${roadId}`, 'info');
    } else {
        state.closedRoads.add(roadId);
        edge._polyline.setStyle({
            color: '#ff69b4',  // Pink
            weight: 4,
            dashArray: '10, 6'
        });
        showToast(`Closed road: ${roadId}`, 'info');
    }
    
    updateClosedRoadsList();
}

function updateClosedRoadsList() {
    const container = document.getElementById('closed-roads-list');
    
    if (state.closedRoads.size === 0) {
        container.innerHTML = '<p class="empty-message">No roads closed</p>';
        return;
    }
    
    container.innerHTML = '';
    state.closedRoads.forEach(roadId => {
        const item = document.createElement('div');
        item.className = 'closed-road-item';
        item.innerHTML = `
            <span>${roadId}</span>
            <button onclick="removeClosedRoad('${roadId}')">×</button>
        `;
        container.appendChild(item);
    });
}

function removeClosedRoad(roadId) {
    state.closedRoads.delete(roadId);
    
    // Update visual
    const edge = state.graphData.edges.find(e => 
        `${e.source}-${e.target}` === roadId
    );
    if (edge && edge._polyline) {
        edge._polyline.setStyle({
            color: CONFIG.COLORS.road,
            dashArray: null
        });
    }
    
    updateClosedRoadsList();
}

function clearClosures() {
    state.closedRoads.forEach(roadId => removeClosedRoad(roadId));
    state.closedRoads.clear();
    updateClosedRoadsList();
    showToast('All road closures cleared', 'info');
}

// ============================================================
// UI UPDATES
// ============================================================

function updateStatusUI(data) {
    const badge = document.getElementById('status-badge');
    const text = document.getElementById('status-text');
    const deviceText = document.getElementById('device-text');
    
    if (data.status === 'online') {
        badge.className = 'status-badge online';
        text.textContent = 'Online';
        
        document.getElementById('model-status').textContent = 
            data.model_loaded ? '✓ Loaded' : '✗ Not loaded';
        document.getElementById('model-status').className = 
            'value ' + (data.model_loaded ? 'success' : 'error');
            
        document.getElementById('graph-status').textContent = 
            data.graph_loaded ? '✓ Loaded' : '✗ Not loaded';
        document.getElementById('graph-status').className = 
            'value ' + (data.graph_loaded ? 'success' : 'error');
            
        document.getElementById('node-count').textContent = data.nodes || '--';
        document.getElementById('edge-count').textContent = data.edges || '--';
        
        deviceText.textContent = data.device?.toUpperCase() || 'CPU';
    } else {
        badge.className = 'status-badge offline';
        text.textContent = 'Offline';
    }
}

function updateStatsUI(stats) {
    if (!stats) return;
    
    document.getElementById('stat-mean').textContent = stats.mean_congestion.toFixed(2);
    document.getElementById('stat-max').textContent = stats.max_congestion.toFixed(2);
    document.getElementById('stat-road').textContent = stats.road_mean.toFixed(2);
    document.getElementById('stat-metro').textContent = stats.metro_mean.toFixed(2);
}

function showNodeInfo(node) {
    const panel = document.getElementById('info-panel');
    const title = document.getElementById('info-title');
    const content = document.getElementById('info-content');
    
    title.textContent = `Node ${node.id}`;
    content.innerHTML = `
        <p><span class="label">Zone:</span> <span class="value">${node.zone}</span></p>
        <p><span class="label">Population:</span> <span class="value">${node.population.toLocaleString()}</span></p>
        <p><span class="label">Amenity:</span> <span class="value">${node.amenity || 'None'}</span></p>
        <p><span class="label">Metro Station:</span> <span class="value">${node.is_metro ? 'Yes' : 'No'}</span></p>
        <p><span class="label">Position:</span> <span class="value">(${node.x.toFixed(2)}, ${node.y.toFixed(2)})</span></p>
    `;
    
    panel.classList.add('visible');
}

function resetView() {
    if (state.graphData) {
        renderGraph();
    }
    state.predictions = null;
    
    // Reset stats
    document.getElementById('stat-mean').textContent = '--';
    document.getElementById('stat-max').textContent = '--';
    document.getElementById('stat-road').textContent = '--';
    document.getElementById('stat-metro').textContent = '--';
    
    showToast('View reset', 'info');
}

// ============================================================
// UTILITIES
// ============================================================

function showLoading(message = 'Loading...') {
    document.getElementById('loading-text').textContent = message;
    document.getElementById('loading-overlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.add('hidden');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? 'check-circle' : 
                 type === 'error' ? 'exclamation-circle' : 'info-circle';
    
    toast.innerHTML = `<i class="fas fa-${icon}"></i> ${message}`;
    container.appendChild(toast);
    
    setTimeout(() => toast.remove(), 3000);
}

// Make removeClosedRoad available globally for onclick
window.removeClosedRoad = removeClosedRoad;
