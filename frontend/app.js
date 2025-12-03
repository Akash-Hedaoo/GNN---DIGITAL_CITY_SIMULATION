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
        congested: '#f39c12',
        closed: '#e74c3c',
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
        amenities: false
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
            zoomControl: true
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
            zoomControl: true
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
        
        showToast('Prediction complete!', 'success');
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
        const amenity = (node.amenity || 'none').toLowerCase();
        
        // Determine color and size based on amenity type
        if (amenity.includes('hospital')) {
            color = CONFIG.COLORS.hospital;
            radius = 8;
            isAmenity = true;
        } else if (amenity.includes('park')) {
            color = CONFIG.COLORS.park;
            radius = 7;
            isAmenity = true;
        } else if (amenity.includes('school')) {
            color = CONFIG.COLORS.school;
            radius = 7;
            isAmenity = true;
        } else if (amenity.includes('mall')) {
            color = CONFIG.COLORS.mall;
            radius = 8;
            isAmenity = true;
        } else if (amenity.includes('factory')) {
            color = CONFIG.COLORS.factory;
            radius = 6;
            isAmenity = true;
        } else if (amenity.includes('warehouse')) {
            color = CONFIG.COLORS.warehouse;
            radius = 5;
            isAmenity = true;
        } else if (amenity.includes('office')) {
            color = CONFIG.COLORS.office;
            radius = 6;
            isAmenity = true;
        } else if (amenity.includes('community')) {
            color = CONFIG.COLORS.community_center;
            radius = 6;
            isAmenity = true;
        }
        
        // Metro stations get special treatment
        if (amenity.includes('metro_station') || node.is_metro) {
            color = CONFIG.COLORS.nodeMetro;
            radius = Math.max(radius, 8);
            isAmenity = true;
        }
        
        const circle = L.circleMarker(pos, {
            radius: radius,
            fillColor: color,
            fillOpacity: isAmenity ? 0.9 : 0.5,
            color: isAmenity ? '#fff' : color,
            weight: isAmenity ? 2 : 1
        });
        
        // Add tooltip for amenities
        if (isAmenity) {
            circle.bindTooltip(`${amenity.replace('_', ' ').toUpperCase()}`, {
                permanent: false,
                direction: 'top'
            });
        }
        
        circle.on('click', () => showNodeInfo(node));
        
        if (isAmenity) {
            state.layers.amenities.addLayer(circle);
        }
        state.layers.nodes.addLayer(circle);
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
    
    // Get all congestion values to calculate dynamic thresholds
    const congestionValues = predictions.map(p => p.congestion);
    const minCong = Math.min(...congestionValues);
    const maxCong = Math.max(...congestionValues);
    const avgCong = congestionValues.reduce((a, b) => a + b, 0) / congestionValues.length;
    
    console.log(`Congestion range: ${minCong.toFixed(2)} - ${maxCong.toFixed(2)}, avg: ${avgCong.toFixed(2)}`);
    
    predictions.forEach(p => {
        predMap[`${p.source}-${p.target}`] = p.congestion;
    });
    
    // Calculate percentile thresholds
    const sorted = [...congestionValues].sort((a, b) => a - b);
    const p25 = sorted[Math.floor(sorted.length * 0.25)];
    const p50 = sorted[Math.floor(sorted.length * 0.50)];
    const p75 = sorted[Math.floor(sorted.length * 0.75)];
    const p90 = sorted[Math.floor(sorted.length * 0.90)];
    
    // Update edge colors based on congestion (using percentiles)
    state.graphData.edges.forEach(edge => {
        if (!edge._polyline) return;
        if (edge.is_metro) return;  // Don't color metro
        
        const congestion = predMap[`${edge.source}-${edge.target}`] || avgCong;
        const isClosed = state.closedRoads.has(`${edge.source}-${edge.target}`);
        
        let color;
        if (isClosed) {
            color = CONFIG.COLORS.closed;
        } else if (congestion >= p90) {
            color = '#e74c3c';  // Top 10% - red (high congestion)
        } else if (congestion >= p75) {
            color = '#f39c12';  // Top 25% - orange
        } else if (congestion >= p50) {
            color = '#f1c40f';  // Top 50% - yellow
        } else {
            color = '#2ecc71';  // Bottom 50% - green (low congestion)
        }
        
        edge._polyline.setStyle({ color: color });
    });
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
            dashArray: null
        });
        showToast(`Opened road: ${roadId}`, 'info');
    } else {
        state.closedRoads.add(roadId);
        edge._polyline.setStyle({
            color: CONFIG.COLORS.closed,
            dashArray: '8, 8'
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
