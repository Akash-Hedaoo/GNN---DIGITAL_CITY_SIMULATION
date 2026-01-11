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
        route: '#00ff88',           // Bright green for route
        routeGlow: '#00ffaa',       // Route glow color
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
    },
    // Time-based congestion multipliers
    TIME_MULTIPLIERS: {
        0: 0.3,   // 12 AM - very low
        1: 0.2,   // 1 AM
        2: 0.2,   // 2 AM
        3: 0.2,   // 3 AM
        4: 0.3,   // 4 AM
        5: 0.5,   // 5 AM - early risers
        6: 0.7,   // 6 AM
        7: 0.9,   // 7 AM - morning rush starts
        8: 1.3,   // 8 AM - peak morning
        9: 1.5,   // 9 AM - peak morning
        10: 1.1,  // 10 AM
        11: 1.0,  // 11 AM
        12: 1.1,  // 12 PM - lunch
        13: 1.0,  // 1 PM
        14: 0.9,  // 2 PM
        15: 0.9,  // 3 PM
        16: 1.0,  // 4 PM
        17: 1.3,  // 5 PM - evening rush starts
        18: 1.5,  // 6 PM - peak evening
        19: 1.4,  // 7 PM
        20: 1.0,  // 8 PM
        21: 0.8,  // 9 PM
        22: 0.6,  // 10 PM
        23: 0.4   // 11 PM
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
    baselineStats: null,
    closedRoads: new Set(),
    removedNodes: new Set(),  // NEW: Nodes removed from simulation
    nodeImpactAnalysis: {},  // NEW: Store impact analysis for each removed node
    layers: {
        roads: null,
        metro: null,
        nodes: null,
        amenities: null,
        route: null  // NEW: Route layer
    },
    visible: {
        roads: true,
        metro: true,
        nodes: true,
        amenities: true
    },
    // NEW: Theme state
    theme: 'dark',
    tileLayer: null,
    // NEW: Current simulated hour
    currentHour: 8,
    // NEW: Route state
    currentRoute: null,
    routeMarkers: [],
    // NEW: Node map for quick lookup
    nodeMap: {},
    // NEW: Transform function reference
    transform: null
};

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', init);

async function init() {
    try {
        console.log('🚀 Starting initialization...');
        showLoading('Initializing...');
        
        // Initialize map
        console.log('🗺️ Initializing map...');
        initMap();
        console.log('✅ Map initialized');
        
        // Setup event listeners
        console.log('🔧 Setting up event listeners...');
        setupEventListeners();
        console.log('✅ Event listeners ready');
        
        // Check API status
        showLoading('Connecting to server...');
        console.log('📡 Checking API status...');
        const status = await checkStatus();
        console.log('✅ API status:', status);
        
        // Load graph data
        showLoading('Loading city graph...');
        console.log('📊 Loading graph data...');
        await loadGraphData();
        
        hideLoading();
        console.log('✅ Initialization complete');
    } catch (error) {
        console.error('❌ Initialization error:', error);
        hideLoading();
        showToast('Failed to initialize: ' + error.message, 'error');
    }
}

function initMap() {
    // Load saved theme
    const savedTheme = localStorage.getItem('citySimTheme') || 'dark';
    state.theme = savedTheme;
    document.documentElement.setAttribute('data-theme', savedTheme);
    
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

        // Add tile layer based on theme
        updateMapTiles();
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

// NEW: Update map tiles based on theme
function updateMapTiles() {
    if (state.tileLayer) {
        state.map.removeLayer(state.tileLayer);
    }
    
    if (state.theme === 'dark') {
        state.tileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        });
    } else {
        state.tileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        });
    }
    
    state.tileLayer.addTo(state.map);
}

function setupEventListeners() {
    // Add null checks for all elements before adding event listeners
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', runPrediction);
    }
    
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetSimulation);
    }
    
    const viewAnalysisBtn = document.getElementById('view-analysis-btn');
    if (viewAnalysisBtn) {
        viewAnalysisBtn.addEventListener('click', goToAnalysis);
    }
    
    const ctmInitBtn = document.getElementById('ctm-init-btn');
    if (ctmInitBtn) {
        console.log('✅ CTM Init button found');
        ctmInitBtn.addEventListener('click', initCTM);
    } else {
        console.error('❌ CTM Init button NOT found');
    }
    
    const ctmStepBtn = document.getElementById('ctm-step-btn');
    if (ctmStepBtn) {
        console.log('✅ CTM Step button found');
        ctmStepBtn.addEventListener('click', () => stepCTM(1));
    } else {
        console.error('❌ CTM Step button NOT found');
    }
    
    const ctmRunBtn = document.getElementById('ctm-run-btn');
    if (ctmRunBtn) {
        console.log('✅ CTM Run button found');
        ctmRunBtn.addEventListener('click', () => stepCTM(10));
    } else {
        console.error('❌ CTM Run button NOT found');
    }
    
    const ctmResetBtn = document.getElementById('ctm-reset-btn');
    if (ctmResetBtn) {
        console.log('✅ CTM Reset button found');
        ctmResetBtn.addEventListener('click', resetCTM);
    } else {
        console.error('❌ CTM Reset button NOT found');
    }
    
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportData);
    }
    
    // Add more null checks for any other elements with event listeners
    const closeRoadBtn = document.getElementById('close-road-btn');
    if (closeRoadBtn) {
        closeRoadBtn.addEventListener('click', handleCloseRoad);
    }
    
    const reopenRoadBtn = document.getElementById('reopen-road-btn');
    if (reopenRoadBtn) {
        reopenRoadBtn.addEventListener('click', handleReopenRoad);
    }
    
    // Layer toggle checkboxes - FIX: These were missing!
    const layerRoads = document.getElementById('layer-roads');
    if (layerRoads) {
        layerRoads.addEventListener('change', (e) => {
            state.visible.roads = e.target.checked;
            updateLayerVisibility();
        });
    }
    
    // Analysis button
    const btnAnalysis = document.getElementById('btn-analysis');
    if (btnAnalysis) {
        btnAnalysis.addEventListener('click', goToAnalysis);
    }
    
    // NEW: Theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    // NEW: Search functionality
    setupSearchListeners();
    
    // NEW: Route planner
    const btnFindRoute = document.getElementById('btn-find-route');
    if (btnFindRoute) {
        btnFindRoute.addEventListener('click', findRoute);
    }
    
    const btnClearRoute = document.getElementById('btn-clear-route');
    if (btnClearRoute) {
        btnClearRoute.addEventListener('click', clearRoute);
    }
    
    // NEW: Time slider
    setupTimeSlider();
}

// NEW: Toggle between dark and light theme
function toggleTheme() {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', state.theme);
    localStorage.setItem('citySimTheme', state.theme);
    
    if (CONFIG.USE_REAL_MAP) {
        updateMapTiles();
    }
    
    showToast(`Switched to ${state.theme} theme`, 'info');
}

// NEW: Setup search listeners
function setupSearchListeners() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    
    let searchTimeout = null;
    
    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        const query = e.target.value.trim().toLowerCase();
        
        if (query.length < 2) {
            searchResults.classList.remove('active');
            return;
        }
        
        searchTimeout = setTimeout(() => performSearch(query), 300);
    });
    
    searchInput.addEventListener('focus', () => {
        if (searchInput.value.trim().length >= 2) {
            searchResults.classList.add('active');
        }
    });
    
    // Close search results when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-container')) {
            searchResults.classList.remove('active');
        }
    });
}

// NEW: Perform search
function performSearch(query) {
    if (!state.graphData) return;
    
    const results = [];
    const { nodes } = state.graphData;
    
    // Search through nodes
    nodes.forEach(node => {
        const amenity = (node.amenity || '').toLowerCase();
        const zone = (node.zone || '').toLowerCase();
        const nodeId = String(node.id).toLowerCase();
        
        let match = false;
        let priority = 0;
        
        // Match by ID
        if (nodeId.includes(query)) {
            match = true;
            priority = nodeId === query ? 3 : 1;
        }
        
        // Match by amenity
        if (amenity.includes(query)) {
            match = true;
            priority = 2;
        }
        
        // Match by zone
        if (zone.includes(query)) {
            match = true;
            priority = 1;
        }
        
        if (match) {
            results.push({ ...node, priority });
        }
    });
    
    // Sort by priority and limit results
    results.sort((a, b) => b.priority - a.priority);
    const topResults = results.slice(0, 10);
    
    // Display results
    displaySearchResults(topResults);
}

// NEW: Display search results
function displaySearchResults(results) {
    const container = document.getElementById('search-results');
    
    if (results.length === 0) {
        container.innerHTML = '<div class="search-result-item"><span class="result-text">No results found</span></div>';
        container.classList.add('active');
        return;
    }
    
    container.innerHTML = results.map(node => {
        const emoji = getAmenityEmoji(node.amenity);
        const amenityText = node.amenity ? node.amenity.replace('_', ' ') : 'Node';
        
        return `
            <div class="search-result-item" data-node-id="${node.id}">
                <span class="result-icon">${emoji}</span>
                <div class="result-text">
                    <div class="result-name">Node ${node.id}</div>
                    <div class="result-details">${amenityText} • ${node.zone} • Pop: ${node.population}</div>
                </div>
            </div>
        `;
    }).join('');
    
    // Add click handlers
    container.querySelectorAll('.search-result-item').forEach(item => {
        item.addEventListener('click', () => {
            const nodeId = item.dataset.nodeId;
            focusOnNode(nodeId);
            container.classList.remove('active');
            document.getElementById('search-input').value = '';
        });
    });
    
    container.classList.add('active');
}

// NEW: Get emoji for amenity type
function getAmenityEmoji(amenity) {
    if (!amenity) return '📍';
    const a = amenity.toLowerCase();
    if (a.includes('hospital')) return '🏥';
    if (a.includes('park')) return '🌳';
    if (a.includes('school')) return '🏫';
    if (a.includes('mall')) return '🛒';
    if (a.includes('factory')) return '🏭';
    if (a.includes('warehouse')) return '📦';
    if (a.includes('office')) return '🏢';
    if (a.includes('metro')) return '🚇';
    if (a.includes('community')) return '🏛️';
    return '📍';
}

// NEW: Focus map on a specific node
function focusOnNode(nodeId) {
    const node = state.nodeMap[nodeId];
    if (!node || !state.transform) return;
    
    const pos = state.transform(node.x, node.y);
    state.map.setView(pos, 16, { animate: true });
    
    // Show a temporary marker
    const pulseIcon = L.divIcon({
        html: '<div class="pulse-marker"></div>',
        className: 'pulse-icon',
        iconSize: [30, 30]
    });
    
    const marker = L.marker(pos, { icon: pulseIcon }).addTo(state.map);
    setTimeout(() => state.map.removeLayer(marker), 3000);
    
    // Show node info
    showNodeInfo(node);
    showToast(`Found: Node ${nodeId}`, 'success');
}

// NEW: Setup time slider
function setupTimeSlider() {
    const slider = document.getElementById('time-slider');
    const timeDisplay = document.getElementById('current-time');
    const periodDisplay = document.getElementById('time-period');
    
    slider.addEventListener('input', (e) => {
        const hour = parseInt(e.target.value);
        state.currentHour = hour;
        updateTimeDisplay(hour);
    });
    
    // Quick time buttons
    document.querySelectorAll('.time-quick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const hour = parseInt(btn.dataset.hour);
            slider.value = hour;
            state.currentHour = hour;
            updateTimeDisplay(hour);
            
            // Update button states
            document.querySelectorAll('.time-quick-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Auto-run prediction with new time
            if (state.graphData) {
                runPrediction();
            }
        });
    });
}

// NEW: Update time display
function updateTimeDisplay(hour) {
    const timeDisplay = document.getElementById('current-time');
    const periodDisplay = document.getElementById('time-period');
    
    // Format time
    const displayHour = hour === 0 ? 12 : (hour > 12 ? hour - 12 : hour);
    const ampm = hour < 12 ? 'AM' : 'PM';
    timeDisplay.textContent = `${String(displayHour).padStart(2, '0')}:00 ${ampm}`;
    
    // Determine period
    let period, periodClass;
    if (hour >= 6 && hour < 10) {
        period = '🌅 Morning Rush';
        periodClass = 'rush';
    } else if (hour >= 10 && hour < 16) {
        period = '☀️ Daytime';
        periodClass = 'morning';
    } else if (hour >= 16 && hour < 20) {
        period = '🌆 Evening Rush';
        periodClass = 'rush';
    } else if (hour >= 20 || hour < 6) {
        period = '🌙 Night';
        periodClass = 'night';
    }
    
    periodDisplay.textContent = period;
    periodDisplay.className = `time-period ${periodClass}`;
}

// NEW: Find route between two nodes
async function findRoute() {
    const sourceInput = document.getElementById('route-source').value.trim();
    const targetInput = document.getElementById('route-target').value.trim();
    
    if (!sourceInput || !targetInput) {
        showToast('Please enter both source and destination nodes', 'error');
        return;
    }
    
    showLoading('Finding route...');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/shortest-path`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: sourceInput, target: targetInput })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Store and display route
        state.currentRoute = data;
        displayRoute(data);
        
        showToast(`Route found: ${data.hops} hops`, 'success');
    } catch (error) {
        console.error('Route finding failed:', error);
        showToast('Route not found: ' + error.message, 'error');
    }
    
    hideLoading();
}

// NEW: Display route on map
function displayRoute(routeData) {
    // Clear previous route
    clearRouteLayer();
    
    if (!routeData.path || routeData.path.length < 2) return;
    
    const { path, length, hops } = routeData;
    
    // Create route layer
    state.layers.route = L.layerGroup();
    
    // Build path coordinates
    const pathCoords = [];
    path.forEach(nodeId => {
        const node = state.nodeMap[nodeId];
        if (node && state.transform) {
            pathCoords.push(state.transform(node.x, node.y));
        }
    });
    
    // Draw route line (with glow effect)
    const glowLine = L.polyline(pathCoords, {
        color: CONFIG.COLORS.routeGlow,
        weight: 10,
        opacity: 0.3
    });
    
    const routeLine = L.polyline(pathCoords, {
        color: CONFIG.COLORS.route,
        weight: 5,
        opacity: 0.9,
        dashArray: '10, 5'
    });
    
    state.layers.route.addLayer(glowLine);
    state.layers.route.addLayer(routeLine);
    
    // Add start marker
    const startIcon = L.divIcon({
        html: '<div class="route-marker start">🟢</div>',
        className: 'route-marker-icon',
        iconSize: [30, 30]
    });
    const startMarker = L.marker(pathCoords[0], { icon: startIcon });
    state.layers.route.addLayer(startMarker);
    state.routeMarkers.push(startMarker);
    
    // Add end marker
    const endIcon = L.divIcon({
        html: '<div class="route-marker end">🔴</div>',
        className: 'route-marker-icon',
        iconSize: [30, 30]
    });
    const endMarker = L.marker(pathCoords[pathCoords.length - 1], { icon: endIcon });
    state.layers.route.addLayer(endMarker);
    state.routeMarkers.push(endMarker);
    
    // Add to map
    state.layers.route.addTo(state.map);
    
    // Fit bounds to show entire route
    state.map.fitBounds(L.latLngBounds(pathCoords), { padding: [50, 50] });
    
    // Update route info panel
    const routeInfo = document.getElementById('route-info');
    document.getElementById('route-distance').textContent = `${length.toFixed(2)} units`;
    document.getElementById('route-time').textContent = `${(length * 2).toFixed(1)} min`;
    document.getElementById('route-hops').textContent = `${hops} nodes`;
    routeInfo.classList.add('active');
}

// NEW: Clear route layer
function clearRouteLayer() {
    if (state.layers.route) {
        state.map.removeLayer(state.layers.route);
        state.layers.route = null;
    }
    state.routeMarkers = [];
    state.currentRoute = null;
}

// NEW: Clear route (button handler)
function clearRoute() {
    clearRouteLayer();
    document.getElementById('route-source').value = '';
    document.getElementById('route-target').value = '';
    document.getElementById('route-info').classList.remove('active');
    showToast('Route cleared', 'info');
}

// Navigate to analysis page with current state
function goToAnalysis() {
    // Check if prediction has been run
    if (!state.predictions) {
        showToast('Please run prediction first before viewing analysis!', 'error');
        return;
    }
    
    // Save current state to localStorage including predictions
    const analysisData = {
        closedRoads: Array.from(state.closedRoads),
        baseline: state.baselinePredictions,
        withClosures: state.predictions,
        timestamp: Date.now()
    };
    localStorage.setItem('cityAnalysisData', JSON.stringify(analysisData));
    
    console.log('📊 Saving analysis data:', {
        closedRoads: analysisData.closedRoads.length,
        hasBaseline: !!analysisData.baseline,
        hasWithClosures: !!analysisData.withClosures,
        baselinePredictionsCount: analysisData.baseline?.predictions?.length,
        withClosuresPredictionsCount: analysisData.withClosures?.predictions?.length
    });
    
    showToast('Opening analysis page...', 'info');
    
    // Open analysis page in new tab (so main page doesn't reload)
    window.open('analysis.html', '_blank');
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
    try {
        console.log('📡 Fetching graph data from API...');
        const response = await fetch(`${CONFIG.API_BASE}/graph`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        state.graphData = await response.json();
        
        if (state.graphData.error) {
            throw new Error(state.graphData.error);
        }
        
        // Debug: count metro edges
        const metroEdges = state.graphData.edges.filter(e => e.is_metro);
        console.log(`✅ Loaded ${state.graphData.node_count} nodes, ${state.graphData.edge_count} edges`);
        console.log(`🚇 Metro edges: ${metroEdges.length}`);
        
        // Render the graph
        console.log('🗺️ Rendering graph...');
        renderGraph();
        console.log('✅ Graph rendered successfully');
        
        showToast(`Loaded ${state.graphData.node_count} nodes`, 'success');
    } catch (error) {
        console.error('❌ Failed to load graph:', error);
        showToast('Failed to load graph data: ' + error.message, 'error');
        throw error;
    }
}

async function runPrediction() {
    showLoading('Running prediction...');
    
    try {
        // Get time multiplier first (used for both baseline and prediction)
        const timeMultiplier = CONFIG.TIME_MULTIPLIERS[state.currentHour] || 1.0;
        
        // FIRST: Get baseline prediction (no closures) for comparison
        const baselineResponse = await fetch(`${CONFIG.API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ closed_roads: [], hour: state.currentHour })
        });
        state.baselinePredictions = await baselineResponse.json();
        
        // Apply time multiplier to baseline predictions too
        state.baselinePredictions.predictions.forEach(p => {
            p.congestion = p.congestion * timeMultiplier;
        });
        state.baselinePredictions.stats.mean_congestion *= timeMultiplier;
        state.baselinePredictions.stats.max_congestion *= timeMultiplier;
        state.baselinePredictions.stats.road_mean *= timeMultiplier;
        state.baselinePredictions.stats.metro_mean *= timeMultiplier;
        state.baselineStats = state.baselinePredictions?.stats || null;
        
        // THEN: Get prediction with current road closures
        const response = await fetch(`${CONFIG.API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                closed_roads: Array.from(state.closedRoads),
                hour: state.currentHour
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Apply time-based multiplier to predictions (client-side for visualization)
        data.predictions.forEach(p => {
            p.congestion = p.congestion * timeMultiplier;
        });
        data.stats.mean_congestion *= timeMultiplier;
        data.stats.max_congestion *= timeMultiplier;
        data.stats.road_mean *= timeMultiplier;
        data.stats.metro_mean *= timeMultiplier;
        
        state.predictions = data;
        
        // Update visualization
        updatePredictionVisualization();
        const baselineStats = state.closedRoads.size > 0 ? state.baselineStats : null;
        updateStatsUI(data.stats, baselineStats);
        
        // Show result message with time info
        const hour = state.currentHour;
        const timeStr = `${hour === 0 ? 12 : (hour > 12 ? hour - 12 : hour)}${hour < 12 ? 'AM' : 'PM'}`;
        
        if (state.closedRoads.size > 0) {
            console.log('=== PREDICTION WITH ROAD CLOSURES ===');
            console.log(`Closed roads: ${Array.from(state.closedRoads).join(', ')}`);
            console.log(`Time: ${timeStr}, Multiplier: ${timeMultiplier}x`);
            showToast(`Prediction complete! Click "View Analysis" to see detailed impact.`, 'success');
        } else {
            showToast(`Prediction at ${timeStr}: Current traffic levels`, 'success');
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
    if (!state.graphData) {
        console.error('❌ renderGraph: No graph data available');
        return;
    }
    
    const { nodes, edges } = state.graphData;
    console.log(`🗺️ renderGraph: Processing ${nodes.length} nodes and ${edges.length} edges`);
    
    if (!state.map) {
        console.error('❌ renderGraph: Map not initialized');
        return;
    }
    
    // Calculate bounds of graph coordinates
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    // Build node map for quick lookups (NEW)
    state.nodeMap = {};
    nodes.forEach(node => {
        state.nodeMap[node.id] = node;
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
    
    // Transform function: maps graph coords to real-world lat/lng (store in state)
    if (CONFIG.USE_REAL_MAP) {
        // Map to real Pune coordinates
        // Scale to cover roughly 5km x 5km area (0.045 degrees ~ 5km)
        const scaleFactor = 0.045 / Math.max(rangeX, rangeY);
        state.transform = (x, y) => [
            CONFIG.MAP_CENTER[0] + (y - centerY) * scaleFactor,
            CONFIG.MAP_CENTER[1] + (x - centerX) * scaleFactor
        ];
    } else {
        // Simple scaling for abstract view
        const scale = 10;
        state.transform = (x, y) => [y * scale, x * scale];
    }
    
    const transform = state.transform;
    
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
        const source = state.nodeMap[edge.source];
        const target = state.nodeMap[edge.target];
        
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
        
        // Store edge data for CTM updates
        polyline.feature = {
            source: edge.source,
            target: edge.target,
            is_metro: isMetro
        };
        
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
    
    console.log(`📊 Layer counts - Roads: ${state.layers.roads.getLayers().length}, Metro: ${state.layers.metro.getLayers().length}, Nodes: ${state.layers.nodes.getLayers().length}`);
    
    // Add visible layers to map (amenities ON by default now)
    if (state.visible.roads) {
        state.layers.roads.addTo(state.map);
        console.log('✅ Roads layer added to map');
    }
    if (state.visible.metro) {
        state.layers.metro.addTo(state.map);
        console.log('✅ Metro layer added to map');
    }
    if (state.visible.nodes) {
        state.layers.nodes.addTo(state.map);
        console.log('✅ Nodes layer added to map');
    }
    state.layers.amenities.addTo(state.map);  // Always show amenities
    
    // Fit bounds to show all nodes
    if (CONFIG.USE_REAL_MAP) {
        // Create bounds from transformed node positions
        const allPositions = nodes.map(n => transform(n.x, n.y));
        const bounds = L.latLngBounds(allPositions);
        console.log('📍 Fitting bounds:', bounds.toBBoxString());
        state.map.fitBounds(bounds, { padding: [50, 50] });
    } else {
        const scale = 10;
        const bounds = [
            [minY * scale, minX * scale],
            [maxY * scale, maxX * scale]
        ];
        state.map.fitBounds(bounds, { padding: [50, 50] });
    }
    
    // Force map refresh to ensure tiles and layers display properly
    setTimeout(() => {
        state.map.invalidateSize();
        console.log('🔄 Map size invalidated');
    }, 100);
    
    console.log('✅ Graph rendering complete');
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
    if (!state.predictions || !state.graphData) {
        console.warn('Cannot update prediction visualization: missing data');
        return;
    }
    
    const predictions = state.predictions.predictions;
    if (!predictions || predictions.length === 0) {
        console.warn('No predictions to visualize');
        return;
    }
    
    console.log(`Visualizing ${predictions.length} predictions`);
    
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
    let visualizedCount = 0;
    
    state.graphData.edges.forEach(edge => {
        if (!edge._polyline) {
            console.warn(`Edge ${edge.source}-${edge.target} has no polyline reference`);
            return;
        }
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
        visualizedCount++;
    });
    
    console.log(`✅ Visualization complete: ${visualizedCount} roads colored`);
    if (hasClosures) {
        console.log(`📊 Traffic redistribution: ${increasedCount} roads got busier, ${decreasedCount} roads got less busy`);
    } else {
        console.log(`📊 Showing current traffic levels (no road closures)`);
    }
}

// ============================================================
// ROAD CLOSURE
// ============================================================

async function toggleRoadClosure(edge) {
    const roadId = `${edge.source}-${edge.target}`;
    
    if (state.closedRoads.has(roadId)) {
        // Reopen the road
        try {
            const response = await fetch(`${CONFIG.API_BASE}/ctm/reopen-road`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source: edge.source, target: edge.target })
            });
            const data = await response.json();
            if (data.error) {
                console.warn('CTM reopen-road:', data.error);
            }
        } catch (e) {
            console.warn('CTM not active, updating UI only');
        }
        
        state.closedRoads.delete(roadId);
        edge._polyline.setStyle({
            color: CONFIG.COLORS.road,
            weight: 2,
            dashArray: null
        });
        showToast(`Opened road: ${roadId}`, 'info');
    } else {
        // Close the road
        try {
            const response = await fetch(`${CONFIG.API_BASE}/ctm/close-road`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source: edge.source, target: edge.target })
            });
            const data = await response.json();
            if (data.error) {
                console.warn('CTM close-road:', data.error);
            }
        } catch (e) {
            console.warn('CTM not active, updating UI only');
        }
        
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

// NEW: Remove a node from the simulation
async function removeNode(nodeId) {
    if (state.removedNodes.has(nodeId)) {
        showToast(`Node ${nodeId} is already removed`, 'warning');
        return;
    }
    
    showLoading(`Analyzing traffic impact of removing node ${nodeId}...`);
    
    try {
        // Call backend API to analyze node removal impact
        const response = await fetch(`${CONFIG.API_BASE}/analyze-node-removal`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: nodeId,
                hour: state.currentHour
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Store the impact analysis
        state.removedNodes.add(nodeId);
        state.nodeImpactAnalysis[nodeId] = data.impact_analysis;
        
        // Close all edges connected to this node
        const affectedEdges = data.affected_edges;
        affectedEdges.forEach(roadId => {
            if (!state.closedRoads.has(roadId)) {
                state.closedRoads.add(roadId);
                
                // Update visual
                const edge = state.graphData.edges.find(e => 
                    `${e.source}-${e.target}` === roadId
                );
                if (edge && edge._polyline) {
                    edge._polyline.setStyle({
                        color: '#ff69b4',  // Pink
                        weight: 4,
                        dashArray: '10, 6'
                    });
                }
            }
        });
        
        // Update UI
        updateClosedRoadsList();
        updateRemovedNodesList();
        
        // Update predictions based on new state
        await runPrediction();
        
        // Show impact summary
        showNodeRemovalImpact(nodeId, data.impact_analysis);
        
        hideLoading();
        showToast(`Node ${nodeId} removed - Impact analysis complete`, 'success');
        
    } catch (error) {
        console.error('Node removal analysis failed:', error);
        showToast('Node removal analysis failed: ' + error.message, 'error');
        state.removedNodes.delete(nodeId);
        hideLoading();
    }
}

// NEW: Restore a removed node
async function restoreNode(nodeId) {
    if (!state.removedNodes.has(nodeId)) {
        showToast(`Node ${nodeId} was not removed`, 'warning');
        return;
    }
    
    showLoading(`Restoring node ${nodeId}...`);
    
    try {
        // Find and reopen affected edges
        const impactAnalysis = state.nodeImpactAnalysis[nodeId];
        const affectedEdges = [];
        
        // Get all edges connected to this node to find which ones were affected
        state.graphData.edges.forEach(edge => {
            const roadId = `${edge.source}-${edge.target}`;
            if (edge.source == nodeId || edge.target == nodeId) {
                affectedEdges.push(roadId);
            }
        });
        
        // Remove from state
        state.removedNodes.delete(nodeId);
        delete state.nodeImpactAnalysis[nodeId];
        
        // Reopen edges (remove from closed roads)
        affectedEdges.forEach(roadId => {
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
        });
        
        // Update UI
        updateClosedRoadsList();
        updateRemovedNodesList();
        
        // Update predictions
        await runPrediction();
        
        hideLoading();
        showToast(`Node ${nodeId} restored`, 'success');
        
    } catch (error) {
        console.error('Node restoration failed:', error);
        showToast('Node restoration failed: ' + error.message, 'error');
        hideLoading();
    }
}

// NEW: Update removed nodes list in UI
function updateRemovedNodesList() {
    const container = document.getElementById('removed-nodes-list');
    
    if (!container) return;
    
    if (state.removedNodes.size === 0) {
        container.innerHTML = '<p class="empty-message">No nodes removed</p>';
        return;
    }
    
    container.innerHTML = '';
    state.removedNodes.forEach(nodeId => {
        const item = document.createElement('div');
        item.className = 'removed-node-item';
        const impact = state.nodeImpactAnalysis[nodeId];
        const affectedEdges = impact.closed_edges_count;
        
        item.innerHTML = `
            <div class="node-info">
                <strong>Node ${nodeId}</strong>
                <small>${affectedEdges} edges closed • ${impact.node_details.amenity}</small>
            </div>
            <button class="btn-restore" onclick="restoreNode('${nodeId}')">Restore</button>
        `;
        container.appendChild(item);
    });
}

// NEW: Show detailed impact analysis for removed node
function showNodeRemovalImpact(nodeId, impact) {
    const panel = document.getElementById('info-panel');
    const title = document.getElementById('info-title');
    const content = document.getElementById('info-content');
    
    const node = impact.node_details;
    const congestionPercent = (impact.mean_congestion * 100).toFixed(1);
    const maxCongestionPercent = (impact.max_congestion * 100).toFixed(1);
    const closedEdgeCongestionPercent = (impact.mean_closed_edge_congestion * 100).toFixed(1);
    
    title.textContent = `Node ${nodeId} - Removal Impact Analysis`;
    content.innerHTML = `
        <div class="impact-analysis">
            <h5>Node Details</h5>
            <p><span class="label">Zone:</span> <span class="value">${node.zone}</span></p>
            <p><span class="label">Population:</span> <span class="value">${node.population.toLocaleString()}</span></p>
            <p><span class="label">Amenity:</span> <span class="value">${node.amenity || 'None'}</span></p>
            
            <h5 style="margin-top: 15px;">Traffic Impact</h5>
            <p><span class="label">Edges Closed:</span> <span class="value highlight">${impact.closed_edges_count}</span></p>
            <p><span class="label">Mean Congestion (Closed Edges):</span> <span class="value">${closedEdgeCongestionPercent}%</span></p>
            <p><span class="label">Max Congestion (Closed Edges):</span> <span class="value">${(impact.max_closed_edge_congestion * 100).toFixed(1)}%</span></p>
            <p><span class="label">Overall Mean Congestion:</span> <span class="value">${congestionPercent}%</span></p>
            <p><span class="label">Overall Max Congestion:</span> <span class="value">${maxCongestionPercent}%</span></p>
            
            <h5 style="margin-top: 15px;">Transport Mode Impact</h5>
            <p><span class="label">Road Average:</span> <span class="value">${(impact.road_mean * 100).toFixed(1)}%</span></p>
            <p><span class="label">Metro Average:</span> <span class="value">${(impact.metro_mean * 100).toFixed(1)}%</span></p>
        </div>
    `;
    
    panel.classList.add('visible');
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

function updateStatsUI(stats, baselineStats = null) {
    if (!stats) return;
    
    const toPercent = (val) => {
        if (val === undefined || val === null) return null;
        const num = Number(val);
        if (Number.isNaN(num)) return null;
        return num > 1 ? num : num * 100;
    };
    
    const formatValue = (val) => {
        const pct = toPercent(val);
        if (pct === null) return '--';
        return `${pct.toFixed(1)}%`;
    };
    
    const formatDelta = (current, baseline) => {
        if (!baselineStats) return '';
        const currentPct = toPercent(current);
        const baselinePct = toPercent(baseline);
        if (currentPct === null || baselinePct === null) return '';
        const delta = currentPct - baselinePct;
        if (Math.abs(delta) < 0.1) {
            return `<span class="stat-delta neutral">No change</span>`;
        }
        const direction = delta > 0 ? 'positive' : 'negative';
        const arrow = delta > 0 ? '▲' : '▼';
        return `<span class="stat-delta ${direction}">${arrow} ${Math.abs(delta).toFixed(1)} pp</span>`;
    };
    
    const setStatValue = (elementId, currentValue, baselineValue) => {
        const el = document.getElementById(elementId);
        if (!el) return;
        const deltaHtml = baselineValue !== undefined && baselineValue !== null
            ? formatDelta(currentValue, baselineValue)
            : '';
        el.innerHTML = `${formatValue(currentValue)}${deltaHtml}`;
    };
    
    setStatValue('stat-mean', stats.mean_congestion, baselineStats?.mean_congestion);
    setStatValue('stat-max', stats.max_congestion, baselineStats?.max_congestion);
    setStatValue('stat-road', stats.road_mean, baselineStats?.road_mean);
    setStatValue('stat-metro', stats.metro_mean, baselineStats?.metro_mean);
}

function showNodeInfo(node) {
    const panel = document.getElementById('info-panel');
    const title = document.getElementById('info-title');
    const content = document.getElementById('info-content');
    
    title.textContent = `Node ${node.id}`;
    
    const isRemoved = state.removedNodes.has(node.id);
    const removeButtonText = isRemoved ? 'Restore Node' : 'Remove Node';
    const removeButtonClass = isRemoved ? 'btn-restore-node' : 'btn-remove-node';
    const removeButtonAction = isRemoved ? `restoreNode('${node.id}')` : `removeNode('${node.id}')`;
    
    content.innerHTML = `
        <p><span class="label">Zone:</span> <span class="value">${node.zone}</span></p>
        <p><span class="label">Population:</span> <span class="value">${node.population.toLocaleString()}</span></p>
        <p><span class="label">Amenity:</span> <span class="value">${node.amenity || 'None'}</span></p>
        <p><span class="label">Metro Station:</span> <span class="value">${node.is_metro ? 'Yes' : 'No'}</span></p>
        <p><span class="label">Position:</span> <span class="value">(${node.x.toFixed(2)}, ${node.y.toFixed(2)})</span></p>
        <div style="margin-top: 15px; display: flex; gap: 10px;">
            <button class="btn ${removeButtonClass}" onclick="${removeButtonAction}">
                <i class="fas fa-${isRemoved ? 'undo' : 'trash'}"></i> ${removeButtonText}
            </button>
        </div>
    `;
    
    panel.classList.add('visible');
}

function resetSimulation() {
    // Clear all closures
    state.closedRoads.clear();
    state.predictions = null;
    state.baselinePredictions = null;
    state.baselineStats = null;
    
    // Hide View Analysis button
    const analysisBtn = document.getElementById('view-analysis-btn');
    if (analysisBtn) analysisBtn.style.display = 'none';
    
    // Re-render graph to remove highlighting
    if (state.graphData) {
        renderGraph();
    }
    
    // Reset stats
    document.getElementById('stat-mean').textContent = '--';
    document.getElementById('stat-max').textContent = '--';
    document.getElementById('stat-road').textContent = '--';
    document.getElementById('stat-metro').textContent = '--';
    
    // Update closed roads list
    updateClosedRoadsList();
    
    showToast('Simulation reset', 'success');
}

// ============================================================
// CTM (Cell Transmission Model) FUNCTIONS
// ============================================================

async function initCTM() {
    showLoading('Initializing CTM simulation...');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/ctm/initialize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                closed_edges: Array.from(state.closedRoads).map(road => {
                    const [source, target] = road.split('-');
                    return [source, target, 0];
                })
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        console.log('✅ CTM initialized:', data);
        showToast(`CTM initialized: ${data.total_cells} cells`, 'success');
        
        // Update UI to show CTM is active
        const ctmStatus = document.getElementById('ctm-status');
        if (ctmStatus) {
            ctmStatus.textContent = 'Active';
            ctmStatus.classList.add('active');
        }
        
    } catch (error) {
        console.error('❌ CTM initialization failed:', error);
        showToast('CTM initialization failed: ' + error.message, 'error');
    }
    
    hideLoading();
}

async function stepCTM(steps = 1) {
    showLoading(`Running ${steps} CTM step(s)...`);
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/ctm/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ steps: steps })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        console.log('✅ CTM step complete:', data);
        
        // Update visualization with CTM data
        if (data.edge_congestion) {
            updateCTMVisualization(data.edge_congestion);
        }
        
        // Update CTM status display
        const ctmTime = document.getElementById('ctm-time');
        if (ctmTime) {
            ctmTime.textContent = `${Math.round(data.simulation_time || 0)} min`;
        }
        
        const ctmVehicles = document.getElementById('ctm-vehicles');
        if (ctmVehicles) {
            ctmVehicles.textContent = (data.total_vehicles || 0).toLocaleString();
        }
        
        const ctmDensity = document.getElementById('ctm-density');
        if (ctmDensity && data.stats) {
            ctmDensity.textContent = `${(data.stats.average_congestion * 100).toFixed(1)}%`;
        }
        
        showToast(`CTM: ${Math.round(data.simulation_time)} min, ${data.total_vehicles.toLocaleString()} vehicles`, 'success');
        
    } catch (error) {
        console.error('❌ CTM step failed:', error);
        showToast('CTM step failed: ' + error.message, 'error');
    }
    
    hideLoading();
}

async function resetCTM() {
    showLoading('Resetting CTM simulation...');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/ctm/reset`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        console.log('✅ CTM reset:', data);
        showToast('CTM simulation reset', 'success');
        
        // Reset visualization
        renderGraph();
        
        // Reset CTM status display
        const ctmTime = document.getElementById('ctm-time');
        if (ctmTime) ctmTime.textContent = '0 min';
        
        const ctmVehicles = document.getElementById('ctm-vehicles');
        if (ctmVehicles) ctmVehicles.textContent = '0';
        
        const ctmDensity = document.getElementById('ctm-density');
        if (ctmDensity) ctmDensity.textContent = '0%';
        
    } catch (error) {
        console.error('❌ CTM reset failed:', error);
        showToast('CTM reset failed: ' + error.message, 'error');
    }
    
    hideLoading();
}

function updateCTMVisualization(edgeCongestion) {
    if (!state.graphData || !state.layers.roads) return;
    
    console.log('Updating CTM visualization with', Object.keys(edgeCongestion).length, 'edges');
    
    // Update road colors based on CTM congestion
    state.layers.roads.eachLayer(polyline => {
        if (!polyline.feature) return;
        
        // Try both key formats
        const key1 = `${polyline.feature.source}-${polyline.feature.target}`;
        const key2 = `${polyline.feature.source}-${polyline.feature.target}-0`;
        const congestion = edgeCongestion[key1] ?? edgeCongestion[key2];
        
        if (congestion !== undefined) {
            const color = getCongestionColor(congestion);
            polyline.setStyle({ color: color });
        }
    });
}

function getCongestionColor(congestion) {
    // Map congestion (0-1) to color gradient (green -> yellow -> red)
    if (congestion < 0.3) {
        return CONFIG.COLORS.lowCongestion;  // Green
    } else if (congestion < 0.6) {
        return '#f1c40f';  // Yellow
    } else if (congestion < 0.8) {
        return '#e67e22';  // Orange
    } else {
        return CONFIG.COLORS.highCongestion;  // Red
    }
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

// NEW: Make node removal functions available globally for onclick
window.removeNode = removeNode;
window.restoreNode = restoreNode;
