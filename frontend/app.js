// GNN City Simulator - PUNE Interactive Map
const API_BASE = 'http://localhost:5000';

// State Management
const state = {
  map: null,
  currentScenario: null,
  scenarioResults: null,
  selectedEdge: null,
  selectedNode: null,
  layers: {
    roads: null,
    metros: null,
    amenities: null
  },
  cityData: null,
  edgeMap: {},           // Map edge IDs to edge objects for quick lookup
  nodeMap: {},           // Map node IDs to node objects
  originalColors: {},    // Store original edge colors
  predictionsBefore: [],
  predictionsAfter: [],
  currentPredictions: []
};

// Initialize app on DOM load
document.addEventListener('DOMContentLoaded', async () => {
  console.log('🚀 Initializing Pune GNN City Simulator...');
  
  try {
    // 1. Load city data from backend or local file
    console.log('📥 Attempting to load graph data...');
    try {
      // Try local JSON first (full geometry)
      const localResponse = await fetch('../graph_full_data.json');
      if (localResponse.ok) {
        state.cityData = await localResponse.json();
        console.log('✅ Loaded from local graph_full_data.json:', {
          nodes: state.cityData.nodes?.length,
          edges: state.cityData.edges?.length
        });
      } else {
        throw new Error('Local file not available');
      }
    } catch (e) {
      // Fallback to backend
      try {
        const backendResponse = await fetch(`${API_BASE}/city-data`);
        if (backendResponse.ok) {
          state.cityData = await backendResponse.json();
          console.log('✅ Loaded from backend:', {
            nodes: state.cityData.nodes?.length,
            edges: state.cityData.edges?.length
          });
        } else {
          throw new Error('Backend unavailable');
        }
      } catch (backendError) {
        console.warn('⚠️ Could not load city data:', backendError.message);
      }
    }
    
    // Build data maps
    if (state.cityData) {
      (state.cityData.edges || []).forEach(edge => {
        state.edgeMap[edge.id] = edge;
        state.originalColors[edge.id] = getCongestionColor(edge.congestion);
      });
      (state.cityData.nodes || []).forEach(node => {
        state.nodeMap[node.id] = node;
      });
      console.log('✅ Data maps built');
    }
    
    // 2. Initialize map
    initializeMap();
    
    // 3. Render all roads and POIs
    renderRoadNetwork();
    
    // 4. Setup event listeners
    setupEventListeners();
    
    // 5. Check backend health
    checkBackendHealth();
  } catch (error) {
    console.error('❌ Initialization failed:', error);
    alert('Error initializing application: ' + error.message);
  }
});

/**
 * Initialize Leaflet Map - Centered on Pune, India
 */
function initializeMap() {
  // Pune, India center coordinates
  const puneCenter = [18.55, 73.85];
  
  state.map = L.map('map', {
    center: puneCenter,
    zoom: 12,
    zoomControl: true,
    attributionControl: true,
    preferCanvas: true
  });

  // Add OpenStreetMap layer (good for India)
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
    minZoom: 10
  }).addTo(state.map);

  // Initialize feature groups
  state.layers.roads = L.featureGroup().addTo(state.map);
  state.layers.metros = L.featureGroup().addTo(state.map);
  state.layers.amenities = L.featureGroup().addTo(state.map);

  console.log('✅ Leaflet map initialized for Pune');
}

/**
 * Load Pune city data from backend
 */

/**
 * Render Pune road network with all roads interactive
 */
function renderRoadNetwork() {
  if (!state.cityData) {
    console.warn('No city data available');
    return;
  }

  console.log('🗺️ Rendering road network...');
  state.layers.roads.clearLayers();
  state.layers.metros.clearLayers();
  state.layers.amenities.clearLayers();

  const { nodes, edges, metros, amenities } = state.cityData;

  // Render all road edges
  if (edges && Array.isArray(edges)) {
    let renderedCount = 0;
    
    edges.forEach((edge, idx) => {
      try {
        // Calculate congestion color and width
        const congestion = edge.congestion || 0;
        const color = getCongestionColor(congestion);
        const weight = 1.5 + (congestion * 3);  // Road width based on congestion
        const opacity = 0.7;

        // Use full geometry coordinates if available, otherwise simple line
        let coordinates = [];
        if (edge.coordinates && Array.isArray(edge.coordinates)) {
          // Full geometry: convert coordinate pairs to [lat, lon] for Leaflet
          coordinates = edge.coordinates.map(c => [c[0], c[1]]);
        } else {
          // Fallback: simple line between source and target
          const sourceNode = state.nodeMap[edge.source];
          const targetNode = state.nodeMap[edge.target];
          
          if (!sourceNode || !targetNode) return;
          coordinates = [
            [sourceNode.lat, sourceNode.lon],
            [targetNode.lat, targetNode.lon]
          ];
        }

        // Create polyline for road with full geometry
        const polyline = L.polyline(
          coordinates,
          {
            color: color,
            weight: weight,
            opacity: opacity,
            className: `edge-${edge.id}`,
            dashArray: edge.oneway ? '5,5' : 'none',  // Dashed lines for one-way roads
            lineCap: 'round',
            lineJoin: 'round'
          }
        );

        // Add popup with road info
        const roadInfo = `
          <div style="font-size: 12px; font-weight: bold; color: #1e88e5;">🛣️ ${edge.name || 'Road Segment'}</div>
          <hr style="margin: 4px 0; border: none; border-top: 1px solid #ddd;">
          <small>
            <div><strong>Type:</strong> ${edge.highway || 'unknown'}</div>
            <div><strong>Length:</strong> ${(edge.length / 1000).toFixed(2)} km</div>
            <div><strong>Congestion:</strong> <span style="color: ${color}"><strong>${(edge.congestion * 100).toFixed(1)}%</strong></span></div>
            ${edge.oneway ? '<div><strong>⬆️ One-way road</strong></div>' : ''}
            <div><strong>Edge ID:</strong> ${edge.id}</div>
          </small>
        `;
        polyline.bindPopup(roadInfo);

        // Add hover effects
        polyline.on('mouseover', function() {
          this.setStyle({ weight: weight * 1.5, opacity: 1.0 });
        });
        polyline.on('mouseout', function() {
          this.setStyle({ weight: weight, opacity: opacity });
        });

        // Add click handler to select road
        polyline.on('click', function(e) {
          L.DomEvent.stopPropagation(e);
          selectRoad(edge.id);
        });

        // Add to map
        polyline.addTo(state.layers.roads);
        renderedCount++;
        
      } catch (error) {
        // Silently skip problematic edges
      }
    });
    
    console.log(`✅ Rendered ${renderedCount} road segments`);
  }

  // Render metro stations
  if (metros && Array.isArray(metros)) {
    metros.forEach(metro => {
      try {
        const icon = L.icon({
          iconUrl: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxNiIgY3k9IjE2IiByPSIxNCIgZmlsbD0iI0ZGQzEwNyIgc3Ryb2tlPSIjRkY5ODAwIiBzdHJva2Utd2lkdGg9IjIiLz48dGV4dCB4PSIxNiIgeT0iMjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiMzMzMzMzMiIGZvbnQtd2VpZ2h0PSJib2xkIiBmb250LXNpemU9IjE0Ij5N PC90ZXh0Pjwvc3ZnPg==',
          iconSize: [32, 32],
          iconAnchor: [16, 16],
          popupAnchor: [0, -16]
        });
        
        const marker = L.marker([metro.lat, metro.lon], { icon, title: metro.name });
        marker.bindPopup(`
          <div style="font-size: 12px;">
            <strong>🚇 ${metro.name || 'Metro Station'}</strong><br>
            Amenity Type: ${metro.amenity_type}
          </div>
        `);
        
        state.layers.metros.addLayer(marker);
      } catch (error) {
        console.warn('Error rendering metro:', error);
      }
    });
    
    console.log(`✅ Rendered ${metros.length} metro stations`);
  }

  // Render amenities (hospitals, schools, etc.)
  if (amenities && Array.isArray(amenities)) {
    amenities.forEach(amenity => {
      try {
        const icon = L.icon({
          iconUrl: getAmenityIcon(amenity.amenity_type),
          iconSize: [24, 24],
          iconAnchor: [12, 12],
          popupAnchor: [0, -12]
        });
        
        const marker = L.marker([amenity.lat, amenity.lon], { icon, title: amenity.name });
        marker.bindPopup(`
          <div style="font-size: 12px;">
            <strong>${amenity.name || 'POI'}</strong><br>
            Type: ${amenity.amenity_type}
          </div>
        `);
        
        state.layers.amenities.addLayer(marker);
      } catch (error) {
        console.warn('Error rendering amenity:', error);
      }
    });
    
    console.log(`✅ Rendered ${amenities.length} amenities`);
  }

  // Fit map to bounds if we have nodes
  if (nodes && nodes.length > 0) {
    const bounds = L.latLngBounds(
      nodes.map(n => [n.lat, n.lon])
    );
    state.map.fitBounds(bounds, { padding: [50, 50] });
  }

  console.log('🎉 Road network rendering complete!');
}

/**
 * Calculate statistics from predictions
 */
function calculateStats(predictions) {
  if (!predictions || predictions.length === 0) {
    return { average: 0, max: 0, min: 0, total: 0 };
  }

  const sum = predictions.reduce((a, b) => a + b, 0);
  return {
    average: sum / predictions.length,
    max: Math.max(...predictions),
    min: Math.min(...predictions),
    total: predictions.length
  };
}

/**
 * Reset scenario and restore original road colors
 */
function resetScenario() {
  console.log('🔄 Resetting scenario...');
  
  // Clear state
  state.selectedEdge = null;
  state.predictionsBefore = [];
  state.predictionsAfter = [];
  state.scenarioResults = null;

  // Restore original road colors
  const edges = state.cityData?.edges || [];
  edges.forEach((edge, idx) => {
    const originalColor = getCongestionColor(edge.congestion);
    const element = document.querySelector(`.edge-${edge.id}`);
    if (element) {
      element.setAttribute('stroke', originalColor);
      element.setAttribute('stroke-width', 1.5 + (edge.congestion * 3));
    }
  });

  // Clear UI
  document.querySelector('.selection-info').innerHTML = '<p style="color: #999;">No road selected</p>';
  document.querySelector('.quick-stats').innerHTML = '';

  console.log('✅ Scenario reset');
}

/**
 * Get color based on congestion level
 */
function getCongestionColor(congestion) {
  if (congestion < 0.2) return '#4CAF50'; // Green
  if (congestion < 0.4) return '#8BC34A'; // Light green
  if (congestion < 0.6) return '#FFC107'; // Yellow
  if (congestion < 0.8) return '#FF9800'; // Orange
  return '#FF5722'; // Red
}

/**
 * Get SVG icon for amenity type
 */
function getAmenityIcon(type) {
  const icons = {
    hospital: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB4PSIyIiB5PSIyIiB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHJ4PSIyIiBmaWxsPSIjRkY1NzIyIi8+PHBhdGggZD0iTTEwIDYuNVYxMy41TTE2LjUgMTBIMy41IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9zdmc+',
    school: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMiAxNEw5IDMuNUwxNiAxNEgyTTUgMTRWMTBIMTNWMTQiIHN0cm9rZT0iIzJFRjQxRiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGZpbGw9Im5vbmUiLz48L3N2Zz4=',
    park: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMCIgY3k9IjEwIiByPSI4IiBmaWxsPSIjNENBRjUwIi8+PHRleHQgeD0iMTAiIHk9IjEzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSJ3aGl0ZSIgZm9udC1zaXplPSI3Ij7wn4yo PC90ZXh0Pjwvc3ZnPg==',
    restaurant: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMCIgY3k9IjEwIiByPSI4IiBmaWxsPSIjRkY5OTAwIi8+PGNpcmNsZSBjeD0iOCIgY3k9IjgiIHI9IjEuNSIgZmlsbD0id2hpdGUiLz48L3N2Zz4=',
    default: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMCIgY3k9IjEwIiByPSI4IiBmaWxsPSIjMWU4OGU1Ii8+PC9zdmc+'
  };
  return icons[type] || icons.default;
}

/**
 * Select a road edge for analysis
 */
function selectRoad(edgeId) {
  state.selectedEdge = edgeId;
  const edge = state.edgeMap[edgeId];
  
  if (edge) {
    const infoPanel = document.querySelector('.selection-info');
    infoPanel.innerHTML = `
      <div style="text-align: left;">
        <strong style="color: #1e88e5;">🛣️ Road Selected</strong><br>
        <small>Edge ID: ${edgeId}</small><br>
        <small>Length: ${edge.length?.toFixed(1) || 'N/A'} m</small><br>
        <small>Current Congestion: <strong style="color: ${getCongestionColor(edge.congestion)}">${(edge.congestion * 100).toFixed(1)}%</strong></small>
      </div>
    `;
    console.log('Selected road:', edge);
  }
}

/**
 * Select node and update panel
 */
function selectNode(nodeId, type) {
  state.selectedNode = { id: nodeId, type };
  updateSelectionInfo();
}

/**
 * Update selection info panel
 */
function updateSelectionInfo() {
  const infoPanel = document.querySelector('.selection-info');
  if (state.selectedNode) {
    infoPanel.innerHTML = `
      <div>
        <strong>${state.selectedNode.type.toUpperCase()}</strong><br>
        ID: ${state.selectedNode.id}<br>
        <small class="text-muted">Click roads for detailed analysis</small>
      </div>
    `;
  } else {
    infoPanel.innerHTML = '<small class="text-muted">Click roads/amenities on map</small>';
  }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
  // Run scenario button
  document.getElementById('runScenario').addEventListener('click', runScenario);
  
  // Detailed analysis button
  document.getElementById('detailedAnalysis').addEventListener('click', showDetailedAnalysis);
  
  // Reset scenario button
  document.getElementById('resetScenario').addEventListener('click', resetScenario);
  
  // Close modal button
  document.getElementById('closeAnalysis').addEventListener('click', closeModal);
  
  // Close modal when clicking outside
  document.getElementById('analysisModal').addEventListener('click', (e) => {
    if (e.target.id === 'analysisModal') {
      closeModal();
    }
  });
  
  // Update severity label
  document.getElementById('severity').addEventListener('input', (e) => {
    document.getElementById('severityValue').textContent = (e.target.value * 100).toFixed(0) + '%';
  });
}

/**
 * Run scenario on Pune road network
 */
async function runScenario() {
  const scenarioType = document.getElementById('scenarioType').value;
  const severity = parseFloat(document.getElementById('severity').value);
  const duration = parseInt(document.getElementById('duration').value);

  console.log('🚀 Running scenario:', { scenarioType, severity, duration });

  // Collect current road congestions as features
  const features = state.cityData.edges?.map(e => e.congestion) || [];
  
  if (features.length === 0) {
    alert('No road data available');
    return;
  }

  try {
    // Get baseline prediction
    console.log('📊 Getting baseline predictions...');
    const beforeResponse = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });

    if (!beforeResponse.ok) throw new Error('Prediction failed');
    const beforeData = await beforeResponse.json();
    state.predictionsBefore = beforeData.predictions || [];

    // Run what-if scenario
    console.log('🔄 Running what-if analysis...');
    const whatifResponse = await fetch(`${API_BASE}/whatif`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        features,
        scenario: {
          type: scenarioType,
          severity: severity / 10,  // Convert 1-10 to 0-1
          duration: duration
        }
      })
    });

    if (!whatifResponse.ok) throw new Error('What-if analysis failed');
    const result = await whatifResponse.json();
    state.scenarioResults = result;
    state.predictionsAfter = result.after || result.predictions || [];

    // Update map visualization
    updateMapWithPredictions();
    
    // Update quick stats
    updateQuickStats();
    
    console.log('✅ Scenario completed!');
    
  } catch (error) {
    console.error('❌ Error running scenario:', error);
    alert('Error: ' + error.message);
  }
}

/**
 * Update map roads with new predictions
 */
function updateMapWithPredictions() {
  if (!state.predictionsAfter || state.predictionsAfter.length === 0) {
    console.warn('No predictions to display');
    return;
  }

  const edges = state.cityData?.edges || [];
  
  // Update each road segment's color
  edges.forEach((edge, idx) => {
    if (idx < state.predictionsAfter.length) {
      const newCongestion = state.predictionsAfter[idx];
      const newColor = getCongestionColor(newCongestion);
      const newWeight = 1.5 + (newCongestion * 3);
      
      // Find and update the polyline in the DOM
      const element = document.querySelector(`.edge-${edge.id}`);
      if (element) {
        element.setAttribute('stroke', newColor);
        element.setAttribute('stroke-width', newWeight);
      }
    }
  });

  console.log('✅ Map visualization updated with predictions');
}

/**
 * Update quick stats panel
 */
function updateQuickStats() {
  const before = calculateStats(state.predictionsBefore);
  const after = calculateStats(state.predictionsAfter);
  const changePercent = ((after.average - before.average) / before.average * 100).toFixed(1);

  const html = `
    <div class="stat-item">
      <span class="stat-label">Avg Congestion Before</span>
      <span class="stat-value">${(before.average * 100).toFixed(1)}%</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Avg Congestion After</span>
      <span class="stat-value">${(after.average * 100).toFixed(1)}%</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Change</span>
      <span class="stat-value" style="color: ${changePercent > 0 ? '#FF5722' : '#4CAF50'}">
        ${changePercent > 0 ? '+' : ''}${changePercent}%
      </span>
    </div>
  `;

  document.querySelector('.quick-stats').innerHTML = html;
}

/**
 * Calculate statistics
 */
function calculateStats(predictions) {
  if (!predictions || predictions.length === 0) {
    return { average: 0, max: 0, min: 0 };
  }
  
  return {
    average: predictions.reduce((a, b) => a + b, 0) / predictions.length,
    max: Math.max(...predictions),
    min: Math.min(...predictions)
  };
}

/**
 * Show detailed analysis modal
 */
function showDetailedAnalysis() {
  if (!state.scenarioResults) {
    alert('Please run a scenario first');
    return;
  }

  const modal = document.getElementById('analysisModal');
  modal.classList.add('active');

  // Render charts
  setTimeout(() => {
    renderAnalysisCharts();
    renderStatsTable();
  }, 100);
}

/**
 * Render analysis charts
 */
function renderAnalysisCharts() {
  const stats1 = calculateStats(state.predictionsBefore);
  const stats2 = calculateStats(state.predictionsAfter);

  // Chart 1: Congestion Comparison
  const ctx1 = document.getElementById('comparisonChart').getContext('2d');
  new Chart(ctx1, {
    type: 'bar',
    data: {
      labels: ['Before', 'After'],
      datasets: [{
        label: 'Average Congestion (%)',
        data: [(stats1.average * 100).toFixed(1), (stats2.average * 100).toFixed(1)],
        backgroundColor: ['#1e88e5', '#ff6b6b'],
        borderColor: ['#1565c0', '#ff5252'],
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#e2e8f0' } }
      },
      scales: {
        y: { ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' } },
        x: { ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' } }
      }
    }
  });

  // Chart 2: Distribution
  const ctx2 = document.getElementById('distributionChart').getContext('2d');
  new Chart(ctx2, {
    type: 'line',
    data: {
      labels: state.predictionsBefore.map((_, i) => `Road ${i + 1}`),
      datasets: [
        {
          label: 'Before Scenario',
          data: state.predictionsBefore.map(p => (p * 100).toFixed(1)),
          borderColor: '#1e88e5',
          backgroundColor: 'rgba(30, 136, 229, 0.1)',
          tension: 0.3
        },
        {
          label: 'After Scenario',
          data: state.predictionsAfter.map(p => (p * 100).toFixed(1)),
          borderColor: '#ff6b6b',
          backgroundColor: 'rgba(255, 107, 107, 0.1)',
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#e2e8f0' } }
      },
      scales: {
        y: { ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' } },
        x: { ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' } }
      }
    }
  });
}

/**
 * Render stats table
 */
function renderStatsTable() {
  const before = calculateStats(state.predictionsBefore);
  const after = calculateStats(state.predictionsAfter);

  const html = `
    <h4>Impact Summary</h4>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Before</th>
          <th>After</th>
          <th>Change</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Average Congestion</td>
          <td>${(before.average * 100).toFixed(1)}%</td>
          <td>${(after.average * 100).toFixed(1)}%</td>
          <td style="color: ${after.average > before.average ? '#FF5722' : '#4CAF50'}">${((after.average - before.average) * 100).toFixed(1)}%</td>
        </tr>
        <tr>
          <td>Peak Congestion</td>
          <td>${(before.max * 100).toFixed(1)}%</td>
          <td>${(after.max * 100).toFixed(1)}%</td>
          <td style="color: ${after.max > before.max ? '#FF5722' : '#4CAF50'}">${((after.max - before.max) * 100).toFixed(1)}%</td>
        </tr>
        <tr>
          <td>Minimum Congestion</td>
          <td>${(before.min * 100).toFixed(1)}%</td>
          <td>${(after.min * 100).toFixed(1)}%</td>
          <td style="color: ${after.min > before.min ? '#FF5722' : '#4CAF50'}">${((after.min - before.min) * 100).toFixed(1)}%</td>
        </tr>
      </tbody>
    </table>
  `;

  document.querySelector('.stats-table').innerHTML = html;
}

/**
 * Reset scenario
 */
function resetScenario() {
  state.currentScenario = null;
  state.scenarioResults = null;
  state.predictionsBefore = [];
  state.predictionsAfter = [];
  
  // Reset map to original state
  renderCityLayers();
  
  // Reset quick stats
  document.querySelector('.quick-stats').innerHTML = '<small class="text-muted">Run a scenario to see statistics</small>';
  
  console.log('Scenario reset');
}

/**
 * Close modal
 */
function closeModal() {
  document.getElementById('analysisModal').classList.remove('active');
}

/**
 * Check backend health
 */
async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (response.ok) {
      console.log('✓ Backend is healthy');
    }
  } catch (error) {
    console.warn('⚠ Backend not accessible:', error.message);
  }
}

/**
 * Generate mock city data if backend is unavailable
 */
function generateMockCityData() {
  const nodes = [
    { id: 1, lat: 40.7128, lng: -74.0060, name: 'Central Hub' },
    { id: 2, lat: 40.7180, lng: -74.0040, name: 'North Station' },
    { id: 3, lat: 40.7100, lng: -74.0100, name: 'South Station' },
    { id: 4, lat: 40.7200, lng: -74.0150, name: 'East Terminal' },
    { id: 5, lat: 40.7050, lng: -74.0000, name: 'West Terminal' }
  ];

  const edges = [
    { id: 1, source: 1, target: 2, flow: 450, congestion: 0.3 },
    { id: 2, source: 1, target: 3, flow: 380, congestion: 0.25 },
    { id: 3, source: 2, target: 4, flow: 290, congestion: 0.4 },
    { id: 4, source: 3, target: 5, flow: 320, congestion: 0.35 },
    { id: 5, source: 4, target: 5, flow: 410, congestion: 0.45 }
  ];

  const amenities = [
    { id: 'h1', lat: 40.7140, lng: -74.0030, name: 'City Hospital', type: 'hospital' },
    { id: 's1', lat: 40.7160, lng: -74.0070, name: 'Central School', type: 'school' },
    { id: 'p1', lat: 40.7110, lng: -74.0120, name: 'Central Park', type: 'park' },
    { id: 'r1', lat: 40.7090, lng: -74.0050, name: 'Downtown Diner', type: 'restaurant' }
  ];

  const metros = [
    { id: 'm1', lat: 40.7128, lng: -74.0060, name: 'Times Square', line: 'Red', stations: 15 },
    { id: 'm2', lat: 40.7180, lng: -74.0040, name: 'Grand Central', line: 'Blue', stations: 20 }
  ];

  return { nodes, edges, amenities, metros };
}
