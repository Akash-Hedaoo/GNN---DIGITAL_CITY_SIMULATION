// GNN City Simulator - Interactive App
const API_BASE = 'http://localhost:5000';

// State Management
const state = {
  map: null,
  currentScenario: null,
  scenarioResults: null,
  selectedNode: null,
  layers: {
    roads: null,
    metros: null,
    amenities: null,
    heatmap: null
  },
  cityData: null,
  predictionsBefore: [],
  predictionsAfter: []
};

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
  console.log('Initializing GNN City Simulator...');
  
  // Initialize map
  initializeMap();
  
  // Load city data
  await loadCityData();
  
  // Setup event listeners
  setupEventListeners();
  
  // Check backend health
  checkBackendHealth();
});

/**
 * Initialize Leaflet Map
 */
function initializeMap() {
  state.map = L.map('map', {
    center: [40.7128, -74.0060], // NYC coordinates
    zoom: 12,
    zoomControl: true,
    attributionControl: true,
    darkMode: true
  });

  // Add tile layer (dark theme)
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; CartoDB',
    maxZoom: 19,
    minZoom: 10
  }).addTo(state.map);

  // Initialize feature layers
  state.layers.roads = L.featureGroup().addTo(state.map);
  state.layers.metros = L.featureGroup().addTo(state.map);
  state.layers.amenities = L.featureGroup().addTo(state.map);

  console.log('Map initialized successfully');
}

/**
 * Load city data from backend
 */
async function loadCityData() {
  try {
    const response = await fetch(`${API_BASE}/city-data`);
    if (response.ok) {
      state.cityData = await response.json();
      renderCityLayers();
      console.log('City data loaded:', state.cityData);
    } else {
      console.warn('City data endpoint not available, using mock data');
      state.cityData = generateMockCityData();
      renderCityLayers();
    }
  } catch (error) {
    console.warn('Could not load city data from backend:', error);
    state.cityData = generateMockCityData();
    renderCityLayers();
  }
}

/**
 * Render city layers on map
 */
function renderCityLayers() {
  if (!state.cityData) return;

  // Clear existing layers
  state.layers.roads.clearLayers();
  state.layers.metros.clearLayers();
  state.layers.amenities.clearLayers();

  const { nodes, edges, amenities } = state.cityData;

  // Render roads
  if (edges && Array.isArray(edges)) {
    edges.forEach((edge, idx) => {
      const source = nodes.find(n => n.id === edge.source);
      const target = nodes.find(n => n.id === edge.target);
      
      if (source && target) {
        const congestion = edge.congestion || 0;
        const color = getCongestionColor(congestion);
        const weight = 2 + (congestion / 10);
        
        const polyline = L.polyline(
          [[source.lat, source.lng], [target.lat, target.lng]],
          {
            color: color,
            weight: weight,
            opacity: 0.7,
            className: `road-segment road-${edge.id}`,
            interactive: true
          }
        );
        
        polyline.bindPopup(`
          <div class="popup-content">
            <strong>Road Segment ${edge.id}</strong><br>
            Congestion: <span style="color:${color}">${(congestion * 100).toFixed(1)}%</span><br>
            Flow: ${edge.flow || 'N/A'} vehicles/hour
          </div>
        `);
        
        polyline.on('click', () => selectNode(edge.id, 'road'));
        state.layers.roads.addLayer(polyline);
      }
    });
  }

  // Render metros (if available)
  if (state.cityData.metros && Array.isArray(state.cityData.metros)) {
    state.cityData.metros.forEach(metro => {
      const icon = L.icon({
        iconUrl: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgZmlsbD0iIzFGMjEzNiIgc3Ryb2tlPSIjRkZDMTA3IiBzdHJva2Utd2lkdGg9IjIiLz48dGV4dCB4PSIxMiIgeT0iMTYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNGRkMxMDciIGZvbnQtc2l6ZT0iOCIgZm9udC13ZWlnaHQ9ImJvbGQiPk08L3RleHQ+PC9zdmc+',
        iconSize: [24, 24],
        iconAnchor: [12, 12],
        popupAnchor: [0, -12]
      });
      
      const marker = L.marker([metro.lat, metro.lng], { icon });
      marker.bindPopup(`
        <div class="popup-content">
          <strong>${metro.name}</strong><br>
          Line: ${metro.line}<br>
          Stations: ${metro.stations || 'N/A'}
        </div>
      `);
      marker.on('click', () => selectNode(metro.id, 'metro'));
      state.layers.metros.addLayer(marker);
    });
  }

  // Render amenities
  if (amenities && Array.isArray(amenities)) {
    amenities.forEach(amenity => {
      const icon = L.icon({
        iconUrl: getAmenityIcon(amenity.type),
        iconSize: [20, 20],
        iconAnchor: [10, 10],
        popupAnchor: [0, -10]
      });
      
      const marker = L.marker([amenity.lat, amenity.lng], { icon });
      marker.bindPopup(`
        <div class="popup-content">
          <strong>${amenity.name}</strong><br>
          Type: ${amenity.type}
        </div>
      `);
      marker.on('click', () => selectNode(amenity.id, 'amenity'));
      state.layers.amenities.addLayer(marker);
    });
  }

  console.log('City layers rendered');
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
        <small class="text-muted">Click on map to select different locations</small>
      </div>
    `;
  } else {
    infoPanel.innerHTML = '<small class="text-muted">Select a node on the map to view details</small>';
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
 * Run scenario
 */
async function runScenario() {
  const scenarioType = document.getElementById('scenarioType').value;
  const severity = parseFloat(document.getElementById('severity').value);
  const duration = parseInt(document.getElementById('duration').value);

  console.log('Running scenario:', { scenarioType, severity, duration });

  // Collect prediction features from map
  const features = collectMapFeatures();
  
  try {
    // Get baseline prediction
    const beforeResponse = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });

    if (!beforeResponse.ok) throw new Error('Prediction failed');
    
    state.predictionsBefore = await beforeResponse.json();

    // Run what-if scenario
    const whatifResponse = await fetch(`${API_BASE}/whatif`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        features,
        scenario: {
          type: scenarioType,
          severity: severity,
          duration: duration
        }
      })
    });

    if (!whatifResponse.ok) throw new Error('What-if analysis failed');

    const result = await whatifResponse.json();
    state.scenarioResults = result;
    state.predictionsAfter = result.after || [];

    // Update map visualization
    updateMapVisualization();
    
    // Show quick stats
    updateQuickStats(result);
    
    console.log('Scenario completed:', result);
  } catch (error) {
    console.error('Error running scenario:', error);
    alert('Error running scenario: ' + error.message);
  }
}

/**
 * Collect features from current map state
 */
function collectMapFeatures() {
  // Collect congestion values from all road segments
  const features = [];
  
  if (state.cityData && state.cityData.edges) {
    state.cityData.edges.forEach(edge => {
      features.push(edge.congestion || Math.random() * 0.5);
    });
  }
  
  // Ensure we have at least some features
  if (features.length === 0) {
    features.push(...[0.3, 0.4, 0.2, 0.5, 0.1]);
  }
  
  return features;
}

/**
 * Update map visualization after scenario
 */
function updateMapVisualization() {
  if (!state.cityData || !state.scenarioResults) return;

  const { edges } = state.cityData;
  const predictions = state.predictionsAfter;

  // Update each road segment with new congestion value
  edges.forEach((edge, idx) => {
    const newCongestion = predictions[idx] !== undefined ? predictions[idx] : edge.congestion;
    const color = getCongestionColor(newCongestion);
    
    // Update road segment visual
    const roadSegment = document.querySelector(`.road-${edge.id}`);
    if (roadSegment) {
      roadSegment.setAttribute('stroke', color);
    }
  });

  console.log('Map visualization updated');
}

/**
 * Update quick stats
 */
function updateQuickStats(result) {
  const statsBefore = calculateStats(state.predictionsBefore);
  const statsAfter = calculateStats(state.predictionsAfter);

  const html = `
    <div class="stat-item">
      <span class="stat-label">Avg Congestion Before</span>
      <span class="stat-value">${(statsBefore.average * 100).toFixed(1)}%</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Avg Congestion After</span>
      <span class="stat-value">${(statsAfter.average * 100).toFixed(1)}%</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Congestion Change</span>
      <span class="stat-value" style="color: ${statsAfter.average > statsBefore.average ? '#FF5722' : '#4CAF50'}">
        ${((statsAfter.average - statsBefore.average) * 100).toFixed(1)}%
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
