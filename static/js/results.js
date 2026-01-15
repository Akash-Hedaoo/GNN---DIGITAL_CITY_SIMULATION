// Road name cache to avoid repeated API calls
const roadNameCache = new Map();

// Load and display simulation results
document.addEventListener('DOMContentLoaded', function() {
    const resultsData = sessionStorage.getItem('simulationResults');
    const selectedEdges = sessionStorage.getItem('selectedEdges');
    const graphData = sessionStorage.getItem('graphData');
    
    if (!resultsData) {
        alert('No simulation results found. Please run a simulation first.');
        window.location.href = '/';
        return;
    }
    
    const results = JSON.parse(resultsData);
    const closedEdges = JSON.parse(selectedEdges || '[]');
    
    // Store graph data if available
    if (graphData) {
        window.graphData = JSON.parse(graphData);
    } else {
        // Load graph data for road name lookup
        loadGraphData();
    }
    
    // Display metrics
    displayMetrics(results.metrics);
    
    // Display top 5 critical edges with road names
    if (results.top_5_critical_edges && results.top_5_critical_edges.length > 0) {
        displayTop5CriticalEdges(results.top_5_critical_edges);
    }
    
    // Display impacted edges
    displayImpactedEdges(results.impacted_edges);
    
    // Draw visualization chart
    drawChart(results.impacted_edges);
});

// Load graph data for coordinate lookup
async function loadGraphData() {
    try {
        const response = await fetch('/api/graph-data');
        const data = await response.json();
        window.graphData = data;
        sessionStorage.setItem('graphData', JSON.stringify(data));
    } catch (error) {
        console.error('Error loading graph data:', error);
    }
}

// Get road name from coordinates
async function getRoadName(lat, lon, edgeId) {
    // Check cache first
    const cacheKey = `${lat.toFixed(4)}_${lon.toFixed(4)}`;
    if (roadNameCache.has(cacheKey)) {
        return roadNameCache.get(cacheKey);
    }
    
    try {
        const response = await fetch('/api/get-road-name', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ lat, lon })
        });
        
        const data = await response.json();
        const roadName = data.road_name || `Road Segment ${edgeId.split('_')[0]}`;
        
        // Cache the result
        roadNameCache.set(cacheKey, roadName);
        
        // Rate limiting - be respectful to Nominatim API
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        return roadName;
    } catch (error) {
        console.error('Error fetching road name:', error);
        return `Road Segment ${edgeId.split('_')[0]}`;
    }
}

// Get coordinates for an edge
function getEdgeCoordinates(edgeId) {
    if (!window.graphData) return null;
    
    const parts = edgeId.split('_');
    if (parts.length < 2) return null;
    
    const sourceId = parts[0];
    const targetId = parts[1];
    
    // Find source node
    const sourceNode = window.graphData.nodes.find(n => n.id === sourceId);
    const targetNode = window.graphData.nodes.find(n => n.id === targetId);
    
    if (sourceNode && targetNode) {
        // Use midpoint for road name lookup
        return {
            lat: (sourceNode.lat + targetNode.lat) / 2,
            lon: (sourceNode.lon + targetNode.lon) / 2
        };
    }
    
    return null;
}

// Display metrics with simplified explanations
function displayMetrics(metrics) {
    if (!metrics) {
        console.error('Metrics data is missing');
        return;
    }
    
    const netChangeEl = document.getElementById('netChange');
    const impactedSegmentsEl = document.getElementById('impactedSegments');
    const peakBottleneckEl = document.getElementById('peakBottleneck');
    const avgCongestionEl = document.getElementById('avgCongestion');
    
    if (netChangeEl) {
        const change = metrics.net_traffic_change_pct || 0;
        netChangeEl.textContent = Math.abs(change).toFixed(1) + '%';
        
        // Color code net change
        if (change > 5) {
            netChangeEl.style.color = '#e74c3c'; // Red for significant increase
            netChangeEl.parentElement.querySelector('.metric-description').textContent = 'Traffic increased significantly';
        } else if (change > 0) {
            netChangeEl.style.color = '#f39c12'; // Orange for moderate increase
            netChangeEl.parentElement.querySelector('.metric-description').textContent = 'Traffic increased slightly';
        } else if (change < -5) {
            netChangeEl.style.color = '#27ae60'; // Green for significant decrease
            netChangeEl.parentElement.querySelector('.metric-description').textContent = 'Traffic decreased significantly';
        } else if (change < 0) {
            netChangeEl.style.color = '#2ecc71'; // Light green for moderate decrease
            netChangeEl.parentElement.querySelector('.metric-description').textContent = 'Traffic decreased slightly';
        } else {
            netChangeEl.style.color = '#7f8c8d'; // Gray for no change
            netChangeEl.parentElement.querySelector('.metric-description').textContent = 'No significant change';
        }
    }
    
    if (impactedSegmentsEl) {
        impactedSegmentsEl.textContent = metrics.impacted_segments || 0;
    }
    
    if (peakBottleneckEl) {
        const peak = metrics.peak_bottleneck_spike || 0;
        peakBottleneckEl.textContent = peak.toFixed(1) + 'x';
        
        // Add description
        const description = peakBottleneckEl.parentElement.querySelector('.metric-description');
        if (description) {
            if (peak > 5) {
                description.textContent = 'Severe bottleneck detected';
            } else if (peak > 3) {
                description.textContent = 'Major bottleneck detected';
            } else {
                description.textContent = 'Moderate bottleneck';
            }
        }
    }
    
    if (avgCongestionEl) {
        const avg = metrics.avg_congestion || 0;
        avgCongestionEl.textContent = avg.toFixed(1) + 'x';
        
        // Add description
        const description = avgCongestionEl.parentElement.querySelector('.metric-description');
        if (description) {
            if (avg > 3) {
                description.textContent = 'High congestion overall';
            } else if (avg > 2) {
                description.textContent = 'Moderate congestion';
            } else {
                description.textContent = 'Low congestion';
            }
        }
    }
}

// Display top 5 critical edges with road names
async function displayTop5CriticalEdges(criticalEdges) {
    const container = document.getElementById('top5EdgesList');
    container.innerHTML = '<div class="loading-road-names">Loading road names...</div>';
    
    if (criticalEdges.length === 0) {
        container.innerHTML = '<p style="padding: 20px; text-align: center; color: #7f8c8d;">No critical bottlenecks found.</p>';
        return;
    }
    
    // Process edges and fetch road names
    const edgesWithNames = await Promise.all(
        criticalEdges.map(async (edge, index) => {
            const coords = getEdgeCoordinates(edge.edge_id);
            let roadName = `Road Segment ${index + 1}`;
            
            if (coords) {
                roadName = await getRoadName(coords.lat, coords.lon, edge.edge_id);
            }
            
            return { ...edge, roadName };
        })
    );
    
    container.innerHTML = '';
    
    edgesWithNames.forEach((edge, index) => {
        const rankNum = index + 1;
        const card = document.createElement('div');
        card.className = 'critical-edge-card';
        
        // Calculate percentage change
        const baselineValue = edge.baseline_congestion || edge.congestion;
        const currentValue = edge.current_congestion || edge.congestion;
        const percentageChange = edge.pct_change || 0;
        
        // Determine severity
        const isIncrease = percentageChange > 0;
        const changeSymbol = isIncrease ? '+' : '';
        let severity = 'Moderate';
        let severityColor = '#f39c12';
        
        if (Math.abs(percentageChange) > 50) {
            severity = 'Critical';
            severityColor = '#e74c3c';
        } else if (Math.abs(percentageChange) > 25) {
            severity = 'High';
            severityColor = '#e67e22';
        } else if (Math.abs(percentageChange) > 10) {
            severity = 'Moderate';
            severityColor = '#f39c12';
        } else {
            severity = 'Low';
            severityColor = '#95a5a6';
        }
        
        // Calculate travel time impact
        const timeIncrease = ((currentValue - baselineValue) / baselineValue) * 100;
        const timeIncreaseText = timeIncrease > 0 ? `+${timeIncrease.toFixed(0)}% longer` : `${timeIncrease.toFixed(0)}% shorter`;
        
        card.innerHTML = `
            <div class="critical-edge-header">
                <div class="critical-edge-rank rank-${rankNum}">#${rankNum}</div>
                <div class="severity-badge" style="background: ${severityColor}">${severity} Impact</div>
            </div>
            <div class="critical-edge-road-name">
                <i class="road-icon"></i>
                <strong>${edge.roadName}</strong>
            </div>
            <div class="critical-edge-congestion">
                <div class="congestion-metric baseline">
                    <div class="congestion-label">Normal Traffic</div>
                    <div class="congestion-value">${baselineValue.toFixed(1)}x</div>
                    <div class="congestion-subtext">Baseline congestion</div>
                </div>
                <div class="congestion-arrow">→</div>
                <div class="congestion-metric current">
                    <div class="congestion-label">After Closure</div>
                    <div class="congestion-value" style="color: ${severityColor}">${currentValue.toFixed(1)}x</div>
                    <div class="congestion-subtext">New congestion level</div>
                </div>
            </div>
            <div class="critical-edge-impact">
                <div class="impact-stat">
                    <div class="impact-label">Travel Time</div>
                    <div class="impact-value" style="color: ${severityColor}">${timeIncreaseText}</div>
                </div>
                <div class="impact-stat">
                    <div class="impact-label">Change</div>
                    <div class="impact-value" style="color: ${severityColor}">${changeSymbol}${percentageChange.toFixed(1)}%</div>
                </div>
            </div>
        `;
        
        container.appendChild(card);
    });
}

// Display impacted edges list with road names
async function displayImpactedEdges(impactedEdges) {
    const container = document.getElementById('impactList');
    container.innerHTML = '<div class="loading-road-names">Loading road names...</div>';
    
    if (impactedEdges.length === 0) {
        container.innerHTML = '<p style="padding: 20px; text-align: center; color: #7f8c8d;">No significantly impacted segments found.</p>';
        return;
    }
    
    // Sort by impact (descending)
    const sorted = impactedEdges.sort((a, b) => Math.abs(b.pct_change) - Math.abs(a.pct_change));
    
    // Process first 20 edges with road names (to avoid too many API calls)
    const edgesToProcess = sorted.slice(0, 20);
    const remainingEdges = sorted.slice(20);
    
    const edgesWithNames = await Promise.all(
        edgesToProcess.map(async (edge) => {
            const coords = getEdgeCoordinates(edge.edge_id);
            let roadName = edge.edge_id;
            
            if (coords) {
                roadName = await getRoadName(coords.lat, coords.lon, edge.edge_id);
            }
            
            return { ...edge, roadName };
        })
    );
    
    container.innerHTML = '';
    
    // Display edges with names
    edgesWithNames.forEach(edge => {
        const item = document.createElement('div');
        item.className = 'impact-item';
        
        const isIncrease = edge.pct_change > 0;
        const changeColor = isIncrease ? '#e74c3c' : '#27ae60';
        const changeSymbol = isIncrease ? '+' : '';
        
        // Determine impact level
        const impactLevel = Math.abs(edge.pct_change) > 25 ? 'high' : Math.abs(edge.pct_change) > 10 ? 'moderate' : 'low';
        
        item.innerHTML = `
            <div class="impact-item-info">
                <div class="impact-item-road-name">
                    <i class="road-icon-small"></i>
                    <strong>${edge.roadName}</strong>
                </div>
                <div class="impact-item-metrics">
                    <span class="metric-tag">
                        <span class="metric-label">Congestion:</span>
                        <strong>${edge.congestion.toFixed(1)}x</strong>
                    </span>
                    <span class="metric-tag impact-${impactLevel}">
                        <span class="metric-label">Change:</span>
                        <strong style="color: ${changeColor}">${changeSymbol}${edge.pct_change.toFixed(1)}%</strong>
                    </span>
                </div>
            </div>
            <div class="impact-item-value" style="color: ${changeColor}">
                ${changeSymbol}${edge.change.toFixed(2)}x
            </div>
        `;
        
        container.appendChild(item);
    });
    
    // Show remaining edges count if any
    if (remainingEdges.length > 0) {
        const moreItem = document.createElement('div');
        moreItem.className = 'impact-item-more';
        moreItem.innerHTML = `
            <p>... and ${remainingEdges.length} more impacted road segments</p>
        `;
        container.appendChild(moreItem);
    }
}

// Draw visualization chart
function drawChart(impactedEdges) {
    const canvas = document.getElementById('impactChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (impactedEdges.length === 0) {
        ctx.fillStyle = '#7f8c8d';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No data to display', canvas.width / 2, canvas.height / 2);
        return;
    }
    
    // Sort by impact
    const sorted = impactedEdges.slice(0, 20).sort((a, b) => 
        Math.abs(b.pct_change) - Math.abs(a.pct_change)
    );
    
    const padding = 60;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;
    const barWidth = chartWidth / sorted.length;
    const maxChange = Math.max(...sorted.map(e => Math.abs(e.pct_change)));
    
    // Draw bars
    sorted.forEach((edge, index) => {
        const x = padding + index * barWidth;
        const barHeight = (Math.abs(edge.pct_change) / maxChange) * chartHeight;
        const y = padding + chartHeight - barHeight;
        
        const isIncrease = edge.pct_change > 0;
        ctx.fillStyle = isIncrease ? '#e74c3c' : '#27ae60';
        
        ctx.fillRect(x + 5, y, barWidth - 10, barHeight);
        
        // Draw value label
        ctx.fillStyle = '#2c3e50';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.save();
        ctx.translate(x + barWidth / 2, y - 5);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(edge.pct_change.toFixed(1) + '%', 0, 0);
        ctx.restore();
    });
    
    // Draw axes
    ctx.strokeStyle = '#2c3e50';
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding, padding + chartHeight);
    ctx.lineTo(padding + chartWidth, padding + chartHeight);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, padding + chartHeight);
    ctx.stroke();
    
    // Y-axis labels
    ctx.fillStyle = '#2c3e50';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
        const value = (maxChange / 5) * i;
        const y = padding + chartHeight - (i / 5) * chartHeight;
        ctx.fillText(value.toFixed(1) + '%', padding - 10, y + 4);
    }
    
    // Title
    ctx.fillStyle = '#2c3e50';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Top Impacted Road Segments', canvas.width / 2, 30);
}
