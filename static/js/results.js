// Load and display simulation results
document.addEventListener('DOMContentLoaded', function() {
    const resultsData = sessionStorage.getItem('simulationResults');
    const selectedEdges = sessionStorage.getItem('selectedEdges');
    
    if (!resultsData) {
        alert('No simulation results found. Please run a simulation first.');
        window.location.href = '/';
        return;
    }
    
    const results = JSON.parse(resultsData);
    const closedEdges = JSON.parse(selectedEdges || '[]');
    
    // Display metrics
    displayMetrics(results.metrics);
    
    // Display top 5 critical edges
    if (results.top_5_critical_edges && results.top_5_critical_edges.length > 0) {
        displayTop5CriticalEdges(results.top_5_critical_edges);
    }
    
    // Display impacted edges
    displayImpactedEdges(results.impacted_edges);
    
    // Draw visualization chart
    drawChart(results.impacted_edges);
});

// Display metrics
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
        netChangeEl.textContent = change.toFixed(2) + '%';
        // Color code net change
        if (change > 0) {
            netChangeEl.style.color = '#e74c3c'; // Red for increase
        } else if (change < 0) {
            netChangeEl.style.color = '#27ae60'; // Green for decrease
        } else {
            netChangeEl.style.color = '#7f8c8d'; // Gray for no change
        }
    }
    
    if (impactedSegmentsEl) {
        impactedSegmentsEl.textContent = metrics.impacted_segments || 0;
    }
    
    if (peakBottleneckEl) {
        peakBottleneckEl.textContent = (metrics.peak_bottleneck_spike || 0).toFixed(2);
    }
    
    if (avgCongestionEl) {
        avgCongestionEl.textContent = (metrics.avg_congestion || 0).toFixed(2);
    }
}

// Display top 5 critical edges
function displayTop5CriticalEdges(criticalEdges) {
    const container = document.getElementById('top5EdgesList');
    container.innerHTML = '';
    
    if (criticalEdges.length === 0) {
        container.innerHTML = '<p style="padding: 20px; text-align: center; color: #7f8c8d;">No critical edges found.</p>';
        return;
    }
    
    criticalEdges.forEach((edge, index) => {
        const rankNum = index + 1;
        const card = document.createElement('div');
        card.className = 'critical-edge-card';
        
        // Calculate percentage change
        const baselineValue = edge.baseline_congestion || edge.congestion;
        const currentValue = edge.current_congestion || edge.congestion;
        const percentageChange = edge.pct_change || 0;
        
        // Format node references
        const sourceNode = edge.source_node || edge.edge_id.split('_')[0];
        const targetNode = edge.target_node || edge.edge_id.split('_')[1];
        
        // Determine color based on percentage change
        const isIncrease = percentageChange > 0;
        const changeSymbol = isIncrease ? '+' : '';
        
        card.innerHTML = `
            <div class="critical-edge-rank">#${rankNum} Critical Bottleneck</div>
            <div class="critical-edge-nodes">
                <strong>Road Segment:</strong><br>
                Node ${sourceNode} → Node ${targetNode}
            </div>
            <div class="critical-edge-congestion">
                <div class="congestion-metric baseline">
                    <div class="congestion-label">Baseline</div>
                    <div class="congestion-value">${baselineValue.toFixed(2)}x</div>
                </div>
                <div class="congestion-metric current">
                    <div class="congestion-label">Current</div>
                    <div class="congestion-value">${currentValue.toFixed(2)}x</div>
                </div>
            </div>
            <div class="critical-edge-change">
                <div class="change-percentage" style="color: ${isIncrease ? '#e74c3c' : '#27ae60'}">
                    ${changeSymbol}${percentageChange.toFixed(1)}%
                </div>
                <div class="change-label">Congestion Change</div>
                <div class="change-amount">
                    Increase: ${(currentValue - baselineValue).toFixed(2)}x
                </div>
            </div>
        `;
        
        container.appendChild(card);
    });
}

// Display impacted edges list
function displayImpactedEdges(impactedEdges) {
    const container = document.getElementById('impactList');
    container.innerHTML = '';
    
    if (impactedEdges.length === 0) {
        container.innerHTML = '<p style="padding: 20px; text-align: center; color: #7f8c8d;">No significantly impacted segments found.</p>';
        return;
    }
    
    // Sort by impact (descending)
    const sorted = impactedEdges.sort((a, b) => Math.abs(b.pct_change) - Math.abs(a.pct_change));
    
    sorted.forEach(edge => {
        const item = document.createElement('div');
        item.className = 'impact-item';
        
        const isIncrease = edge.pct_change > 0;
        const changeColor = isIncrease ? '#e74c3c' : '#27ae60';
        const changeSymbol = isIncrease ? '+' : '';
        
        item.innerHTML = `
            <div class="impact-item-info">
                <div class="impact-item-id">${edge.edge_id}</div>
                <div class="impact-item-metrics">
                    <span>Congestion: <strong>${edge.congestion.toFixed(2)}</strong></span>
                    <span>Change: <strong style="color: ${changeColor}">${changeSymbol}${edge.pct_change.toFixed(2)}%</strong></span>
                </div>
            </div>
            <div class="impact-item-value" style="color: ${changeColor}">
                ${changeSymbol}${edge.change.toFixed(2)}
            </div>
        `;
        
        container.appendChild(item);
    });
}

// Draw visualization chart
function drawChart(impactedEdges) {
    const canvas = document.getElementById('impactChart');
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

