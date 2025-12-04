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
    
    // Display impacted edges
    displayImpactedEdges(results.impacted_edges);
    
    // Draw visualization chart
    drawChart(results.impacted_edges);
});

// Display metrics
function displayMetrics(metrics) {
    document.getElementById('netChange').textContent = 
        metrics.net_traffic_change_pct.toFixed(2) + '%';
    
    document.getElementById('impactedSegments').textContent = 
        metrics.impacted_segments;
    
    document.getElementById('peakBottleneck').textContent = 
        metrics.peak_bottleneck_spike.toFixed(2);
    
    document.getElementById('avgCongestion').textContent = 
        metrics.avg_congestion.toFixed(2);
    
    // Color code net change
    const netChangeEl = document.getElementById('netChange');
    if (metrics.net_traffic_change_pct > 0) {
        netChangeEl.style.color = '#e74c3c'; // Red for increase
    } else {
        netChangeEl.style.color = '#27ae60'; // Green for decrease
    }
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

