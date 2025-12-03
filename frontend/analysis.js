/**
 * Traffic Analysis Page
 * Shows detailed analysis of road closures impact
 */

const API_BASE = 'http://localhost:5000/api';

const state = {
    closedRoads: [],
    baseline: null,
    withClosures: null,
    pieChart: null,
    theme: 'dark'
};

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Load theme from localStorage (synced with index.html)
    loadTheme();
    
    // Setup theme toggle
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
    
    // Load closed roads from localStorage
    loadFromMapPage();
    
    // Setup button
    document.getElementById('btn-analyze').addEventListener('click', runAnalysis);
    
    // Auto-run if we have closed roads
    if (state.closedRoads.length > 0) {
        runAnalysis();
    }
}

// ============================================
// THEME MANAGEMENT
// ============================================

function loadTheme() {
    const savedTheme = localStorage.getItem('citySimTheme') || 'dark';
    state.theme = savedTheme;
    document.documentElement.setAttribute('data-theme', savedTheme);
}

function toggleTheme() {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', state.theme);
    localStorage.setItem('citySimTheme', state.theme);
    
    // Update charts if they exist (for proper colors)
    if (state.pieChart) {
        updateChartColors();
    }
    
    showToast(`Switched to ${state.theme} theme`, 'info');
}

function updateChartColors() {
    // Update chart text colors based on theme
    const textColor = state.theme === 'light' ? '#2c3e50' : '#ffffff';
    
    if (state.pieChart) {
        state.pieChart.options.plugins.legend.labels.color = textColor;
        state.pieChart.update();
    }
}

function loadFromMapPage() {
    const saved = localStorage.getItem('cityAnalysisData');
    if (saved) {
        try {
            const data = JSON.parse(saved);
            state.closedRoads = data.closedRoads || [];
            displayBlockedRoads();
        } catch (e) {
            console.error('Failed to parse closed roads:', e);
        }
    }
}

function displayBlockedRoads() {
    const container = document.getElementById('blocked-roads-display');
    const badge = document.getElementById('blocked-count');
    
    badge.textContent = state.closedRoads.length;
    
    if (state.closedRoads.length === 0) {
        container.innerHTML = '<p class="empty-msg">No roads blocked. Go to map first.</p>';
        return;
    }
    
    container.innerHTML = state.closedRoads.map(road => 
        `<span class="blocked-road-tag">${road}</span>`
    ).join('');
}

// ============================================
// ANALYSIS
// ============================================

async function runAnalysis() {
    if (state.closedRoads.length === 0) {
        showToast('No roads blocked. Go to map and block some roads first.', 'error');
        return;
    }
    
    showLoading('Fetching baseline traffic...');
    
    try {
        // Get baseline (no closures)
        const baselineRes = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ closed_roads: [] })
        });
        state.baseline = await baselineRes.json();
        
        showLoading('Predicting traffic with closures...');
        
        // Get prediction with closures
        const closureRes = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ closed_roads: state.closedRoads })
        });
        state.withClosures = await closureRes.json();
        
        showLoading('Analyzing results...');
        
        // Analyze and display
        analyzeResults();
        
        showToast('Analysis complete!', 'success');
        
    } catch (error) {
        console.error('Analysis failed:', error);
        showToast('Analysis failed: ' + error.message, 'error');
    }
    
    hideLoading();
}

function analyzeResults() {
    const before = state.baseline;
    const after = state.withClosures;
    
    // Build maps
    const beforeMap = {};
    const afterMap = {};
    
    before.predictions.forEach(p => {
        beforeMap[`${p.source}-${p.target}`] = p.congestion;
    });
    
    after.predictions.forEach(p => {
        afterMap[`${p.source}-${p.target}`] = p.congestion;
    });
    
    // Calculate percentiles for congestion levels
    const congestionValues = after.predictions.map(p => p.congestion);
    const sorted = [...congestionValues].sort((a, b) => a - b);
    const p20 = sorted[Math.floor(sorted.length * 0.20)];
    const p40 = sorted[Math.floor(sorted.length * 0.40)];
    const p60 = sorted[Math.floor(sorted.length * 0.60)];
    const p80 = sorted[Math.floor(sorted.length * 0.80)];
    
    // Count by congestion level
    let veryHigh = 0, high = 0, medium = 0, low = 0, veryLow = 0;
    
    // Calculate changes and categorize
    const allRoads = [];
    let increased = 0, decreased = 0, unchanged = 0;
    
    for (const key in afterMap) {
        const congestion = afterMap[key];
        const baseline = beforeMap[key];
        const diff = congestion - baseline;
        
        // Get congestion level
        let level;
        if (congestion >= p80) { level = 'very-high'; veryHigh++; }
        else if (congestion >= p60) { level = 'high'; high++; }
        else if (congestion >= p40) { level = 'medium'; medium++; }
        else if (congestion >= p20) { level = 'low'; low++; }
        else { level = 'very-low'; veryLow++; }
        
        allRoads.push({ road: key, congestion, baseline, diff, level });
        
        if (diff > 0.01) increased++;
        else if (diff < -0.01) decreased++;
        else unchanged++;
    }
    
    // Sort by congestion (highest first)
    allRoads.sort((a, b) => b.congestion - a.congestion);
    
    // Update Impact Summary
    document.getElementById('increased-count').textContent = increased;
    document.getElementById('decreased-count').textContent = decreased;
    document.getElementById('unchanged-count').textContent = unchanged;
    
    // Update Congestion Distribution
    updateCongestionChart(veryHigh, high, medium, low, veryLow);
    document.getElementById('level-very-high').textContent = veryHigh;
    document.getElementById('level-high').textContent = high;
    document.getElementById('level-medium').textContent = medium;
    document.getElementById('level-low').textContent = low;
    document.getElementById('level-very-low').textContent = veryLow;
    
    // Update Before/After Comparison
    updateComparison(before.stats, after.stats);
    
    // Update Affected Roads Lists
    updateAffectedRoads(allRoads);
}

function updateCongestionChart(veryHigh, high, medium, low, veryLow) {
    const ctx = document.getElementById('congestion-pie').getContext('2d');
    
    if (state.pieChart) {
        state.pieChart.destroy();
    }
    
    // Get text color based on theme
    const textColor = state.theme === 'light' ? '#2c3e50' : '#ffffff';
    
    state.pieChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Very High', 'High', 'Medium', 'Low', 'Very Low'],
            datasets: [{
                data: [veryHigh, high, medium, low, veryLow],
                backgroundColor: ['#c0392b', '#e67e22', '#f1c40f', '#2ecc71', '#1e8449'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { 
                    display: false 
                },
                tooltip: {
                    bodyColor: textColor,
                    titleColor: textColor
                }
            },
            cutout: '60%'
        }
    });
}

function updateComparison(before, after) {
    // Average congestion
    const avgBefore = before.mean_congestion;
    const avgAfter = after.mean_congestion;
    const avgDiff = avgAfter - avgBefore;
    
    document.getElementById('avg-before').textContent = (avgBefore * 100).toFixed(1) + '%';
    document.getElementById('avg-after').textContent = (avgAfter * 100).toFixed(1) + '%';
    updateChangeElement('avg-change', avgDiff);
    
    // Max congestion
    const maxBefore = before.max_congestion;
    const maxAfter = after.max_congestion;
    const maxDiff = maxAfter - maxBefore;
    
    document.getElementById('max-before').textContent = (maxBefore * 100).toFixed(1) + '%';
    document.getElementById('max-after').textContent = (maxAfter * 100).toFixed(1) + '%';
    updateChangeElement('max-change', maxDiff);
    
    // Total network load (sum approximation using mean * count)
    const totalBefore = before.road_mean || before.mean_congestion;
    const totalAfter = after.road_mean || after.mean_congestion;
    const totalDiff = totalAfter - totalBefore;
    
    document.getElementById('total-before').textContent = (totalBefore * 100).toFixed(1) + '%';
    document.getElementById('total-after').textContent = (totalAfter * 100).toFixed(1) + '%';
    updateChangeElement('total-change', totalDiff);
}

function updateChangeElement(id, diff) {
    const el = document.getElementById(id);
    const pct = (diff * 100).toFixed(2);
    
    if (diff > 0.001) {
        el.textContent = `+${pct}%`;
        el.className = 'stat-change positive';
    } else if (diff < -0.001) {
        el.textContent = `${pct}%`;
        el.className = 'stat-change negative';
    } else {
        el.textContent = 'No change';
        el.className = 'stat-change neutral';
    }
}

function updateAffectedRoads(roads) {
    // Most congested (top 10)
    const topCongested = roads.slice(0, 10);
    const increasedList = document.getElementById('increased-roads');
    
    increasedList.innerHTML = topCongested.map(r => `
        <div class="road-item">
            <span class="road-id">${r.road}</span>
            <span class="road-congestion ${r.level}">${(r.congestion * 100).toFixed(1)}%</span>
        </div>
    `).join('');
    
    // Least congested (bottom 10)
    const bottomCongested = roads.slice(-10).reverse();
    const decreasedList = document.getElementById('decreased-roads');
    
    decreasedList.innerHTML = bottomCongested.map(r => `
        <div class="road-item">
            <span class="road-id">${r.road}</span>
            <span class="road-congestion ${r.level}">${(r.congestion * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}

// ============================================
// UTILITIES
// ============================================

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
    
    setTimeout(() => toast.remove(), 4000);
}
