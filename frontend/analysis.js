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
    
    // Auto-run only if we need to fetch data
    if (state.closedRoads.length > 0 && !state.baseline) {
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
    console.log('📦 Loading analysis data from localStorage:', saved ? 'Found' : 'Not found');
    
    if (saved) {
        try {
            const data = JSON.parse(saved);
            console.log('📊 Parsed analysis data:', {
                closedRoads: data.closedRoads?.length,
                hasBaseline: !!data.baseline,
                hasWithClosures: !!data.withClosures,
                baselinePredictions: data.baseline?.predictions?.length,
                withClosuresPredictions: data.withClosures?.predictions?.length
            });
            
            state.closedRoads = data.closedRoads || [];
            state.baseline = data.baseline;
            state.withClosures = data.withClosures;
            displayBlockedRoads();
            
            // If we have both baseline and withClosures, auto-analyze
            if (state.baseline && state.withClosures) {
                console.log('📊 Data loaded from map page, running analysis...');
                setTimeout(() => analyzeResults(), 100);
            } else {
                console.warn('⚠️ Missing baseline or withClosures data. Please run prediction from map first.');
                showToast('No prediction data found. Please run prediction from the map page first.', 'error');
            }
        } catch (e) {
            console.error('Failed to parse analysis data:', e);
            showToast('Failed to load analysis data', 'error');
        }
    } else {
        console.warn('⚠️ No analysis data in localStorage');
        showToast('No analysis data found. Please block roads and run prediction from the map page.', 'info');
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
    console.log('🔍 Starting analysis...');
    const before = state.baseline;
    const after = state.withClosures;
    
    if (!before || !after) {
        console.error('❌ Missing data:', { before, after });
        showToast('Missing prediction data. Please run prediction from map page first.', 'error');
        return;
    }
    
    if (!before.predictions || !after.predictions) {
        console.error('❌ Missing predictions:', { before, after });
        showToast('Invalid prediction data structure', 'error');
        return;
    }
    
    console.log(`📊 Analyzing ${before.predictions.length} baseline vs ${after.predictions.length} closure predictions`);
    console.log('Before stats:', before.stats);
    console.log('After stats:', after.stats);
    
    // Build maps
    const beforeMap = {};
    const afterMap = {};
    
    before.predictions.forEach(p => {
        beforeMap[`${p.source}-${p.target}`] = p.congestion;
    });
    
    after.predictions.forEach(p => {
        afterMap[`${p.source}-${p.target}`] = p.congestion;
    });
    
    // Use absolute thresholds for congestion levels
    // Values are typically 0-10 (raw) or 0-1 (normalized)
    const normalize = (val) => val > 1 ? val / 10 : val; // Convert to 0-1 if needed
    
    // Count by congestion level using absolute thresholds
    let veryHigh = 0, high = 0, medium = 0, low = 0, veryLow = 0;
    
    // Calculate changes and categorize
    const allRoads = [];
    let increased = 0, decreased = 0, unchanged = 0;
    
    for (const key in afterMap) {
        const congestion = afterMap[key];
        const baseline = beforeMap[key] || 0;
        const diff = congestion - baseline;
        
        // Normalize for categorization (0-1 scale)
        const normalizedCongestion = normalize(congestion);
        
        // Get congestion level using absolute thresholds
        let level;
        if (normalizedCongestion >= 0.8) { level = 'very-high'; veryHigh++; }
        else if (normalizedCongestion >= 0.6) { level = 'high'; high++; }
        else if (normalizedCongestion >= 0.4) { level = 'medium'; medium++; }
        else if (normalizedCongestion >= 0.2) { level = 'low'; low++; }
        else { level = 'very-low'; veryLow++; }
        
        allRoads.push({ road: key, congestion, baseline, diff, level });
        
        // Use percentage-based change detection for more accurate comparison
        // If the difference is less than 1% of the baseline, consider it unchanged
        const percentChange = baseline > 0 ? Math.abs(diff / baseline) * 100 : (diff === 0 ? 0 : 100);
        
        if (percentChange < 1) {
            // Less than 1% change = unchanged
            unchanged++;
        } else if (diff > 0) {
            increased++;
        } else {
            decreased++;
        }
    }
    
    console.log(`📈 Impact: ${increased} increased, ${decreased} decreased, ${unchanged} unchanged`);
    console.log(`📊 Distribution: VH=${veryHigh}, H=${high}, M=${medium}, L=${low}, VL=${veryLow}`);
    
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
    
    console.log('✅ Analysis complete!');
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

function formatCongestionValue(value) {
    // Detect if value is raw congestion (typically 0-10) or normalized ratio (0-1)
    // If raw (>1), treat as percentage directly
    // If normalized (<=1), multiply by 100
    if (value > 1) {
        return value.toFixed(1) + '%';
    } else {
        return (value * 100).toFixed(1) + '%';
    }
}

function updateComparison(before, after) {
    console.log('📊 Comparison data:', { before, after });
    
    // Average congestion
    const avgBefore = before.mean_congestion;
    const avgAfter = after.mean_congestion;
    const avgDiff = avgAfter - avgBefore;
    
    console.log('Avg:', { avgBefore, avgAfter, avgDiff });
    
    document.getElementById('avg-before').textContent = formatCongestionValue(avgBefore);
    document.getElementById('avg-after').textContent = formatCongestionValue(avgAfter);
    updateChangeElement('avg-change', avgDiff, avgBefore);
    
    // Max congestion
    const maxBefore = before.max_congestion;
    const maxAfter = after.max_congestion;
    const maxDiff = maxAfter - maxBefore;
    
    console.log('Max:', { maxBefore, maxAfter, maxDiff });
    
    document.getElementById('max-before').textContent = formatCongestionValue(maxBefore);
    document.getElementById('max-after').textContent = formatCongestionValue(maxAfter);
    updateChangeElement('max-change', maxDiff, maxBefore);
    
    // Total network load (sum approximation using mean * count)
    const totalBefore = before.road_mean || before.mean_congestion;
    const totalAfter = after.road_mean || after.mean_congestion;
    const totalDiff = totalAfter - totalBefore;
    
    console.log('Total:', { totalBefore, totalAfter, totalDiff });
    
    document.getElementById('total-before').textContent = formatCongestionValue(totalBefore);
    document.getElementById('total-after').textContent = formatCongestionValue(totalAfter);
    updateChangeElement('total-change', totalDiff, totalBefore);
}

function updateChangeElement(id, diff, baseValue) {
    const el = document.getElementById(id);
    if (!el) {
        console.warn(`Element ${id} not found`);
        return;
    }
    
    console.log(`Change for ${id}:`, { diff, baseValue });
    
    // Calculate percentage points difference
    // The values are stored as raw (0-10) or normalized (0-1)
    // We need to show the difference in percentage points
    let pctPointsDiff;
    
    if (Math.abs(diff) > 1) {
        // Raw values (0-10 scale), difference is already in percentage points
        pctPointsDiff = diff.toFixed(2);
    } else {
        // Normalized values (0-1 scale), multiply by 100 to get percentage points
        pctPointsDiff = (diff * 100).toFixed(2);
    }
    
    // Threshold for showing change (0.05 percentage points)
    const threshold = Math.abs(diff) > 1 ? 0.05 : 0.0005;
    
    if (diff > threshold) {
        el.textContent = `+${pctPointsDiff} pp`;
        el.className = 'stat-change positive';
    } else if (diff < -threshold) {
        el.textContent = `${pctPointsDiff} pp`;
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
    
    increasedList.innerHTML = topCongested.map(r => {
        const congestionPct = r.congestion > 1 ? r.congestion.toFixed(1) : (r.congestion * 100).toFixed(1);
        return `
        <div class="road-item">
            <span class="road-id">${r.road}</span>
            <span class="road-congestion ${r.level}">${congestionPct}%</span>
        </div>
        `;
    }).join('');
    
    // Least congested (bottom 10)
    const bottomCongested = roads.slice(-10).reverse();
    const decreasedList = document.getElementById('decreased-roads');
    
    decreasedList.innerHTML = bottomCongested.map(r => {
        const congestionPct = r.congestion > 1 ? r.congestion.toFixed(1) : (r.congestion * 100).toFixed(1);
        return `
        <div class="road-item">
            <span class="road-id">${r.road}</span>
            <span class="road-congestion ${r.level}">${congestionPct}%</span>
        </div>
        `;
    }).join('');
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
