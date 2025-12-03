/**
 * Utility Functions for GNN City Simulator
 */

/**
 * Format a number as percentage
 */
function formatPercent(value) {
  return (value * 100).toFixed(1) + '%';
}

/**
 * Format a number with thousand separator
 */
function formatNumber(value) {
  return value.toLocaleString();
}

/**
 * Calculate distance between two coordinates (Haversine formula)
 */
function calculateDistance(lat1, lng1, lat2, lng2) {
  const R = 6371; // Earth radius in km
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a = 
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLng / 2) * Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

/**
 * Get bounds from array of coordinates
 */
function getCoordinateBounds(coordinates) {
  let minLat = Infinity, maxLat = -Infinity;
  let minLng = Infinity, maxLng = -Infinity;

  coordinates.forEach(coord => {
    minLat = Math.min(minLat, coord.lat);
    maxLat = Math.max(maxLat, coord.lat);
    minLng = Math.min(minLng, coord.lng);
    maxLng = Math.max(maxLng, coord.lng);
  });

  return {
    northEast: [maxLat, maxLng],
    southWest: [minLat, minLng]
  };
}

/**
 * Generate random color
 */
function getRandomColor() {
  const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'];
  return colors[Math.floor(Math.random() * colors.length)];
}

/**
 * Debounce function for event handlers
 */
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

/**
 * Throttle function for event handlers
 */
function throttle(func, limit) {
  let inThrottle;
  return function(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

/**
 * Deep clone object
 */
function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

/**
 * Check if object is empty
 */
function isEmpty(obj) {
  return Object.keys(obj).length === 0;
}

/**
 * Get value from nested object safely
 */
function getNestedValue(obj, path, defaultValue = null) {
  const keys = path.split('.');
  let result = obj;
  
  for (let key of keys) {
    if (result && typeof result === 'object' && key in result) {
      result = result[key];
    } else {
      return defaultValue;
    }
  }
  
  return result;
}

/**
 * Normalize array values to 0-1 range
 */
function normalize(array) {
  const min = Math.min(...array);
  const max = Math.max(...array);
  const range = max - min;
  
  if (range === 0) return array.map(() => 0.5);
  
  return array.map(val => (val - min) / range);
}

/**
 * Calculate weighted average
 */
function weightedAverage(values, weights) {
  const sum = values.reduce((acc, val, idx) => acc + val * weights[idx], 0);
  const weightSum = weights.reduce((acc, w) => acc + w, 0);
  return sum / weightSum;
}

/**
 * Group array by key
 */
function groupBy(array, key) {
  return array.reduce((result, item) => {
    const group = item[key];
    if (!result[group]) result[group] = [];
    result[group].push(item);
    return result;
  }, {});
}

/**
 * Map bounds to zoom level
 */
function boundsToZoom(bounds, mapWidth, mapHeight) {
  const maxZoom = 20;
  
  for (let z = maxZoom; z > 0; z--) {
    const maxPixels = 256 * Math.pow(2, z);
    const widthZoom = Math.log2(maxPixels / mapWidth);
    const heightZoom = Math.log2(maxPixels / mapHeight);
    
    if (Math.min(widthZoom, heightZoom) >= z) {
      return z;
    }
  }
  
  return 0;
}

/**
 * Create heatmap data for Leaflet
 */
function createHeatmapData(edges, predictions) {
  return edges.map((edge, idx) => {
    const intensity = predictions[idx] || 0;
    return {
      lat: (edge.source.lat + edge.target.lat) / 2,
      lng: (edge.source.lng + edge.target.lng) / 2,
      value: intensity
    };
  });
}

/**
 * Get color interpolation between two colors
 */
function interpolateColor(color1, color2, factor) {
  const c1 = parseInt(color1.slice(1), 16);
  const c2 = parseInt(color2.slice(1), 16);

  const r1 = (c1 >> 16) & 255;
  const g1 = (c1 >> 8) & 255;
  const b1 = c1 & 255;

  const r2 = (c2 >> 16) & 255;
  const g2 = (c2 >> 8) & 255;
  const b2 = c2 & 255;

  const r = Math.round(r1 + (r2 - r1) * factor);
  const g = Math.round(g1 + (g2 - g1) * factor);
  const b = Math.round(b1 + (b2 - b1) * factor);

  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`;
}

/**
 * Validate email address
 */
function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

/**
 * Generate unique ID
 */
function generateId() {
  return `_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Format time difference in human readable format
 */
function formatTimeDiff(seconds) {
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

/**
 * Local storage helper
 */
const storage = {
  get: (key, defaultValue = null) => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (e) {
      return defaultValue;
    }
  },
  
  set: (key, value) => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (e) {
      return false;
    }
  },
  
  remove: (key) => {
    try {
      localStorage.removeItem(key);
      return true;
    } catch (e) {
      return false;
    }
  },
  
  clear: () => {
    try {
      localStorage.clear();
      return true;
    } catch (e) {
      return false;
    }
  }
};

/**
 * API helper for common requests
 */
const apiHelper = {
  async get(url, options = {}) {
    const response = await fetch(url, {
      method: 'GET',
      ...options
    });
    return response.json();
  },

  async post(url, data, options = {}) {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...options.headers },
      body: JSON.stringify(data),
      ...options
    });
    return response.json();
  },

  async put(url, data, options = {}) {
    const response = await fetch(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', ...options.headers },
      body: JSON.stringify(data),
      ...options
    });
    return response.json();
  },

  async delete(url, options = {}) {
    const response = await fetch(url, {
      method: 'DELETE',
      ...options
    });
    return response.json();
  }
};

// Export utilities (if using modules)
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    formatPercent,
    formatNumber,
    calculateDistance,
    getCoordinateBounds,
    getRandomColor,
    debounce,
    throttle,
    deepClone,
    isEmpty,
    getNestedValue,
    normalize,
    weightedAverage,
    groupBy,
    boundsToZoom,
    createHeatmapData,
    interpolateColor,
    isValidEmail,
    generateId,
    formatTimeDiff,
    storage,
    apiHelper
  };
}
