#!/usr/bin/env python
"""
Integration test for GNN City Simulator Modern UI
Tests all backend endpoints and frontend requirements
"""

import requests
import json
import sys
import time

BASE_URL = 'http://localhost:5000'

def test_health():
    """Test /health endpoint"""
    print("Testing /health endpoint...", end=" ")
    try:
        response = requests.get(f'{BASE_URL}/health', timeout=2)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert 'status' in data, "Missing 'status' field"
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False

def test_city_data():
    """Test /city-data endpoint"""
    print("Testing /city-data endpoint...", end=" ")
    try:
        response = requests.get(f'{BASE_URL}/city-data', timeout=2)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        required_fields = ['nodes', 'edges', 'amenities', 'metros']
        for field in required_fields:
            assert field in data, f"Missing '{field}' field"
        
        assert isinstance(data['nodes'], list), "nodes must be a list"
        assert isinstance(data['edges'], list), "edges must be a list"
        assert len(data['nodes']) > 0, "nodes list is empty"
        assert len(data['edges']) > 0, "edges list is empty"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False

def test_predict():
    """Test /predict endpoint"""
    print("Testing /predict endpoint...", end=" ")
    try:
        payload = {'features': [0.3, 0.4, 0.2, 0.5, 0.1]}
        response = requests.post(f'{BASE_URL}/predict', json=payload, timeout=2)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert 'predictions' in data, "Missing 'predictions' field"
        assert isinstance(data['predictions'], list), "predictions must be a list"
        assert len(data['predictions']) > 0, "predictions list is empty"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False

def test_whatif():
    """Test /whatif endpoint"""
    print("Testing /whatif endpoint...", end=" ")
    try:
        payload = {
            'features': [0.3, 0.4, 0.2, 0.5, 0.1],
            'scenario': {
                'type': 'road_closure',
                'severity': 0.7,
                'duration': 30
            }
        }
        response = requests.post(f'{BASE_URL}/whatif', json=payload, timeout=2)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert 'before' in data, "Missing 'before' field"
        assert 'after' in data, "Missing 'after' field"
        assert isinstance(data['before'], list), "before must be a list"
        assert isinstance(data['after'], list), "after must be a list"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False

def test_cors():
    """Test CORS headers"""
    print("Testing CORS headers...", end=" ")
    try:
        response = requests.get(f'{BASE_URL}/health')
        headers = response.headers
        assert 'Access-Control-Allow-Origin' in headers, "Missing CORS header"
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False

def test_frontend_files():
    """Check frontend files exist"""
    print("Testing frontend files...", end=" ")
    try:
        import os
        files = [
            'frontend/index.html',
            'frontend/app.js',
            'frontend/styles.css',
            'frontend/utils.js'
        ]
        
        for file in files:
            assert os.path.exists(file), f"Missing {file}"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False

def test_scenario_flow():
    """Test complete scenario execution flow"""
    print("Testing scenario flow...", end=" ")
    try:
        # Get baseline
        baseline = requests.post(f'{BASE_URL}/predict', 
                                json={'features': [0.3, 0.4, 0.2, 0.5, 0.1]}).json()
        baseline_avg = sum(baseline['predictions']) / len(baseline['predictions'])
        
        # Run scenario
        result = requests.post(f'{BASE_URL}/whatif',
                              json={
                                  'features': [0.3, 0.4, 0.2, 0.5, 0.1],
                                  'scenario': {'type': 'road_closure', 'severity': 0.8}
                              }).json()
        
        after_avg = sum(result['after']) / len(result['after'])
        
        # Verify scenario had impact
        assert baseline_avg != after_avg, "Scenario should change predictions"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False

def main():
    print("\n" + "="*50)
    print("GNN City Simulator - Integration Tests")
    print("="*50 + "\n")
    
    # Check if backend is running
    print(f"Backend: {BASE_URL}")
    try:
        requests.get(f'{BASE_URL}/health', timeout=1)
    except:
        print("✗ ERROR: Backend is not running!")
        print("\nStart the backend with:")
        print("  python -m flask run --port 5000")
        return 1
    
    print("✓ Backend is running\n")
    
    # Run tests
    tests = [
        test_health,
        test_city_data,
        test_predict,
        test_whatif,
        test_cors,
        test_frontend_files,
        test_scenario_flow
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*50)
    print(f"Results: {passed}/{total} tests passed")
    print("="*50 + "\n")
    
    if passed == total:
        print("✓ All tests passed! Frontend is ready to use.")
        print("\nStart the frontend with:")
        print("  python -m http.server 8000")
        print("\nThen open: http://localhost:8000")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed. Please review above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
