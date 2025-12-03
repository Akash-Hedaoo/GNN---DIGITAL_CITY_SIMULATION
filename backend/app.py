from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.model import ModelWrapper

app = Flask(__name__)
CORS(app)

# Try common model filenames in repo
MODEL_CANDIDATES = [
    'real_city_gnn.pt',
    'trained_gnn.pt',
    'real_city_gnn.pth',
    'trained_model.pt',
]

model = ModelWrapper(MODEL_CANDIDATES)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True, silent=True) or {}
    features = payload.get('features', payload)
    preds = model.predict(features)
    return jsonify({'predictions': preds})


@app.route('/whatif', methods=['POST'])
def whatif():
    payload = request.get_json(force=True, silent=True) or {}
    scenario = payload.get('scenario', {})
    # For now we proxy to predict using scenario['features'] if present
    features = scenario.get('features', {}) if isinstance(scenario, dict) else {}
    preds_before = model.predict(payload.get('baseline_features', []))
    preds_after = model.predict(features)
    return jsonify({'before': preds_before, 'after': preds_after})


@app.route('/city-data', methods=['GET'])
def city_data():
    """Return city graph data for map visualization"""
    city_graph = {
        'nodes': [
            {'id': 1, 'lat': 40.7128, 'lng': -74.0060, 'name': 'Central Hub'},
            {'id': 2, 'lat': 40.7180, 'lng': -74.0040, 'name': 'North Station'},
            {'id': 3, 'lat': 40.7100, 'lng': -74.0100, 'name': 'South Station'},
            {'id': 4, 'lat': 40.7200, 'lng': -74.0150, 'name': 'East Terminal'},
            {'id': 5, 'lat': 40.7050, 'lng': -74.0000, 'name': 'West Terminal'},
        ],
        'edges': [
            {'id': 1, 'source': 1, 'target': 2, 'flow': 450, 'congestion': 0.3},
            {'id': 2, 'source': 1, 'target': 3, 'flow': 380, 'congestion': 0.25},
            {'id': 3, 'source': 2, 'target': 4, 'flow': 290, 'congestion': 0.4},
            {'id': 4, 'source': 3, 'target': 5, 'flow': 320, 'congestion': 0.35},
            {'id': 5, 'source': 4, 'target': 5, 'flow': 410, 'congestion': 0.45},
        ],
        'amenities': [
            {'id': 'h1', 'lat': 40.7140, 'lng': -74.0030, 'name': 'City Hospital', 'type': 'hospital'},
            {'id': 's1', 'lat': 40.7160, 'lng': -74.0070, 'name': 'Central School', 'type': 'school'},
            {'id': 'p1', 'lat': 40.7110, 'lng': -74.0120, 'name': 'Central Park', 'type': 'park'},
            {'id': 'r1', 'lat': 40.7090, 'lng': -74.0050, 'name': 'Downtown Diner', 'type': 'restaurant'},
        ],
        'metros': [
            {'id': 'm1', 'lat': 40.7128, 'lng': -74.0060, 'name': 'Times Square', 'line': 'Red', 'stations': 15},
            {'id': 'm2', 'lat': 40.7180, 'lng': -74.0040, 'name': 'Grand Central', 'line': 'Blue', 'stations': 20},
        ]
    }
    return jsonify(city_graph)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
