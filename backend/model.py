import os
import json
import random
import numpy as np

try:
    import torch
except Exception:
    torch = None


class ModelWrapper:
    def __init__(self, paths=None):
        self.model = None
        self.model_path = None
        if isinstance(paths, (list, tuple)):
            for p in paths:
                if os.path.exists(p):
                    self.model_path = p
                    break
        elif isinstance(paths, str) and os.path.exists(paths):
            self.model_path = paths

        if self.model_path and torch is not None:
            # Try TorchScript first
            try:
                loaded = torch.jit.load(self.model_path, map_location='cpu')
            except Exception:
                try:
                    loaded = torch.load(self.model_path, map_location='cpu')
                except Exception:
                    loaded = None

            # Ensure loaded object behaves like a model (has eval attribute)
            if loaded is not None and hasattr(loaded, 'eval'):
                self.model = loaded
            else:
                # If we only got a state_dict (OrderedDict) or similar, ignore and fallback
                self.model = None

    def predict(self, features):
        # Features can be a list, number or dict. Return list of floats.
        if self.model is None or torch is None:
            # Return a deterministic pseudo-random prediction for now
            if isinstance(features, dict):
                n = int(features.get('n', 1))
            elif isinstance(features, list):
                n = len(features)
            else:
                n = 1
            random.seed(0)
            return [round(random.random() * 10, 4) for _ in range(n)]

        # Attempt to convert features to a tensor and run the model
        try:
            if isinstance(features, dict):
                # try to find numeric array inside
                arr = features.get('array') or features.get('features') or []
            else:
                arr = features
            arr = np.array(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            tensor = torch.from_numpy(arr)
            self.model.eval()
            with torch.no_grad():
                out = self.model(tensor)
            try:
                out_np = out.cpu().numpy()
            except Exception:
                out_np = np.array(out)
            # Flatten to list
            return out_np.reshape(-1).tolist()
        except Exception as e:
            return {'error': str(e)}
