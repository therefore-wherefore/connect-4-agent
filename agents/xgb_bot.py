import os
import json
import joblib
import numpy as np
from agents.minimax_bot import Minimax

def extract_features(state):
    """Extracts 11 structural features."""
    p1, p2, count = state
    features = []
    
    current_board = p1 | p2
    col_heights = []
    for col in range(7):
        height = 0
        for row in range(6):
            if current_board & (1 << (col * 7 + row)):
                height += 1
        col_heights.append(height)
    
    # 1. Column heights (7 features)
    features.extend(col_heights)
    # 2. Center preference score (1 feature)
    features.append(col_heights[3] * 3 + (col_heights[2] + col_heights[4]) * 2)
    # 3. Piece counts and normalized move count (3 features)
    features.extend([bin(p1).count('1'), bin(p2).count('1'), count / 42.0])
    
    return np.array(features, dtype=np.float32)

class XGBMinimaxBot(Minimax):
    # Class-level cache to share models across instances with zero redundant disk I/O
    _model_cache = {}

    def __init__(self, player_id=1, depth=3, model_path=None, model_version="champion"):
        super().__init__(player_id=player_id, depth=depth)
        
        resolved_path = model_path
        
        if resolved_path is None:
            registry_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "registry.json"))
            if os.path.exists(registry_path):
                try:
                    with open(registry_path, "r") as f:
                        reg_data = json.load(f)
                        
                        # Resolve semantic aliases: "champion", "latest", or a direct version key
                        target_key = model_version
                        if model_version == "champion":
                            target_key = reg_data.get("champion") or reg_data.get("latest_model")
                        elif model_version == "latest":
                            target_key = reg_data.get("latest_model")
                            
                        if target_key and target_key in reg_data.get("models", {}):
                            candidate_path = reg_data["models"][target_key].get("model_path")
                            if candidate_path and os.path.exists(candidate_path):
                                resolved_path = candidate_path
                except Exception:
                    pass
            
            # Fallback to legacy path if registry lookup fails
            if resolved_path is None or not os.path.exists(resolved_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                legacy_candidate = os.path.abspath(os.path.join(current_dir, "..", "connect4_xgb.json"))
                if os.path.exists(legacy_candidate):
                    resolved_path = legacy_candidate

        if resolved_path and os.path.exists(resolved_path):
            if resolved_path not in XGBMinimaxBot._model_cache:
                XGBMinimaxBot._model_cache[resolved_path] = joblib.load(resolved_path)
                print(f"Successfully loaded XGBoost model from {resolved_path}")
            
            self.model = XGBMinimaxBot._model_cache[resolved_path]
        else:
            self.model = None
            print(f"Warning: XGBoost model not found. Falling back to 0 score heuristic.")

    def score_position(self, state, piece) -> int:
        if self.model is None:
            return 0
        
        features = np.array([extract_features(state)])
        pred = self.model.predict(features)[0]
        scaled_score = int(pred * 10000)
        return scaled_score if piece == self.player_id else -scaled_score