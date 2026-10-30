import os
import joblib
import numpy as np
from agents.minimax_bot import Minimax
from utils.features import extract_features
from persistence.database import fetch_model_registry_from_db

class RandomForestMinimaxBot(Minimax):
    _model_cache = {}

    def __init__(self, player_id=1, depth=3, model_path=None, model_version=None):
        super().__init__(player_id=player_id, depth=depth)
        
        resolved_path = model_path
        
        if resolved_path is None:
            try:
                registry = fetch_model_registry_from_db()
                models_dict = registry.get("models", {})
                
                target_key = model_version
                if model_version == "champion":
                    target_key = registry.get("champion") or registry.get("latest_model")
                elif model_version == "latest" or not model_version:
                    target_key = registry.get("latest_model")
                    
                if target_key and target_key in models_dict:
                    candidate_path = models_dict[target_key].get("model_path")
                    if candidate_path and os.path.exists(candidate_path):
                        resolved_path = candidate_path
            except Exception as e:
                print(f"Error querying model registry from DB: {e}")
        
        if resolved_path and os.path.exists(resolved_path):
            if resolved_path not in RandomForestMinimaxBot._model_cache:
                RandomForestMinimaxBot._model_cache[resolved_path] = joblib.load(resolved_path)
                print(f"Successfully loaded Random Forest model from {resolved_path}")
            
            self.model = RandomForestMinimaxBot._model_cache[resolved_path]
        else:
            self.model = None
            print(f"Warning: Random Forest model file not found at '{resolved_path}'. Falling back to 0 score heuristic.")

    def score_position(self, state, piece) -> int:
        if self.model is None:
            return 0
        
        features = np.array([extract_features(state)])
        pred = self.model.predict(features)[0]
        scaled_score = int(pred * 10000)
        return scaled_score if piece == self.player_id else -scaled_score