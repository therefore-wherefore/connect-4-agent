import os
import torch
import torch.nn as nn
import numpy as np
from agents.minimax_bot import Minimax
from utils.features import extract_features
from persistence.database import fetch_model_registry_from_db

class Connect4Net(nn.Module):
    """Dual-head neural network architecture for Connect 4."""
    def __init__(self, input_dimension=11, hidden_dimensions=[128, 128], output_policy=7):
        super(Connect4Net, self).__init__()
        self.hidden_dims = hidden_dimensions
        
        layers = []
        current_dimension = input_dimension
        for this_hidden_dimension in hidden_dimensions:
            layers.append(nn.Linear(current_dimension, this_hidden_dimension))
            layers.append(nn.ReLU())
            current_dimension = this_hidden_dimension
            
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Sequential(nn.Linear(current_dimension, output_policy), nn.Softmax(dim=-1))
        self.value_head = nn.Sequential(nn.Linear(current_dimension, 1), nn.Tanh())

    def forward(self, x):
        features = self.shared(x)
        return self.policy_head(features), self.value_head(features)

class NeuroMinimaxBot(Minimax):
    _model_cache = {}

    def __init__(self, player_id=1, depth=3, model_path=None, model_version=None):
        super().__init__(player_id=player_id, depth=depth)
        
        resolved_path = model_path
        hidden_dims = [128, 128]  # Default fallback architecture
        
        if resolved_path is None:
            try:
                registry = fetch_model_registry_from_db()
                models_dict = registry.get("models", {})
                
                target_key = model_version
                if model_version == "champion": target_key = registry.get("champion") or registry.get("latest_model")
                elif model_version == "latest" or not model_version:    target_key = registry.get("latest_model")
                    
                if target_key and target_key in models_dict:
                    model_info = models_dict[target_key]
                    candidate_path = model_info.get("model_path")
                    if candidate_path and os.path.exists(candidate_path):
                        resolved_path = candidate_path
                        # Retrieve exact hyperparameters stored during NAS registration
                        hyperparams = model_info.get("hyperparameters", {})
                        if "hidden_dims" in hyperparams:
                            hidden_dims = hyperparams["hidden_dims"]
            except Exception as e:
                print(f"Error querying model registry from DB for NeuroBot: {e}")
        
        if resolved_path and os.path.exists(resolved_path):
            if resolved_path not in NeuroMinimaxBot._model_cache:
                # Instantiate architecture and load state dictionary securely
                model = Connect4Net(hidden_dimensions=hidden_dims)
                state_dict = torch.load(resolved_path, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
                model.eval()
                
                NeuroMinimaxBot._model_cache[resolved_path] = model
                print(f"Successfully loaded Neural Network weights from {resolved_path}")
            
            self.model = NeuroMinimaxBot._model_cache[resolved_path]
            if hasattr(self.model, "n_jobs"):   self.model.n_jobs = 1
        else:
            self.model = None
            print(f"Warning: Neural Network model file not found at '{resolved_path}'. Falling back to 0 score heuristic.")

    def score_position(self, state, piece) -> int:
        if self.model is None:  return 0
        
        features = extract_features(state)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(features_tensor)
            
            if isinstance(output, tuple):
                _, value_tensor = output
                raw_score = value_tensor.item()
            else:   
                raw_score = output.item()
                
        scaled_score = int(raw_score * 10000)
        return scaled_score if piece == self.player_id else -scaled_score