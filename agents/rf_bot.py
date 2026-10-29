import os
import joblib
import numpy as np
from environment.board import get_valid_locations, drop_piece, winning_move
from utils.features import extract_board_features
from persistence.database import get_db_connection

class RandomForestMinimaxBot:
    def __init__(self, player_id, depth=3, model_version=None):
        self.player_id = player_id
        self.depth = depth
        self.model = self._load_model(model_version)

    def _load_model(self, model_version):
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if model_version:
            cursor.execute("SELECT model_path FROM models WHERE version_id = ?", (model_version,))
        else:
            cursor.execute("SELECT model_path FROM models WHERE class_name = 'RandomForestMinimaxBot' ORDER BY created_at DESC LIMIT 1")
            
        row = cursor.fetchone()
        conn.close()
        
        if row and os.path.exists(row["model_path"]):
            return joblib.load(row["model_path"])
        raise FileNotFoundError("No trained Random Forest model checkpoint found in SQLite registry.")

    def get_move(self, state):
        valid_moves = get_valid_locations(state)
        if not valid_moves:
            return None
            
        best_score = -float('inf')
        best_move = valid_moves[0]
        
        for col in valid_moves:
            temp_state = drop_piece(state, col, self.player_id)
            if winning_move(temp_state, self.player_id):
                return col
                
            features = extract_board_features(temp_state, self.player_id)
            score = self.model.predict(features.reshape(1, -1))[0]
            
            if score > best_score:
                best_score = score
                best_move = col
                
        return best_move