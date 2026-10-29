# utils/dataset.py
import json
from persistence.database import fetch_all_games
from utils.features import extract_features
from environment.board import create_board, drop_piece

def load_training_dataset():
    """Parses database rows, replays moves to get historical states, and returns X, y tensors."""
    rows = fetch_all_games()
    if not rows:
        return [], 0
        
    X, y = [], []
    total_games_loaded = 0
    
    for row in rows:
        try:
            moves = json.loads(row["moves"]) if isinstance(row["moves"], str) else row["moves"]
            winner = row["winner"]
            p0 = row["player_0"]
            
            state = create_board()
            game_states = []
            
            for turn_idx, col in enumerate(moves):
                player_id = turn_idx % 2
                game_states.append(state)
                state = drop_piece(state, col, player_id)
            
            reward = 0.0
            if winner == p0 and winner != "tie":
                reward = 1.0
            elif winner != "tie" and winner is not None:
                reward = -1.0
                
            for s in game_states:
                X.append(extract_features(s))
                y.append(reward)
                
            total_games_loaded += 1
        except Exception:
            continue
            
    return list(zip(X, y)), total_games_loaded