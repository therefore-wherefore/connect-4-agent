import os
import sys
import time
import numpy as np

# Automatically find project root and ensure clean imports
if (PROJECT_ROOT := os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.board import create_board, get_valid_locations, drop_piece, is_terminal_node, winning_move
from agents.uniform import UniformDistributionBot
from agents.normal_dist import NormalDistributionBot
from agents.minimax_bot import MiniMaxBot
from persistence.database import init_db, log_game_record

AGENT_MAP = {
    "random_bot": lambda pid: UniformDistributionBot(player_id=pid),
    "normal_bot": lambda pid: NormalDistributionBot(player_id=pid, sigma=1.2),
    "minimax_bot": lambda pid: MiniMaxBot(player_id=pid, depth=3)
}

def play_single_game(p0_type, p1_type):
    """Simulates a single Connect 4 match between two agents and returns the winner and moves."""
    state = create_board()
    turn = 0
    agents = {
        0: AGENT_MAP[p0_type](0),
        1: AGENT_MAP[p1_type](1)
    }
    moves_history = []
    
    while not is_terminal_node(state):
        valid_moves = get_valid_locations(state)
        if not valid_moves:
            break
            
        current_agent = agents[turn]
        col = current_agent.get_move(state)
        
        if col is None or col not in valid_moves:
            col = np.random.choice(valid_moves)
            
        moves_history.append(int(col))
        state = drop_piece(state, col, turn)
        
        if winning_move(state, turn):
            winner = p0_type if turn == 0 else p1_type
            return winner, moves_history
        elif is_terminal_node(state):
            return "tie", moves_history
            
        turn = 1 - turn
        
    return "tie", moves_history

def generate_batch(p0_type, p1_type, num_games):
    """Runs a batch of games for a specific matchup and persists telemetry to SQLite."""
    print(f"\n[Dataset Generation] Running {num_games} games: {p0_type} (P0) vs {p1_type} (P1)...")
    for i in range(num_games):
        winner, moves = play_single_game(p0_type, p1_type)
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "game_type": "dataset_generation",
            "player_0": p0_type,
            "player_1": p1_type,
            "winner": winner,
            "total_moves": len(moves),
            "search_depth": 3,
            "moves": moves
        }
        log_game_record(record)
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{num_games} games completed.")

if __name__ == "__main__":
    print("=== MLOps Dataset Generation Pipeline ===")
    
    # Ensure database schema exists
    init_db()
    
    # Define exact matchup quotas requested
    matchups = [
        ("random_bot", "random_bot", 200),
        ("random_bot", "normal_bot", 200),
        ("normal_bot", "minimax_bot", 200),
        ("random_bot", "minimax_bot", 200)
    ]
    
    start_time = time.time()
    for p0, p1, count in matchups:
        generate_batch(p0, p1, count)
        
    print(f"\nSuccess! All 800 simulation records generated and committed to SQLite in {time.time() - start_time:.2f} seconds.")