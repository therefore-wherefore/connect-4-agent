import os
import sys
import cProfile
import pstats

# Ensure root directory is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.board import create_board, is_terminal_node, drop_piece
from agents.xgb_bot import XGBMinimaxBot
from agents.minimax_bot import MiniMaxBot

def run_benchmark():
    state = create_board()
    
    # Profile your XGBoost champion against the minimax baseline or another XGB bot
    bot0 = XGBMinimaxBot(player_id=0, depth=3, model_version="champion")
    bot1 = MiniMaxBot(player_id=1, depth=3)
    
    bots = {0: bot0, 1: bot1}
    turn = 0
    
    # Simulate enough turns to stress-test tree search and feature extraction
    for _ in range(6): 
        if is_terminal_node(state):
            break
        current_bot = bots[turn]
        col = current_bot.get_move(state)
        state = drop_piece(state, col, turn)
        turn = 1 - turn

if __name__ == "__main__":
    print("Profiling XGBMinimaxBot execution and feature extraction...")
    
    profile_filename = "xgb_minimax_stats.prof"
    cProfile.run("run_benchmark()", profile_filename)
    
    # Load and print top 20 most time-consuming functions sorted by cumulative time
    p = pstats.Stats(profile_filename)
    p.strip_dirs().sort_stats("cumulative").print_stats(20)