import pygame
import sys
import time
import numpy as np
from environment.board import *
from agents.uniform import UniformDistributionBot
from agents.normal_dist import NormalDistributionBot
from agents.minimax_bot import MiniMaxBot
from agents.xgb_bot import XGBMinimaxBot
from persistence.database import init_db, log_game_record

# 1. AGENT REGISTRY
AGENT_REGISTRY = {
    "random_bot": UniformDistributionBot,
    "normal_bot": NormalDistributionBot,
    "minimax_bot": MiniMaxBot,
    "xgb_bot": XGBMinimaxBot,
}

# 2. CONFIGURATION: Set your interactive game mode here
PLAYER_0_TYPE = "xgb_bot"       # Options: "human", or any key in AGENT_REGISTRY
PLAYER_1_TYPE = "minimax_bot"   # Options: "human", or any key in AGENT_REGISTRY
DEPTH = 4

SQUARESIZE = 100
WIDTH = 7 * SQUARESIZE
HEIGHT = (6 + 1) * SQUARESIZE
SIZE = (WIDTH, HEIGHT)

BLUE = (28, 110, 204)
BLACK = (20, 20, 20)
RED = (220, 50, 50)
YELLOW = (240, 200, 50)
WHITE = (245, 245, 245)

def draw_board(screen, state):
    p0_mask, occupancy, _ = state
    p1_mask = occupancy ^ p0_mask  # Derive Player 1 pieces
    
    # Draw background board
    screen.fill(BLUE, (0, SQUARESIZE, WIDTH, HEIGHT - SQUARESIZE))
    
    for c in range(7):
        for r in range(6):
            bit_pos = c * 7 + r
            color = BLACK
            if (p0_mask & (1 << bit_pos)):
                color = RED
            elif (p1_mask & (1 << bit_pos)):
                color = YELLOW
                
            pygame.draw.circle(
                screen, color, 
                (int(c * SQUARESIZE + SQUARESIZE / 2), int(HEIGHT - (r * SQUARESIZE + SQUARESIZE / 2))), 
                int(SQUARESIZE / 2 - 10)
            )
            
    pygame.display.update()

def initialise_agent(p_type, player_id):
    """Dynamically instantiates an agent from the registry based on string type."""
    if p_type == "human":   return None

    if p_type not in AGENT_REGISTRY:raise ValueError(f"Unknown agent type '{p_type}'. Register it in AGENT_REGISTRY.")
    
    agent_class = AGENT_REGISTRY[p_type]
    
    if p_type == "normal_bot":      return agent_class(player_id=player_id, sigma=1.2)
    elif p_type == "minimax_bot":   return agent_class(player_id=player_id, depth=DEPTH)
    elif p_type == "xgb_bot":       return agent_class(player_id=player_id, depth=DEPTH)
    return agent_class(player_id=player_id)

def run_game():
    # Ensure database schema is provisioned
    init_db()
    
    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption(f"Connect 4 - {PLAYER_0_TYPE} (Red) vs {PLAYER_1_TYPE} (Yellow)")
    
    state = create_board()
    game_over = False
    turn = 0  # 0: Player 0 (Red), 1: Player 1 (Yellow)
    moves_history = []
    
    agents = {
        0: initialise_agent(PLAYER_0_TYPE, 0),
        1: initialise_agent(PLAYER_1_TYPE, 1)
    }
    
    screen.fill(BLACK)
    draw_board(screen, state)
    clock = pygame.time.Clock()
    
    winning_agent_label = "tie"
    
    while not game_over:
        clock.tick(30)
        
        current_player_type = PLAYER_0_TYPE if turn == 0 else PLAYER_1_TYPE
        current_agent = agents[turn]
        
        # Handle Bot Turn
        if current_player_type != "human" and not game_over:
            col = current_agent.get_move(state)
            
            if col is not None and col in get_valid_locations(state):
                moves_history.append(int(col))
                state = drop_piece(state, col, turn)
                
                player_color_label = "Red (Player 0)" if turn == 0 else "Yellow (Player 1)"
                if winning_move(state, turn):
                    print(f"{current_player_type} [{player_color_label}] Wins!")
                    winning_agent_label = current_player_type
                    game_over = True
                elif is_terminal_node(state):
                    print("It's a Tie!")
                    winning_agent_label = "tie"
                    game_over = True
                    
                turn = 1 - turn
                draw_board(screen, state)
            continue
            
        # Handle Human Turn Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.MOUSEMOTION and current_player_type == "human" and not game_over:
                screen.fill(BLACK, (0, 0, WIDTH, SQUARESIZE))
                posx = event.pos[0]
                piece_color = RED if turn == 0 else YELLOW
                pygame.draw.circle(screen, piece_color, (posx, int(SQUARESIZE / 2)), int(SQUARESIZE / 2 - 10))
                draw_board(screen, state)
                
            if event.type == pygame.MOUSEBUTTONDOWN and current_player_type == "human" and not game_over:
                screen.fill(BLACK, (0, 0, WIDTH, SQUARESIZE))
                posx = event.pos[0]
                col = posx // SQUARESIZE
                
                valid_moves = get_valid_locations(state)
                if col in valid_moves:
                    moves_history.append(int(col))
                    state = drop_piece(state, col, turn)
                    
                    player_color_label = "Red (Player 0)" if turn == 0 else "Yellow (Player 1)"
                    if winning_move(state, turn):
                        print(f"Human [{player_color_label}] Wins!")
                        winning_agent_label = "human"
                        game_over = True
                    elif is_terminal_node(state):
                        print("It's a Tie!")
                        winning_agent_label = "tie"
                        game_over = True
                        
                    turn = 1 - turn
                    draw_board(screen, state)
                    
    # Log interactive game telemetry directly into SQLite database
    game_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "game_type": "interactive",
        "player_0": PLAYER_0_TYPE,
        "player_1": PLAYER_1_TYPE,
        "winner": winning_agent_label,
        "total_moves": len(moves_history),
        "search_depth": DEPTH,
        "moves": moves_history
    }
    log_game_record(game_record)
    print("Interactive match telemetry successfully persisted to SQLite database.")

    pygame.quit()

if __name__ == "__main__":
    run_game()