import random
from environment.board import (
    get_valid_locations,
    drop_piece,
    winning_move
)

class UniformDistributionBot:
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.opponent_id = 1 - player_id

    def get_move(self, state):
        valid_moves = get_valid_locations(state)
        if not valid_moves:
            return None

        # 1. Immediate win check
        for col in valid_moves:
            temp_state = drop_piece(state, col, self.player_id)
            if winning_move(temp_state, self.player_id):
                return col

        # 2. Immediate block check
        for col in valid_moves:
            temp_state = drop_piece(state, col, self.opponent_id)
            if winning_move(temp_state, self.opponent_id):
                return col

        # 3. Uniform random selection across all valid columns
        return random.choice(valid_moves)