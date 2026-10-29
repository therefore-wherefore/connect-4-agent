import random
import math
from environment.board import *

class NormalDistributionBot:
    def __init__(self, player_id=0, sigma=1.5):
        self.player_id = player_id
        self.opponent_id = 1 - player_id
        self.sigma = sigma

    def get_move(self, state):
        valid_moves = get_valid_locations(state)

        for col in valid_moves:
            temp_state = drop_piece(state, col, self.player_id)
            if winning_move(temp_state, self.player_id):    return col

        for col in valid_moves:
            temp_state = drop_piece(state, col, self.opponent_id)
            if winning_move(temp_state, self.opponent_id):  return col

        # 3. Normal distribution weighting centered at column 3
        mean = 3.0
        weights = [math.exp(-0.5 * ((col - mean) / self.sigma) ** 2) for col in valid_moves]

        return random.choices(valid_moves, weights=weights, k=1)[0]