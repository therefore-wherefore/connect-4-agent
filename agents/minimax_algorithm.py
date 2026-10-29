import math
from environment.board import (
    get_valid_locations,
    drop_piece,
    winning_move,
    is_terminal_node
)

class Minimax:
    def __init__(self, player_id=1, depth=4):
        self.player_id = player_id
        self.opponent_id = 1 - player_id
        self.depth = depth

    def score_position(self, state, piece) -> int:
        """
        Base evaluation hook. Subclasses override this method to provide
        custom heuristics, neural networks, or tree evaluators.
        """
        raise NotImplementedError("Subclasses must implement score_position()")

    def minimax(self, state, depth, alpha, beta, maximizing_player):
        valid_moves = get_valid_locations(state)
        is_terminal = is_terminal_node(state)

        if depth == 0 or is_terminal:
            if is_terminal:
                if winning_move(state, self.player_id):     
                    return (None, 10000000)
                elif winning_move(state, self.opponent_id): 
                    return (None, -10000000)
                else:                                       
                    return (None, 0)
            else:   
                return (None, self.score_position(state, self.player_id))

        if maximizing_player:
            score = -math.inf
            best_col = valid_moves[0] if valid_moves else None
            for col in valid_moves:
                next_state = drop_piece(state, col, self.player_id)
                new_score = self.minimax(next_state, depth - 1, alpha, beta, False)[1]
                if new_score > score:   
                    score, best_col = new_score, col
                alpha = max(alpha, score)
                if alpha >= beta:
                    break  # Beta cutoff
            return best_col, score
        else:
            score = math.inf
            best_col = valid_moves[0] if valid_moves else None
            for col in valid_moves:
                next_state = drop_piece(state, col, self.opponent_id)
                new_score = self.minimax(next_state, depth - 1, alpha, beta, True)[1]
                if new_score < score:
                    score, best_col = new_score, col
                beta = min(beta, score)
                if alpha >= beta:
                    break  # Alpha cutoff
            return best_col, score

    def get_move(self, state):
        col, _ = self.minimax(state, self.depth, -math.inf, math.inf, True)
        return col