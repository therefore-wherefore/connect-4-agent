from agents.minimax_algorithm import Minimax

class MiniMaxBot(Minimax):
    def score_position(self, state, piece) -> int:
        """Simple scoring heuristic counting center column piece dominance."""
        score = 0
        p1_mask, p2_mask, _ = state
        mask = p1_mask if piece == 0 else p2_mask

        # Score center column preference (Column 3)
        center_count = 0
        for r in range(6):
            bit_pos = 3 * 7 + r
            if mask & (1 << bit_pos):
                center_count += 1
        
        score += center_count * 3
        return score