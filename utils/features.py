import numpy as np

def extract_features(state):
    """Extracts 11 structural features from a Connect 4 board state."""
    p1, p2, count = state
    features = []
    
    current_board = p1 | p2
    col_heights = []
    for col in range(7):
        height = 0
        for row in range(6):
            if current_board & (1 << (col * 7 + row)):
                height += 1
        col_heights.append(height)
    
    
    features.extend(col_heights)
    
    features.append(col_heights[3] * 3 + (col_heights[2] + col_heights[4]) * 2)
    
    features.extend([bin(p1).count('1'), bin(p2).count('1'), count / 42.0])
    
    return np.array(features, dtype=np.float32)
