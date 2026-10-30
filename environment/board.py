def create_board():
    """Returns an empty board state: (p0_mask, occupancy_mask, moves_count)"""
    return (0, 0, 0)

def copy_board(state):
    """Returns a copy of the state (tuples are immutable, so a shallow copy is safe)."""
    return (state[0], state[1], state[2])

def get_valid_locations(state):
    """Returns a list of valid columns (0-6) where a piece can be dropped."""
    _, occupancy, _ = state

    return [col for col in range(7) if not (occupancy & (1 << (col * 7 + 5)))]

def drop_piece(state, col, player_id):
    """
    Drops a piece for player_id (0 or 1) into column col.
    Returns a new immutable state tuple.
    """
    p0_mask, occupancy, count = state
    
    for row in range(6):
        bit_pos = col * 7 + row
        if not (occupancy & (1 << bit_pos)):
            new_occupancy = occupancy | (1 << bit_pos)
            new_p0 = p0_mask | (1 << bit_pos) if player_id == 0 else p0_mask
            return (new_p0, new_occupancy, count + 1)
            
    raise ValueError(f"Column {col} is full!")

def winning_move(state, player_id):
    """Checks if player_id (0 or 1) has 4 in a row using bitwise shifts."""
    p0_mask, occupancy, _ = state
    
    if player_id == 0:
        mask = p0_mask
    else:
        mask = occupancy ^ p0_mask  
        
    m = mask & (mask >> 1)
    if m & (m >> 2):    return True  # Vertical
    
    m = mask & (mask >> 7)
    if m & (m >> 14):   return True  # Horizontal
    
    m = mask & (mask >> 6)
    if m & (m >> 12):   return True  # Diagonal /
    
    m = mask & (mask >> 8)
    if m & (m >> 16):   return True  # Diagonal \
    
    return False

def is_terminal_node(state):
    """Returns True if either player has won or the board is completely full (42 moves)."""
    _, _, count = state
    return winning_move(state, 0) or winning_move(state, 1) or count == 42