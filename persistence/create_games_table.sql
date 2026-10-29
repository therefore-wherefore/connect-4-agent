CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    game_type TEXT,
    player_0 TEXT,
    player_1 TEXT,
    winner TEXT,
    total_moves INTEGER,
    search_depth INTEGER,
    moves TEXT
);