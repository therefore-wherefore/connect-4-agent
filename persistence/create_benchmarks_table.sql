CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    agent_a TEXT,                  -- e.g., xgb_bot:xgb_bot_20261029_173223
    agent_b TEXT,                  -- e.g., minimax_bot
    agent_a_wins INTEGER,
    agent_b_wins INTEGER,
    ties INTEGER,
    total_games INTEGER,
    fingerprints TEXT              -- JSON string storing code/model hashes for reproducibility
);