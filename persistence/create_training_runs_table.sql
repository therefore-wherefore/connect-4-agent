CREATE TABLE IF NOT EXISTS training_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    version_id TEXT,               -- Foreign key linked to models.version_id
    historical_games_reused INTEGER,
    new_games_generated INTEGER,
    total_samples INTEGER,
    training_duration_seconds REAL,
    training_mse REAL,
    FOREIGN KEY (version_id) REFERENCES models(version_id)
);