CREATE TABLE IF NOT EXISTS models (
    version_id TEXT PRIMARY KEY,
    model_path TEXT NOT NULL,
    class_name TEXT NOT NULL,
    hyperparameters TEXT,          -- JSON string of hyperparameters
    training_mse REAL,
    git_commit TEXT,
    dataset_checksum TEXT,
    training_samples INTEGER,
    is_champion INTEGER DEFAULT 0, -- 1 if active champion, 0 otherwise
    created_at TEXT
);