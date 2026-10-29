# config.py
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Transition from JSONL to SQLite
DB_PATH = os.path.join(LOG_DIR, "game_database.db")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, "training_log.json")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")

DEFAULT_HYPERPARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 4,
    "objective": "reg:squarederror",
    "random_state": 42
}