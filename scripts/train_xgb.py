import os
import sys
import time
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

if (PROJECT_ROOT := os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) not in sys.path:    sys.path.insert(0, PROJECT_ROOT)

from config import MODELS_DIR, DEFAULT_HYPERPARAMS
from persistence.database import init_db, register_model_record, log_training_run_record
from utils.dataset import load_training_dataset
from utils.mlops import get_git_commit_hash, check_git_cleanliness, get_data_checksum

if __name__ == "__main__":
    print("Starting strict MLOps training pipeline...")
    
    
    init_db()
    
    check_git_cleanliness()
    
    db_samples, total_db_games = load_training_dataset()
    print(f"Loaded {total_db_games} total games from SQLite database.")
    
    training_data_pairs = list(db_samples)
    
    if not training_data_pairs:
        print("Error: Training dataset is empty. Run scripts/generate_datasets.py first.")
        sys.exit(1)

    X = np.array([pair[0] for pair in training_data_pairs])
    y = np.array([pair[1] for pair in training_data_pairs])
    
    print(f"Total training dataset size: {len(X)} feature vectors.")
    print("Training XGBoost model...")
    
    model = xgb.XGBRegressor(**DEFAULT_HYPERPARAMS)
    train_start = time.time()
    model.fit(X, y)
    train_duration = time.time() - train_start
    
    preds = model.predict(X)
    train_mse = float(mean_squared_error(y, preds))
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    version_id = f"xgb_bot_{timestamp_str}"
    model_filename = os.path.join(MODELS_DIR, f"connect4_xgb_{timestamp_str}.json")
    
    joblib.dump(model, model_filename)
    print(f"Success! Versioned model saved to {model_filename}")
    
    seed = DEFAULT_HYPERPARAMS.get("random_state", 42)
    lineage_metadata = {
        "git_commit": get_git_commit_hash(),
        "dataset_checksum": get_data_checksum(X, y),
        "training_samples": len(X),
        "random_seed": seed
    }
    
    register_model_record(
        version_id=version_id,
        model_path=model_filename,
        class_name="XGBMinimaxBot",
        hyperparameters=DEFAULT_HYPERPARAMS,
        training_mse=train_mse,
        lineage=lineage_metadata,
        is_champion=1
    )
    
    run_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version_id": version_id,
        "historical_games_reused": total_db_games,
        "new_games_generated": 0,
        "total_samples": len(X),
        "training_duration_seconds": round(train_duration, 2),
        "training_mse": train_mse
    }
    log_training_run_record(run_record)
    
    print(f"Training telemetry logged and Active Champion '{version_id}' successfully updated in SQLite database.")