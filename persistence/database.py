import sqlite3
import os
import json
import time
from config import DB_PATH

PERSISTENCE_DIR = os.path.dirname(__file__)

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Checks if required tables exist; initialises schema via SQL files only if missing."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name IN ('games', 'models', 'training_runs', 'benchmarks');
    """)
    existing_tables = {row["name"] for row in cursor.fetchall()}
    required_tables = {"games", "models", "training_runs", "benchmarks"}
    
    if required_tables.issubset(existing_tables):   return  conn.close()

    schema_files = ["create_games_table.sql", "create_models_table.sql", "create_training_runs_table.sql", "create_benchmarks_table.sql"]
    
    for filename in schema_files:
        file_path = os.path.join(PERSISTENCE_DIR, filename)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                cursor.executescript(f.read())
        else:
            raise FileNotFoundError(f"Critical schema file missing: {file_path}")
            
    conn.commit()
    conn.close()

def log_game_record(record):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO games (timestamp, game_type, player_0, player_1, winner, total_moves, search_depth, moves)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get("timestamp"), record.get("game_type"), record.get("player_0"),
        record.get("player_1"), record.get("winner"), record.get("total_moves"),
        record.get("search_depth"), json.dumps(record.get("moves", []))
    ))
    conn.commit()
    conn.close()

def fetch_all_games():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT player_0, winner, moves FROM games")
    rows = cursor.fetchall()
    conn.close()
    return rows

def fetch_model_registry_from_db():
    """Fetches all registered models from SQLite to mimic the model registry interface."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT version_id, model_path, class_name, hyperparameters, 
               training_mse, git_commit, dataset_checksum, training_samples, 
               is_champion, created_at 
        FROM models 
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    models_dict = {}
    latest_model = None
    for row in rows:
        v_id = row["version_id"]
        if latest_model is None and row["is_champion"] == 1:
            latest_model = v_id
        models_dict[v_id] = {
            "class_name": row["class_name"],
            "model_path": row["model_path"],
            "hyperparameters": json.loads(row["hyperparameters"]) if row["hyperparameters"] else {},
            "training_mse": row["training_mse"],
            "lineage": {
                "git_commit": row["git_commit"],
                "dataset_checksum": row["dataset_checksum"],
                "training_samples": row["training_samples"]
            },
            "updated_at": row["created_at"]
        }
        
    if not latest_model and models_dict:
        latest_model = list(models_dict.keys())[0]
        
    return {
        "registry_version": "1.0",
        "latest_model": latest_model or "",
        "models": models_dict
    }

def log_benchmark_record(stats):
    """Logs a tournament benchmark summary record into the benchmarks table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    pair = stats.get("pair", [])
    agent_a = pair[0] if len(pair) > 0 else ""
    agent_b = pair[1] if len(pair) > 1 else ""
    
    wins_a = stats.get(agent_a, {}).get("wins", 0)
    wins_b = stats.get(agent_b, {}).get("wins", 0)
    ties = stats.get(agent_a, {}).get("ties", 0)
    total_games = stats.get("total_games", 0)
    fingerprints = json.dumps(stats.get("fingerprints", {}))
    
    cursor.execute("""
        INSERT INTO benchmarks (timestamp, agent_a, agent_b, agent_a_wins, agent_b_wins, ties, total_games, fingerprints)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        stats.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
        agent_a,
        agent_b,
        wins_a,
        wins_b,
        ties,
        total_games,
        fingerprints
    ))
    conn.commit()
    conn.close()

def fetch_all_benchmarks():
    """Retrieves all historical benchmark records from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, agent_a, agent_b, agent_a_wins, agent_b_wins, ties, total_games, fingerprints FROM benchmarks")
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        agent_a = row["agent_a"]
        agent_b = row["agent_b"]
        fingerprints = json.loads(row["fingerprints"]) if row["fingerprints"] else {}
        history.append({
            "pair": [agent_a, agent_b],
            "fingerprints": fingerprints,
            agent_a: {"wins": row["agent_a_wins"], "losses": row["agent_b_wins"], "ties": row["ties"]},
            agent_b: {"wins": row["agent_b_wins"], "losses": row["agent_a_wins"], "ties": row["ties"]},
            "total_games": row["total_games"],
            "timestamp": row["timestamp"]
        })
    return history

def register_model_record(version_id, model_path, class_name, hyperparameters, training_mse, lineage, is_champion=0):
    """Inserts a new model version and its lineage into the SQLite models table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if is_champion:
        cursor.execute("UPDATE models SET is_champion = 0")
        
    cursor.execute("""
        INSERT INTO models (
            version_id, model_path, class_name, hyperparameters, 
            training_mse, git_commit, dataset_checksum, training_samples, 
            is_champion, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        version_id,
        model_path,
        class_name,
        json.dumps(hyperparameters),
        training_mse,
        lineage.get("git_commit"),
        lineage.get("dataset_checksum"),
        lineage.get("training_samples"),
        1 if is_champion else 0,
        time.strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def log_training_run_record(run_record):
    """Logs execution telemetry of a training session to the training_runs table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO training_runs (
            timestamp, version_id, historical_games_reused, 
            new_games_generated, total_samples, training_duration_seconds, training_mse
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        run_record.get("timestamp"),
        run_record.get("version_id"),
        run_record.get("historical_games_reused"),
        run_record.get("new_games_generated"),
        run_record.get("total_samples"),
        run_record.get("training_duration_seconds"),
        run_record.get("training_mse")
    ))
    conn.commit()
    conn.close()