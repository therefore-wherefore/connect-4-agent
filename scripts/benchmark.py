import os
import sys
import time
import hashlib
import numpy as np

if (PROJECT_ROOT := os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) not in sys.path:    sys.path.insert(0, PROJECT_ROOT)

from environment.board import create_board, get_valid_locations, drop_piece, is_terminal_node, winning_move
from agents.uniform import UniformDistributionBot
from agents.normal_dist import NormalDistributionBot
from agents.minimax_bot import MiniMaxBot
from agents.rf_bot import RandomForestMinimaxBot
from agents.xgb_bot import XGBMinimaxBot
from agents.neuro_bot import NeuroMinimaxBot
from persistence.database import init_db, log_game_record, fetch_model_registry_from_db, log_benchmark_record, fetch_all_benchmarks

AGENT_REGISTRY_MAP = {
    "random_bot":   lambda player_id, model_version=None: UniformDistributionBot(player_id=player_id),
    "normal_bot":   lambda player_id, model_version=None: NormalDistributionBot(player_id=player_id, sigma=1.2),
    "minimax_bot":  lambda player_id, model_version=None: MiniMaxBot(player_id=player_id, depth=3),
    "xgb_bot":      lambda player_id, model_version=None: XGBMinimaxBot(player_id=player_id, depth=3, model_version=model_version),
    "rf_bot":       lambda player_id, model_version=None: RandomForestMinimaxBot(player_id=player_id, model_version=model_version),
    "neuro_bot":    lambda player_id, model_version=None: NeuroMinimaxBot(player_id=player_id, model_version=model_version),
}

def verify_model_exists(agent_key, model_version):
    """Pre-flight check: ensures the registered model file actually exists on disk before benchmarking."""
    if not model_version or agent_key in ["random_bot", "normal_bot", "minimax_bot"]:
        return True
        
    registry = fetch_model_registry_from_db()
    models_dict = registry.get("models", {})
    
    target_key = model_version if model_version in models_dict else registry.get("latest_model")
    if target_key and target_key in models_dict:
        model_path = models_dict[target_key].get("model_path")
        if model_path:
            full_path = model_path if os.path.isabs(model_path) else os.path.join(PROJECT_ROOT, model_path)
            return os.path.exists(full_path)
            
    return False

def get_agent_fingerprint(agent_key, model_version=None):
    """Computes an MD5 checksum of the agent's source code and its specific active model weights."""
    hasher = hashlib.md5()
    
    file_map = {
        "random_bot": "agents/uniform.py",
        "normal_bot": "agents/normal_dist.py",
        "minimax_bot": "agents/minimax_bot.py",
        "xgb_bot": "agents/xgb_bot.py",
        "rf_bot": "agents/rf_bot.py",
        "neuro_bot": "agents/neuro_bot.py"
    }
    
    target_file = file_map.get(agent_key, "")
    if os.path.exists(target_file):
        with open(target_file, "rb") as f:
            hasher.update(f.read())
            
    if agent_key in ["xgb_bot", "neuro_bot", "rf_bot"]:
        registry = fetch_model_registry_from_db()
        models_dict = registry.get("models", {})
        
        target_key = model_version if model_version else registry.get("latest_model")
        model_path = None
        
        if target_key and target_key in models_dict:
            model_path = models_dict[target_key].get("model_path")
            
        if model_path and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                hasher.update(f.read())
                
    return hasher.hexdigest()

def already_tested_with_current_fingerprints(agent_a, agent_b, history, ver_a=None, ver_b=None):
    """Skips testing if both agents' specific versions and fingerprints match a previous entry."""
    fp_a = get_agent_fingerprint(agent_a, ver_a)
    fp_b = get_agent_fingerprint(agent_b, ver_b)
    
    name_a = f"{agent_a}:{ver_a}" if ver_a else agent_a
    name_b = f"{agent_b}:{ver_b}" if ver_b else agent_b
    
    for entry in history:
        pair = entry.get("pair", [])
        fingerprints = entry.get("fingerprints", {})
        if sorted(pair) == sorted([name_a, name_b]):
            if fingerprints.get(name_a) == fp_a and fingerprints.get(name_b) == fp_b:
                return True
    return False

def instantiate_agent(p_type, player_id, model_version=None):
    if p_type not in AGENT_REGISTRY_MAP:
        raise ValueError(f"Unknown agent type: {p_type}")
    return AGENT_REGISTRY_MAP[p_type](player_id=player_id, model_version=model_version)

def play_match_with_telemetry(agent_p0, agent_p1):
    state = create_board()
    turn = 0
    agents = {0: agent_p0, 1: agent_p1}
    moves_history = []
    
    while not is_terminal_node(state):
        valid_moves = get_valid_locations(state)
        if not valid_moves: break
            
        current_agent = agents[turn]
        col = current_agent.get_move(state)
        
        if col is None or col not in valid_moves:   col = np.random.choice(valid_moves)
            
        moves_history.append(int(col))
        state = drop_piece(state, col, turn)
        
        if winning_move(state, turn):   return turn, moves_history
        elif is_terminal_node(state):   return -1, moves_history
            
        turn = 1 - turn
    return -1, moves_history

def run_smart_benchmark(agent_a_key, agent_b_key, total_games=40, ver_a=None, ver_b=None):
    
    if not verify_model_exists(agent_a_key, ver_a): return print(f"\n[Error] Skipping benchmark: Model weights for '{agent_a_key}:{ver_a}' not found on disk.")
    if not verify_model_exists(agent_b_key, ver_b): return print(f"\n[Error] Skipping benchmark: Model weights for '{agent_b_key}:{ver_b}' not found on disk.")

    history = fetch_all_benchmarks()

    label_a = f"{agent_a_key}:{ver_a}" if ver_a else agent_a_key
    label_b = f"{agent_b_key}:{ver_b}" if ver_b else agent_b_key

    if already_tested_with_current_fingerprints(agent_a_key, agent_b_key, history, ver_a, ver_b):
        print(f"\n[Cache Hit] Skipping benchmark: '{label_a}' vs '{label_b}' have no changes.")
        for entry in history:
            if sorted(entry.get("pair", [])) == sorted([label_a, label_b]):
                return entry
        return None

    print(f"\n[MLOps Benchmark] Running evaluation: {label_a} vs {label_b} ({total_games} games)...")
    
    stats = {
        "pair": [label_a, label_b],
        "fingerprints": {
            label_a: get_agent_fingerprint(agent_a_key, ver_a),
            label_b: get_agent_fingerprint(agent_b_key, ver_b)
        },
        label_a: {"wins": 0, "losses": 0, "ties": 0},
        label_b: {"wins": 0, "losses": 0, "ties": 0},
        "total_games": total_games,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    half_games = total_games // 2
    game_lengths = []

    for _ in range(half_games):
        a_bot = instantiate_agent(agent_a_key, 0, model_version=ver_a)
        b_bot = instantiate_agent(agent_b_key, 1, model_version=ver_b)
        winner, moves = play_match_with_telemetry(a_bot, b_bot)
        
        game_lengths.append(len(moves))
        if winner == 0:
            stats[label_a]["wins"] += 1
            stats[label_b]["losses"] += 1
            winning_agent = label_a
        elif winner == 1:
            stats[label_b]["wins"] += 1
            stats[label_a]["losses"] += 1
            winning_agent = label_b
        else:
            stats[label_a]["ties"] += 1
            stats[label_b]["ties"] += 1
            winning_agent = "tie"
            
        log_game_record({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "game_type": "benchmark",
            "player_0": label_a,
            "player_1": label_b,
            "winner": winning_agent,
            "total_moves": len(moves),
            "search_depth": 3,
            "moves": moves
        })

    for _ in range(half_games):
        b_bot = instantiate_agent(agent_b_key, 0, model_version=ver_b)
        a_bot = instantiate_agent(agent_a_key, 1, model_version=ver_a)
        winner, moves = play_match_with_telemetry(b_bot, a_bot)
        
        game_lengths.append(len(moves))
        if winner == 0:
            stats[label_b]["wins"] += 1
            stats[label_a]["losses"] += 1
            winning_agent = label_b
        elif winner == 1:
            stats[label_a]["wins"] += 1
            stats[label_b]["losses"] += 1
            winning_agent = label_a
        else:
            stats[label_a]["ties"] += 1
            stats[label_b]["ties"] += 1
            winning_agent = "tie"
            
        log_game_record({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "game_type": "benchmark",
            "player_0": label_b,
            "player_1": label_a,
            "winner": winning_agent,
            "total_moves": len(moves),
            "search_depth": 3,
            "moves": moves
        })

    avg_len = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    print("\n=== BENCHMARK SUMMARY ===")
    for agent_label in [label_a, label_b]:
        wins = stats[agent_label]["wins"]
        losses = stats[agent_label]["losses"]
        ties = stats[agent_label]["ties"]
        win_rate = (wins / total_games) * 100
        print(f"Agent: {agent_label}")
        print(f"  Wins: {wins} | Losses: {losses} | Ties: {ties} | Win Rate: {win_rate:.1f}%")
    print(f"  Average Match Length: {avg_len:.1f} moves")

    log_benchmark_record(stats)
        
    return stats

if __name__ == "__main__":
    print("=== MLOps Automated Model Tournament Runner ===")
    
    init_db()
    
    registry = fetch_model_registry_from_db()
    models_dict = registry.get("models", {})
    registered_models = list(models_dict.keys())
    
    if not registered_models:
        print("No registered models found in SQLite database. Running default latest vs minimax.")
        run_smart_benchmark("xgb_bot", "minimax_bot", total_games=40)
        sys.exit(0)

    print(f"Found {len(registered_models)} registered model checkpoints to evaluate against Minimax baseline.")

    baseline_results = []
    for version_id in registered_models:
        model_info = models_dict.get(version_id, {})
        class_name = model_info.get("class_name", "")
        
        if  "RandomForest" in class_name or "rf_" in version_id:
            agent_key = "rf_bot"
        elif ("Connect4Net" in class_name 
            or "rl_"        in version_id 
            or "neuro_"     in class_name):  
            agent_key = "neuro_bot"
        else:
            agent_key = "xgb_bot"
        
        stats = run_smart_benchmark(agent_key, "minimax_bot", total_games=40, ver_a=version_id, ver_b=None)
        if stats:
            label = f"{agent_key}:{version_id}"
            wins = stats.get(label, {}).get("wins", 0)
            win_rate = (wins / 40.0) * 100
            baseline_results.append((version_id, agent_key, win_rate))

    baseline_results.sort(key=lambda x: x[2], reverse=True)
    
    print("\n" + "="*45)
    print("      MINIMAX SCREENING LEADERBOARD      ")
    print("="*45)
    for rank, (vid, a_key, wr) in enumerate(baseline_results, 1):
        print(f"  {rank}. [{a_key}] {vid} -> Win Rate vs Minimax: {wr:.1f}%")
    print("="*45)

    top_3 = baseline_results[:3]
    top_3_versions = [(vid, a_key) for vid, a_key, wr in top_3]
    print(f"\nTop {len(top_3_versions)} models selected for the Champion Round-Robin Tournament.")

    if len(top_3_versions) > 1:
        print("\n=== STARTING TOP-3 ROUND-ROBIN TOURNAMENT ===")
        for i in range(len(top_3_versions)):
            for j in range(i + 1, len(top_3_versions)):
                v_a, key_a = top_3_versions[i]
                v_b, key_b = top_3_versions[j]
                run_smart_benchmark(key_a, key_b, total_games=40, ver_a=v_a, ver_b=v_b)
    else:
        print("\nOnly 1 model available in SQLite. Skipping round-robin tournament.")