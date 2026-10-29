import os
import json
from collections import defaultdict, Counter

DB_PATH = os.path.join("logs", "game_database.jsonl")

def analyze_database():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        print("Run a training or benchmark session first to generate telemetry.")
        return

    total_games = 0
    game_types = Counter()
    matchups = Counter()
    winner_breakdown = Counter()
    depth_stats = defaultdict(lambda: {"total": 0, "p0_wins": 0, "p1_wins": 0, "ties": 0})
    match_lengths = []

    with open(DB_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                total_games += 1
                
                # 1. Game Type Tracking
                g_type = record.get("game_type", "unknown")
                game_types[g_type] += 1
                
                # 2. Matchup Tracking
                p0 = record.get("player_0", "unknown")
                p1 = record.get("player_1", "unknown")
                matchup = f"{p0} (P0) vs {p1} (P1)"
                matchups[matchup] += 1
                
                # 3. Winner Breakdown per Matchup
                winner = record.get("winner", "unknown")
                outcome_key = f"[{matchup}] Winner: {winner}"
                winner_breakdown[outcome_key] += 1
                
                # 4. Search Depth Performance (if logged)
                depth = record.get("search_depth")
                if depth is not None:
                    depth_stats[depth]["total"] += 1
                    if winner == p0:
                        depth_stats[depth]["p0_wins"] += 1
                    elif winner == p1:
                        depth_stats[depth]["p1_wins"] += 1
                    else:
                        depth_stats[depth]["ties"] += 1
                        
                # 5. Match Lengths
                total_moves = record.get("total_moves")
                if total_moves is not None:
                    match_lengths.append(total_moves)
                    
            except json.JSONDecodeError:
                continue

    # --- Print Structured Diagnostic Report ---
    print("\n" + "="*40)
    print("      GAME DATABASE DIAGNOSTIC REPORT      ")
    print("="*40)
    print(f"Total Games Analyzed: {total_games}\n")

    print("--- 1. Breakdown by Game Type ---")
    for g_type, count in game_types.items():
        print(f"  • {g_type}: {count} games ({count / total_games * 100:.1f}%)")
    print()

    print("--- 2. Head-to-Head Matchups ---")
    for matchup, count in matchups.items():
        print(f"  • {matchup}: {count} games")
    print()

    print("--- 3. Detailed Outcome Distribution ---")
    for outcome, count in winner_breakdown.items():
        print(f"  • {outcome}: {count} games")
    print()

    if depth_stats:
        print("--- 4. Performance by P0 Search Depth ---")
        for depth in sorted(depth_stats.keys()):
            stats = depth_stats[depth]
            tot = stats["total"]
            p0_w = stats["p0_wins"]
            p1_w = stats["p1_wins"]
            ties = stats["ties"]
            print(f"  • Search Depth {depth} ({tot} games):")
            print(f"    - P0 Wins: {p0_w} ({p0_w/tot*100:.1f}%)")
            print(f"    - P1 Wins: {p1_w} ({p1_w/tot*100:.1f}%)")
            print(f"    - Ties: {ties} ({ties/tot*100:.1f}%)")
        print()

    if match_lengths:
        avg_len = sum(match_lengths) / len(match_lengths)
        print("--- 5. Match Length Statistics ---")
        print(f"  • Average Length: {avg_len:.1f} moves")
        print(f"  • Shortest Game: {min(match_lengths)} moves")
        print(f"  • Longest Game: {max(match_lengths)} moves")
    print("="*40 + "\n")

if __name__ == "__main__":
    analyze_database()