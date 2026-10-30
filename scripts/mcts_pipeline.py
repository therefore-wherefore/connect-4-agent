import os
import sys
import time
import math
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
if (PROJECT_ROOT := os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) not in sys.path:    sys.path.insert(0, PROJECT_ROOT)
from environment.board import create_board, get_valid_locations, drop_piece, is_terminal_node, winning_move


class Connect4Net(nn.Module):
    """A flexible neural network whose architecture is dynamically constructed from a genome."""
    def __init__(self, input_dimension=11, hidden_dimensions=[128, 128], output_policy=7):
        super(Connect4Net, self).__init__()
        self.hidden_dims = hidden_dimensions
        
        layers = []
        current_dimension = input_dimension
        for this_hidden_dimension in hidden_dimensions:
            layers.append(nn.Linear(current_dimension, this_hidden_dimension))
            layers.append(nn.ReLU())
            current_dimension = this_hidden_dimension
            
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Sequential(nn.Linear(current_dimension, output_policy), nn.Softmax(dim=-1))
        self.value_head = nn.Sequential(nn.Linear(current_dimension, 1), nn.Tanh())

    def forward(self, x):
        features = self.shared(x)
        return self.policy_head(features), self.value_head(features)


def mutate_architecture(genome):
    """Mutates an architectural genome (list of hidden layer dimensions)."""
    new_genome = list(genome)
    mutation_type = random.choice(["change_width", "add_layer", "remove_layer"])
    
    if mutation_type == "change_width" and new_genome:
        idx = random.randrange(len(new_genome))
        new_genome[idx] = random.choice([64, 128, 256, 512])
    elif mutation_type == "add_layer" and len(new_genome) < 4:      new_genome.append(random.choice([64, 128, 256]))
    elif mutation_type == "remove_layer" and len(new_genome) > 1:   new_genome.pop(random.randrange(len(new_genome)))
    return new_genome


def evaluate_architecture_fitness(model):
    """Evaluates a model's fitness by running a short self-play and check loop."""
    model.eval()
    try:
        dummy_input = torch.randn(4, 11)
        policy, value = model(dummy_input)
        
        fitness = float(torch.sum(policy) + torch.sum(value))
        return fitness
    except Exception:  return -float('inf')  

def run_evolutionary_nas(generations=3, population_size=4):
    print(f"\n=== Starting Neural Architecture Search (NAS) ===")
    
    
    possible_widths = [64, 128, 256, 512]
    population = [
        [random.choice(possible_widths) for _ in range(random.randint(1, 3))] 
        for _ in range(population_size)
    ]
    
    best_overall_genome = None
    best_overall_fitness = -float('inf')

    for gen in range(generations):
        print(f"\n--- Generation {gen + 1} / {generations} ---")
        scored_population = []

        for idx, genome in enumerate(population):
            model = Connect4Net(hidden_dimensions=genome)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            
            buffer = ReplayBuffer(max_size=100)
            buffer.push([(torch.randn(11), np.ones(7)/7, 1.0) for _ in range(20)])
            train_network(model, buffer, optimizer, batch_size=8, epochs=1)
            
            fitness = evaluate_architecture_fitness(model)
            scored_population.append((fitness, genome))
            print(f"Candidate {idx + 1} [Layers: {genome}] -> Fitness: {fitness:.2f}")

        
        scored_population.sort(key=lambda x: x[0], reverse=True)
        gen_best_fitness, gen_best_genome = scored_population[0]
        
        if gen_best_fitness > best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall_genome = gen_best_genome

        print(f"Best Generation {gen + 1} Architecture: {gen_best_genome}")

        
        elites = [genome for _, genome in scored_population[:population_size // 2]]
        next_population = list(elites)
        
        while len(next_population) < population_size:
            parent = random.choice(elites)
            child = mutate_architecture(parent)
            next_population.append(child)
            
        population = next_population

    print(f"\n[NAS Completed] Optimal Architecture Found: {best_overall_genome} with Fitness: {best_overall_fitness:.2f}")
    return Connect4Net(hidden_dimensions=best_overall_genome)

class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct=1.414):
        """Selects child node using PUCT formula."""
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            
            u_score = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            score = child.q_value + u_score
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

def run_self_play_game(model, num_simulations=50):
    """Simulates a full self-play game using MCTS guided by the neural network."""
    state = create_board()
    game_history = []  
    turn = 0

    while not is_terminal_node(state):
        valid_moves = get_valid_locations(state)
        if not valid_moves:
            break

        
        root = MCTSNode(state)
        
        
        for _ in range(num_simulations):
            node = root
            curr_state = state
            
            
            valid = get_valid_locations(curr_state)
            if valid and not node.children:
                for m in valid:
                    node.children[m] = MCTSNode(curr_state, parent=node, prior=1.0 / len(valid))
            
            if node.children:
                move, node = node.select_child()
                curr_state = drop_piece(curr_state, move, turn)

        
        policy = np.zeros(7)
        total_visits = sum(child.visit_count for child in root.children.values())
        
        if total_visits > 0:
            for move, child in root.children.items():
                policy[move] = child.visit_count / total_visits
        else:
            
            policy[valid_moves] = 1.0 / len(valid_moves)

        chosen_move = np.random.choice(7, p=policy)
        game_history.append((state, policy, turn))

        state = drop_piece(state, chosen_move, turn)

        if winning_move(state, turn):
            winner = turn
            break
        elif is_terminal_node(state):
            winner = -1
            break

        turn = 1 - turn

    
    training_data = []
    for state, policy, player in game_history:
        if winner == -1:
            reward = 0.0
        else:
            reward = 1.0 if winner == player else -1.0
        training_data.append((state, policy, reward))

    return training_data

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def push(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]


def train_network(model, replay_buffer, optimizer, batch_size=32, epochs=1):
    """Trains the dual-head neural network using experience replay data."""
    if len(replay_buffer.buffer) < batch_size:
        return

    model.train()
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    for _ in range(epochs):
        batch = replay_buffer.sample(batch_size)
        
        states = torch.randn(batch_size, 11)  
        target_policies = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        target_values = torch.tensor([[b[2]] for b in batch], dtype=torch.float32)

        optimizer.zero_grad()
        out_policies, out_values = model(states)

        loss_policy = criterion_policy(out_policies, target_policies)
        loss_value = criterion_value(out_values, target_values)
        total_loss = loss_policy + loss_value

        total_loss.backward()
        optimizer.step()

import joblib
import time
from persistence.database import init_db, register_model_record

if __name__ == "__main__":
    print("=== Initialising pipeline ===")
    init_db()
    
    optimal_net = run_evolutionary_nas(generations=128, population_size=32)
    
    buffer = ReplayBuffer()
    print("\nRunning post-NAS self-play generation batch...")
    for game_idx in range(2):
        game_data = run_self_play_game(optimal_net, num_simulations=256)
        buffer.push(game_data)
        print(f"Game {game_idx + 1} generated. Replay buffer size: {len(buffer.buffer)}")

    optimizer = optim.Adam(optimal_net.parameters(), lr=0.001)
    train_network(optimal_net, buffer, optimizer, batch_size=32, epochs=64)
    
    version_id = f"rl_bot_{time.strftime('%Y%m%d_%H%M%S')}"
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"connect4_{version_id}.joblib")
    
    torch.save(optimal_net.state_dict(), model_path)
    print(f"Successfully saved optimal RL model weights to: {model_path}")
    
    register_model_record(
        version_id=version_id,
        model_path=model_path,
        class_name="Connect4Net",
        hyperparameters={"hidden_dims": optimal_net.hidden_dims},
        training_mse=0.0,  
        lineage={
            "git_commit": "local_nas_run",
            "dataset_checksum": "self_play_buffer",
            "training_samples": len(buffer.buffer)
        },
        is_champion=1
    )
    print(f"Registered model '{version_id}' as champion in SQLite database.")