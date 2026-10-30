# Connect-4 MLOps Pipeline

An end-to-end Machine Learning system combining Monte Carlo Tree Search (MCTS), Evolutionary Neural Architecture Search (NAS), and Gradient Boosted Decision Trees (XGBoost) within a production-ready MLOps framework. 

Built with modular architecture, strict data lineage tracking, and persistent model registry controls to optimise decision-making agents in zero-sum environments.

---

## Highlights

This project demonstrates core competencies required for high-throughput, product-driven AI engineering:

*   **Production MLOps & Lineage Tracking:** Enforces database persistence, Git commit hashing, data checksums, and model versioning via SQLite and joblib/JSON metadata registries.
*   **Deep Learning & Reinforcement Learning:** Combines Monte-Carlo Tree Search self-play data generation with dual-head neural network architecture (Policy + Value heads) written in **PyTorch**.
*   **Automated Machine Learning (AutoML):** Implements evolutionary algorithms for dynamic Neural Architecture Search (NAS) to discover optimal model hidden-layer topologies automatically.
*   **High-Performance Supervised ML:** Leverages XGBoost and Random Forest regression models aligned with game-tree heuristics (Minimax with $\alpha$-$\beta$ pruning).
*   **Clean Code Standards & System Design:** Fully modularised codebase with clean separation of environment, persistence layer, agent abstraction, scripts, and utilities.

---

## Architecture Overview

```
├── agents/            # Strategy implementations (Minimax, XGBoost, RL Neural Bots, Random)
├── environment/       # Core Connect4 state machine & win condition evaluation
├── models/            # Serialised model artifacts (.joblib, .json) & registry tracking
├── persistence/       # SQLite schema definitions, database initialisation, and cleanup scripts
├── scripts/           # Execution pipelines (MCTS/NAS, XGBoost training, benchmarking)
└── utils/             # Data processing, feature extraction, and MLOps tooling
```

---

## Core AI & ML Components

### 1. Evolutionary NAS & MCTS Self-Play (`scripts/mcts_pipeline.py`)
*   **Neural Architecture Search (NAS):** Dynamic genome mutation (altering network width and depth dynamically across generations) evaluating structural efficiency before full-scale training.
*   **Dual-Head PyTorch Model (`Connect4Net`):** Shares underlying feature representation into:
    *   **Policy Head:** Outputs categorical action probabilities over valid columns ($\text{Softmax}$).
    *   **Value Head:** Predicts expected game outcome ($[-1, 1]$ via $\text{Tanh}$).
*   **Monte Carlo Tree Search (MCTS):** Uses Upper Confidence Bounds applied to Trees (PUCT formula) for self-play dataset generation.
*   **Replay Buffer & Batch Optimisation:** Custom ring buffer storing state-action-reward triplets for optimised mini-batch gradient updates using MSE and Cross-Entropy loss combinations:
    $$\mathcal{L} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}}$$

### 2. Gradient Boosting Pipeline (`scripts/train_xgb.py`)
*   **Supervised Heuristic Learning:** Trains XGBoost regression models on board evaluations derived from game states stored in SQLite databases.
*   **MLOps Governance (`utils/mlops.py`):**
    *   Automatic Git cleanliness verification prior to training runs.
    *   Cryptographic hash checks (`get_data_checksum`) on training data arrays to ensure data reproducibility.
    *   Automatic SQLite table updates tracking training duration, mean squared error ($\text{MSE}$), and active champion model flags.

---

## Quickstart

### Prerequisites
* Python 3.10+
* PyTorch, XGBoost, Scikit-Learn, SQLite3

### Installation

```bash
git clone https://github.com/your-username/connect4-ai-mlops.git
cd connect4-ai-mlops
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Pipelines

1. **Train XGBoost Agent with Full MLOps Telemetry:**
   ```bash
   python scripts/train_xgb.py
   ```

2. **Execute Evolutionary NAS & MCTS Self-Play Training:**
   ```bash
   python scripts/mcts_pipeline.py
   ```

---

## Database & Model Lineage Schema

Model metadata and training telemetry are logged centrally to ensure transparency, reproducibility, and roll-back capability across training runs:

| Field | Description |
| :--- | :--- |
| `version_id` | Unique timestamped model identifier |
| `git_commit` | Git hash associated with the precise code state |
| `dataset_checksum` | Hash of features ($X$) and labels ($y$) used during fitting |
| `hyperparameters` | Dynamic JSON configuration of hyperparameter space |
| `is_champion` | Flag indicating currently active production model |

---

## Future Roadmap

- [ ] Implement an open-source evaluation suite benchmarking against Minimax depth baselines.
- [ ] Add FastAPI endpoints to serve policy predictions with sub-10ms response targets.
- [ ] Containerise training environment with Docker for cloud deployment.