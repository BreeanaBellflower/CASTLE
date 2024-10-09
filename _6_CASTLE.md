**CASTLE** is an advanced chess engine that leverages a transformer-based model to predict and execute chess moves. Designed for both simulation and interactive play, CASTLE integrates sophisticated machine learning techniques with traditional chess strategies to provide a versatile and intelligent chess-playing experience.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
  - [Model Integration](#model-integration)
  - [Chassis Design](#chassis-design)
- [Installation](#installation)
- [Usage](#usage)
  - [Simulating Games from a Dataset](#simulating-games-from-a-dataset)
  - [Interactive Play (Human vs CASTLE)](#interactive-play-human-vs-castle)
  - [Self-Play Mode](#self-play-mode)
- [Configuration](#configuration)
- [Monte Carlo Tree Search (MCTS)](#monte-carlo-tree-search-mcts)
- [Move History Management](#move-history-management)
- [Examples](#examples)

## Overview

CASTLE is a state-of-the-art chess engine designed to predict and execute chess moves using a transformer-based neural network. By integrating historical move sequences and current board states, CASTLE can simulate professional-level chess play, respond intelligently to human opponents, and engage in self-play to refine its strategies.

## Features

- **Transformer-Based Move Prediction:** Utilizes a trained transformer model to predict the most probable and strategic chess moves.
- **Multiple Interfaces:** Supports simulation from datasets, interactive play against humans, and self-play between model instances.
- **Standard Opening Patterns:** Implements standard opening moves when initializing new games or responding to early game scenarios.
- **Monte Carlo Tree Search (MCTS):** Enhances decision-making by exploring potential move sequences and evaluating their outcomes.
- **Self-Play Capability:** Allows two instances of CASTLE to compete against each other, facilitating continuous learning and strategy improvement.
- **Move History Management:** Maintains a history of the most recent 20 moves to inform move predictions and MCTS operations.
- **Flexible Input Handling:** Can process complete game datasets or manage real-time interactions during gameplay.

## Architecture

### Model Integration

CASTLE integrates a pre-trained transformer model that processes sequences of chess moves and board states to predict optimal subsequent moves. The model considers both temporal dependencies (move sequences) and spatial relationships (board states) to make informed predictions.

### Chassis Design

The chassis serves as the core framework that interfaces with the transformer model. It provides various functionalities, including:

- **Data Handling:** Processes input files containing game data or manages real-time game states.
- **Move Prediction:** Utilizes the transformer model to predict the best possible moves based on current game conditions.
- **MCTS Integration:** Incorporates MCTS to explore and evaluate potential move sequences, enhancing decision-making.
- **Interface Management:** Facilitates different modes of interaction, such as simulations, human play, and self-play.

## Installation

### Prerequisites

- **Operating System:** Windows, macOS, or Linux
- **Python:** 3.8 or higher
- **CUDA (Optional):** For GPU acceleration

### Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```plaintext
python-chess
torch
numpy
pandas
tqdm
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/CASTLE.git
cd CASTLE
```

### Setup

Ensure that the trained transformer model and move index mappings are placed in the appropriate directories:

- **Model Weights:** `models/transformer_model.pt`
- **Move Indices:** `rules_based_mapping/chess_move_indices.json` and `rules_based_mapping/all_legal_moves.json`

## Usage

CASTLE provides multiple modes of operation to cater to different use cases. Below are the primary functionalities and instructions on how to utilize them.

### Simulating Games from a Dataset

CASTLE can process a dataset of chess games to simulate and validate its performance.

**Command:**

```bash
python simulate_games.py --dataset_path dataset/split/castle_dataset_split_{n}.txt
```

**Parameters:**

- `--dataset_path`: Path to the dataset file containing game sequences.

**Example:**

```bash
python simulate_games.py --dataset_path dataset/split/castle_dataset_split_1.txt
```

### Interactive Play (Human vs CASTLE)

Engage in a live chess game against CASTLE. The engine will handle both sides, allowing you to play as either white or black.

**Command:**

```bash
python interactive_play.py --color white
```

**Parameters:**

- `--color`: Choose to play as `white` or `black`. If you choose `black`, CASTLE will make the first move.

**Example:**

```bash
python interactive_play.py --color black
```

### Self-Play Mode

Enable CASTLE to play games against itself, facilitating self-improvement and strategy refinement.

**Command:**

```bash
python self_play.py --num_games 100
```

**Parameters:**

- `--num_games`: Number of self-play games to run concurrently.

**Example:**

```bash
python self_play.py --num_games 50
```

## Configuration

CASTLE offers various configuration options to tailor its behavior according to specific requirements.

**Configuration File:**

Edit the `config.yaml` file to adjust settings such as:

- **Batch Size**
- **Number of MCTS Simulations**
- **Transformer Model Path**
- **Logging Preferences**

**Example `config.yaml`:**

```yaml
model:
  path: models/transformer_model.pt
  device: cuda

mcts:
  simulations: 1000
  exploration_constant: 1.4

game:
  max_move_history: 20

logging:
  level: INFO
  file: logs/castle.log
```

## Monte Carlo Tree Search (MCTS)

CASTLE integrates MCTS to enhance its decision-making process by exploring potential move sequences and evaluating their outcomes.

### How It Works

1. **Move Exploration:** MCTS explores various move sequences from the current game state.
2. **Simulation:** For each potential move, MCTS simulates future moves up to a specified depth.
3. **Evaluation:** The transformer model evaluates the resulting board states to assess the quality of each move.
4. **Selection:** The best move is selected based on the evaluations and exploration-exploitation balance.

### Configuration

Adjust MCTS parameters in the `config.yaml` file to control the number of simulations and exploration behavior.

**Example:**

```yaml
mcts:
  simulations: 500
  exploration_constant: 1.0
```

## Move History Management

To maintain context and inform move predictions, CASTLE keeps track of the most recent 20 moves in each game.

### How It Works

- **Move Sequence:** CASTLE maintains a sliding window of the last 20 moves, ensuring that the model has sufficient context without overwhelming memory.
- **Integration with MCTS:** During MCTS, each simulated move sequence respects the 20-move limit, ensuring consistency and efficiency.

### Configuration

Adjust the move history length in the `config.yaml` file if needed.

**Example:**

```yaml
game:
  max_move_history: 20
```

## Examples

### Simulating a Single Game from a Dataset

```bash
python simulate_games.py --dataset_path dataset/split/castle_dataset_split_1.txt
```

### Playing Against CASTLE as Black

```bash
python interactive_play.py --color black
```

### Running 10 Self-Play Games

```bash
python self_play.py --num_games 10
```
