# castle_engine.py

import os
import json
import random
import chess
import torch
import numpy as np
from collections import deque
from torch.nn.functional import softmax
from typing import List, Optional, Tuple

# Import the model definition if it's in a separate module
from model.ChessMovePredictionModel import ChessMovePredictionModel

# Constants
MAX_MOVE_HISTORY = 19
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CastleEngine:
    def __init__(self, model_path: str, legal_moves_path: str, config: dict):
        # Load the trained transformer model
        self.model = self.load_model(model_path)
        self.model.to(DEVICE)
        self.model.eval()

        # Load the legal moves mapping
        with open(legal_moves_path, 'r') as f:
            self.legal_moves = json.load(f)
            self.move_to_index = {move: idx + 1 for idx, move in enumerate(self.legal_moves)}
            self.index_to_move = {idx + 1: move for idx, move in enumerate(self.legal_moves)}
            self.move_vocab_size = len(self.legal_moves) + 1  # +1 for padding index 0

        # Configuration
        self.config = config
        self.max_move_history = config.get('game', {}).get('max_move_history', MAX_MOVE_HISTORY)
        self.choices = config.get('search', {}).get('choices', 5)  # Number of top moves to consider at each node
        self.depth = config.get('search', {}).get('depth', 3)      # Depth of exploration

        # Move history
        self.move_history = deque(maxlen=self.max_move_history)
        self.board = chess.Board()

        # Opening book (simple implementation)
        self.opening_moves = self.load_opening_book()

        # Random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def load_model(self, model_path: str):
        # Assuming the model class is defined elsewhere
        model = ChessMovePredictionModel()
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def load_opening_book(self):
        # Placeholder for loading opening moves
        # In a real implementation, load from an opening book file
        return {
            'white': ['e2e4', 'd2d4', 'c2c4', 'g1f3'],
            'black': ['e7e5', 'c7c5', 'e7e6', 'g8f6']
        }

    def reset_game(self):
        self.move_history.clear()
        self.board.reset()

    def make_move(self, move: str):
        # Update the board and move history
        self.board.push_uci(move)
        self.move_history.append(self.move_to_index.get(move, 0))  # Use 0 if move not found

    def get_board_state(self, board=None):
        # Convert the board to a tensor representation
        if board is None:
            board = self.board
        piece_map = board.piece_map()
        board_state = np.zeros(64, dtype=int)
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                # Map piece symbol to integer
                symbol = piece.symbol()
                board_state[square] = self.piece_to_int(symbol)
            else:
                board_state[square] = 0  # Empty square
        return torch.tensor(board_state + 6, dtype=torch.long).to(DEVICE)  # Shift to 0..12

    def piece_to_int(self, symbol: str):
        # Map piece symbol to integer (-6 to 6)
        mapping = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
        }
        return mapping.get(symbol, 0)

    def get_game_rights(self, board=None):
        # Get castling rights, en passant square, and turn
        if board is None:
            board = self.board
        rights = [
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
            int(board.ep_square is not None),
            int(board.turn)  # 1 for white to move, 0 for black
        ]
        return torch.tensor(rights, dtype=torch.float).to(DEVICE)

    def predict_move(self):
        # If in the opening phase, use the opening book
        if len(self.board.move_stack) < 2:
            return self.select_opening_move()

        # Use beam search to explore top N moves up to specified depth
        best_move = self.beam_search(self.board, self.move_history, self.depth, self.choices)
        return best_move

    def select_opening_move(self):
        color = 'white' if self.board.turn == chess.WHITE else 'black'
        possible_openings = self.opening_moves.get(color, [])
        legal_moves = list(self.board.legal_moves)
        legal_move_uci = [move.uci() for move in legal_moves]
        # Choose an opening move that is legal
        for move in possible_openings:
            if move in legal_move_uci:
                return chess.Move.from_uci(move)
        # Fallback to random legal move
        return random.choice(legal_moves)

    def beam_search(self, board: chess.Board, move_history: deque, depth: int, choices: int):
        # Initialize the beam with the current board and move history
        beam = [(board.copy(), list(move_history), 0.0)]  # (board, move_history, cumulative_score)

        for d in range(depth):
            candidates = []
            for b, mh, score in beam:
                # Generate top N moves for this board
                top_moves = self.get_top_n_moves(b, mh, choices)
                for move, move_score in top_moves:
                    # Apply the move
                    new_board = b.copy()
                    new_board.push(move)

                    # Update move history
                    new_move_history = mh.copy()
                    move_index = self.move_to_index.get(move.uci(), 0)
                    new_move_history.append(move_index)

                    # Evaluate opponent's responses if not at leaf node
                    if d < depth - 1:
                        # Opponent's turn
                        opp_moves = self.get_top_n_moves(new_board, new_move_history, choices)
                        for opp_move, opp_move_score in opp_moves:
                            # Apply opponent's move
                            opp_board = new_board.copy()
                            opp_board.push(opp_move)

                            # Update move history
                            opp_move_history = new_move_history.copy()
                            opp_move_index = self.move_to_index.get(opp_move.uci(), 0)
                            opp_move_history.append(opp_move_index)

                            # Add to candidates with cumulative score
                            cumulative_score = score + move_score - opp_move_score
                            candidates.append((opp_board, opp_move_history, cumulative_score))
                    else:
                        # Leaf node, no opponent response
                        cumulative_score = score + move_score
                        candidates.append((new_board, new_move_history, cumulative_score))

            # Keep top N candidates based on cumulative score
            beam = sorted(candidates, key=lambda x: x[2], reverse=True)[:choices]

        # After reaching the desired depth, select the move leading to the best candidate
        best_candidate = max(beam, key=lambda x: x[2])
        best_move_index = best_candidate[1][len(move_history)]  # Get the first move index
        best_move_uci = self.index_to_move.get(best_move_index, None)
        if best_move_uci:
            return chess.Move.from_uci(best_move_uci)
        else:
            # Fallback to a random legal move
            return random.choice(list(self.board.legal_moves))

    def get_top_n_moves(self, board: chess.Board, move_history: List[int], n: int) -> List[Tuple[chess.Move, float]]:
        # Truncate move_history to the last MAX_SEQUENCE_LENGTH moves
        truncated_move_history = move_history[-self.max_move_history:]  # Ensure length <= MAX_SEQUENCE_LENGTH

        # Prepare input tensors
        move_sequence = torch.tensor(truncated_move_history, dtype=torch.long).unsqueeze(0).to(DEVICE)
        attention_mask = (move_sequence != 0).long()
        board_state = self.get_board_state(board).unsqueeze(0)
        game_rights = self.get_game_rights(board).unsqueeze(0)

        # Get logits from the model
        with torch.no_grad():
            outputs = self.model(
                move_sequence=move_sequence,
                attention_mask=attention_mask,
                board_state=board_state,
                game_rights=game_rights
            )
            probabilities = torch.softmax(outputs, dim=-1).squeeze()

        # Mask illegal moves
        legal_moves = list(board.legal_moves)
        legal_move_indices = [self.move_to_index.get(move.uci(), 0) for move in legal_moves]
        mask = torch.zeros(self.move_vocab_size, device=DEVICE)
        mask[legal_move_indices] = 1
        masked_probabilities = probabilities * mask

        # Get top N legal moves
        top_indices = torch.topk(masked_probabilities, n).indices.tolist()
        top_moves = []
        for idx in top_indices:
            if idx == 0:
                continue  # Skip padding index
            move_uci = self.index_to_move.get(idx, None)
            if move_uci:
                move = chess.Move.from_uci(move_uci)
                if move in legal_moves:
                    score = masked_probabilities[idx].item()
                    top_moves.append((move, score))
        return top_moves

    def play(self):
        # Main loop for playing a game
        while not self.board.is_game_over():
            move = self.predict_move()
            self.make_move(move.uci())
            print(f"CASTLE ({'White' if self.board.turn == chess.BLACK else 'Black'}) plays: {move.uci()}")
            print(self.board)
            print('----------')

    def undo_last_move(self):
        if len(self.board.move_stack) < 1:
            print("No moves to undo.")
            return

        # Undo the human's move
        last_move = self.board.pop()
        if self.move_history:
            self.move_history.pop()
        print(f"Undid your move: {last_move.uci()}")

        # Check if there is an engine move to undo
        if len(self.board.move_stack) > 0:
            engine_move = self.board.pop()
            if self.move_history:
                self.move_history.pop()
            print(f"Undid CASTLE's move: {engine_move.uci()}")
        else:
            print("No engine move to undo.")

    def play_against_human(self, human_color: str = 'white'):
        human_color = human_color.lower()
        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                if human_color == 'white':
                    # Human plays
                    print(self.board)
                    move = input("Your move (in UCI format or 'undo' to revert): ").strip()
                    if move.lower() == 'undo':
                        self.undo_last_move()
                        continue
                    try:
                        self.board.push_uci(move)
                        self.move_history.append(self.move_to_index.get(move, 0))
                    except ValueError:
                        print("Invalid move. Try again.")
                        continue
                else:
                    # CASTLE plays
                    move = self.predict_move()
                    self.make_move(move.uci())
                    print(f"CASTLE (White) plays: {move.uci()}")
            else:
                if human_color == 'black':
                    # Human plays
                    print(self.board)
                    move = input("Your move (in UCI format or 'undo' to revert): ").strip()
                    if move.lower() == 'undo':
                        self.undo_last_move()
                        continue
                    try:
                        self.board.push_uci(move)
                        self.move_history.append(self.move_to_index.get(move, 0))
                    except ValueError:
                        print("Invalid move. Try again.")
                        continue
                else:
                    # CASTLE plays
                    move = self.predict_move()
                    self.make_move(move.uci())
                    print(f"CASTLE (Black) plays: {move.uci()}")
            print('----------')

        print("Game over!")
        print(f"Result: {self.board.result()}")

    def simulate_games_from_dataset(self, dataset_path: str):
        # Simulate games from a dataset file
        if not os.path.isfile(dataset_path):
            print(f"Dataset file '{dataset_path}' not found.")
            return

        with open(dataset_path, 'r') as f:
            for line in f:
                self.reset_game()
                moves = line.strip().split()
                for block in moves:
                    move_index_str, board_state_str, rights_str = block.strip().split('|')
                    move_index = int(move_index_str)
                    move_uci = self.index_to_move.get(move_index, None)
                    if move_uci:
                        try:
                            self.make_move(move_uci)
                        except ValueError:
                            print(f"Invalid move: {move_uci}")
                            continue
                    else:
                        print(f"Unknown move index: {move_index}")
                # Evaluate the game or collect statistics
                print("Simulated game:")
                print(self.board)
                print('----------')

    def self_play(self, num_games: int = 1):
        # Run self-play games
        for game_number in range(1, num_games + 1):
            print(f"Starting self-play game {game_number}")
            self.reset_game()
            self.play()
            # Here you can save the game, collect statistics, etc.
            print(f"Game {game_number} completed.")
            print('----------')

    # Additional methods for advanced features as needed

# Example usage
if __name__ == '__main__':
    # Load configuration
    config = {
        'model': {
            'path': 'checkpoints/model_epoch_11.pt',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'game': {
            'max_move_history': 19
        },
        'search': {
            'choices': 5,  # Top N moves to consider
            'depth': 8     # Depth of exploration
        }
    }

    # Instantiate the engine
    engine = CastleEngine(
        model_path=config['model']['path'],
        legal_moves_path='rules_based_mapping/all_legal_moves.json',
        config=config
    )

    # Play a self-play game
    # engine.self_play(num_games=1)

    # Play against a human
    engine.play_against_human(human_color='white')

    # Simulate games from a dataset
    # engine.simulate_games_from_dataset('dataset/split/castle_dataset_split_1.txt')
