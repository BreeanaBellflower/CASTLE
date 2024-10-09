import torch
import torch.nn as nn
import json
import os

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../rules_based_mapping/all_legal_moves.json')
file_path = os.path.normpath(file_path)

# Load the legal moves mapping
with open(file_path, 'r') as f:
    legal_moves = json.load(f)
    # Create index to move mapping (index shifted by +1 to reserve 0 for padding)
    index_to_move = {idx + 1: move for idx, move in enumerate(legal_moves)}
    move_to_index = {move: idx + 1 for idx, move in enumerate(legal_moves)}


MAX_SEQUENCE_LENGTH = 20

MOVE_VOCAB_SIZE = len(legal_moves) + 1  # Total number of unique moves + 1 for padding index 0
BOARD_VOCAB_SIZE = 13  # Values from 0 to 12 after shifting (-6 to 6)
BOARD_STATE_SIZE = 64  # Number of squares on the chessboard
GAME_RIGHTS_SIZE = 6   # Castling rights, en passant, turn indicator

# Model Definition
class ChessMovePredictionModel(nn.Module):
    def __init__(self):
        super(ChessMovePredictionModel, self).__init__()

        # Embedding layers
        self.move_embedding = nn.Embedding(MOVE_VOCAB_SIZE, 256, padding_idx=0)
        self.board_embedding = nn.Embedding(BOARD_VOCAB_SIZE, 256, padding_idx=0)
        self.game_rights_linear = nn.Linear(GAME_RIGHTS_SIZE, 32)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, MAX_SEQUENCE_LENGTH, 256))

        # Transformer Encoder for move sequences
        encoder_layer_moves = nn.TransformerEncoderLayer(d_model=256, nhead=32)
        self.move_transformer = nn.TransformerEncoder(encoder_layer_moves, num_layers=2)

        # Transformer Encoder for board state
        encoder_layer_board = nn.TransformerEncoderLayer(d_model=256, nhead=64)
        self.board_transformer = nn.TransformerEncoder(encoder_layer_board, num_layers=2)

        # MLP for board features
        self.board_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BOARD_STATE_SIZE * 256, 256),  # BOARD_STATE_SIZE = 64
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # Final layers
        self.final_mlp = nn.Sequential(
            nn.Linear(256 + 256 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, MOVE_VOCAB_SIZE)
        )

    def forward(self, move_sequence, attention_mask, board_state, game_rights):
        # Move sequence embedding
        move_embedded = self.move_embedding(move_sequence)
        move_embedded += self.positional_encoding[:, :move_embedded.size(1), :]

        # Apply attention mask
        attention_mask_bool = attention_mask == 0  # True where padding

        # Move sequence transformer
        move_features = self.move_transformer(
            move_embedded.transpose(0, 1), src_key_padding_mask=attention_mask_bool
        ).transpose(0, 1)
        sequence_features = move_features[:, -1, :]  # Use the last token's features

        # Board state embedding
        board_embedded = self.board_embedding(board_state)
        board_features = self.board_transformer(board_embedded.transpose(0, 1)).transpose(0, 1)
        board_features = self.board_mlp(board_features)

        # Game rights embedding
        game_rights_features = self.game_rights_linear(game_rights)

        # Concatenate features
        combined_features = torch.cat([sequence_features, board_features, game_rights_features], dim=1)

        # Final MLP
        logits = self.final_mlp(combined_features)

        return logits
