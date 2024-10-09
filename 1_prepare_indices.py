import chess
import pandas as pd
import json

# Initialize the moves dictionary with empty lists for each piece type
moves = {
    "white_pawn": [],
    "black_pawn": [],
    "white_rook": [],
    "black_rook": [],
    "white_knight": [],
    "black_knight": [],
    "white_bishop": [],
    "black_bishop": [],
    "white_queen": [],
    "black_queen": [],
    "white_king": [],
    "black_king": []
}

all_moves = set()

# Define piece types to work with
piece_types = {
    "white_pawn": chess.PAWN,
    "black_pawn": chess.PAWN,
    "white_rook": chess.ROOK,
    "black_rook": chess.ROOK,
    "white_knight": chess.KNIGHT,
    "black_knight": chess.KNIGHT,
    "white_bishop": chess.BISHOP,
    "black_bishop": chess.BISHOP,
    "white_queen": chess.QUEEN,
    "black_queen": chess.QUEEN,
    "white_king": chess.KING,
    "black_king": chess.KING
}

# Promotion pieces
promotion_pieces = ['q', 'r', 'b', 'n']  # Queen, Rook, Bishop, Knight

# Function to check if the piece is white
def is_white(piece_name):
    return "white" in piece_name

# Initialize an empty chess board
board = chess.Board()
board.clear_board()  # Clear the board to ensure only one piece is present

# Iterate over each piece type and each square
for piece_name, piece_type in piece_types.items():
    # Determine the color based on the piece name
    color = chess.WHITE if is_white(piece_name) else chess.BLACK

    # Iterate over all squares
    for square in chess.SQUARES:
        # Clear the board before placing a new piece
        board.clear_board()

        # Place the piece on the board
        board.set_piece_at(square, chess.Piece(piece_type, color))
        board.turn = color  # Set the turn to the piece's color

        # For pawns, we need to consider promotion moves, including captures
        if piece_type == chess.PAWN:
            rank = chess.square_rank(square)

            # For white pawns
            if color == chess.WHITE:
                if rank == 6:  # Rank 7 (0-indexed)
                    # Generate all moves for white pawn on rank 7
                    for move in board.generate_legal_moves(from_mask=chess.BB_SQUARES[square]):
                        # Check if the move is a promotion
                        if chess.square_rank(move.to_square) == 7:
                            # It's a promotion move
                            for promo_piece in promotion_pieces:
                                promo_move = chess.Move(move.from_square, move.to_square, promotion=chess.Piece.from_symbol(promo_piece).piece_type)
                                uci_move = promo_move.uci()
                                moves[piece_name].append(uci_move)
                                all_moves.add(uci_move)
                else:
                    # Generate normal pawn moves
                    for move in board.generate_legal_moves(from_mask=chess.BB_SQUARES[square]):
                        uci_move = move.uci()
                        moves[piece_name].append(uci_move)
                        all_moves.add(uci_move)

            # For black pawns
            else:
                if rank == 1:  # Rank 2 (0-indexed)
                    # Generate all moves for black pawn on rank 2
                    for move in board.generate_legal_moves(from_mask=chess.BB_SQUARES[square]):
                        # Check if the move is a promotion
                        if chess.square_rank(move.to_square) == 0:
                            # It's a promotion move
                            for promo_piece in promotion_pieces:
                                promo_move = chess.Move(move.from_square, move.to_square, promotion=chess.Piece.from_symbol(promo_piece).piece_type)
                                uci_move = promo_move.uci()
                                moves[piece_name].append(uci_move)
                                all_moves.add(uci_move)
                else:
                    # Generate normal pawn moves
                    for move in board.generate_legal_moves(from_mask=chess.BB_SQUARES[square]):
                        uci_move = move.uci()
                        moves[piece_name].append(uci_move)
                        all_moves.add(uci_move)
        else:
            # Generate legal moves for the piece in UCI notation
            for move in board.generate_legal_moves(from_mask=chess.BB_SQUARES[square]):
                uci_move = move.uci()
                moves[piece_name].append(uci_move)
                all_moves.add(uci_move)

        # No need to reset the square since we clear the board at the start of each loop

# Manually add castling moves
castling_moves = ['e1g1', 'e1c1', 'e8g8', 'e8c8']
for castling_move in castling_moves:
    # Castling is a king move
    if castling_move in ['e1g1', 'e1c1']:
        moves["white_king"].append(castling_move)
    else:
        moves["black_king"].append(castling_move)
    all_moves.add(castling_move)

# --------------------------------------------------------
# Additional Loop to Add Promotion Capture Moves
# --------------------------------------------------------

# Define promotion ranks
white_promotion_rank = 7  # Rank 8 (0-indexed)
black_promotion_rank = 0  # Rank 1 (0-indexed)

# Function to get file of a square
def get_file(square):
    return chess.square_file(square)

# Function to get rank of a square
def get_rank(square):
    return chess.square_rank(square)

# Add promotion capture moves for white pawns
for from_square in range(48, 56):  # Squares a7 to h7
    # Only white pawns on rank 7 can perform promotion captures
    if get_rank(from_square) == 6:
        from_file = get_file(from_square)
        # Possible capture directions: left (-1) and right (+1)
        capture_files = []
        if from_file > 0:
            capture_files.append(from_file - 1)
        if from_file < 7:
            capture_files.append(from_file + 1)
        
        for to_file in capture_files:
            to_square = chess.square(to_file, white_promotion_rank)
            promo_piece_names = promotion_pieces
            for promo_piece in promo_piece_names:
                uci_move = chess.square_name(from_square) + chess.square_name(to_square) + promo_piece
                moves["white_pawn"].append(uci_move)
                all_moves.add(uci_move)

# Add promotion capture moves for black pawns
for from_square in range(8, 16):  # Squares a2 to h2
    # Only black pawns on rank 2 can perform promotion captures
    if get_rank(from_square) == 1:
        from_file = get_file(from_square)
        # Possible capture directions: left (-1) and right (+1)
        capture_files = []
        if from_file > 0:
            capture_files.append(from_file - 1)
        if from_file < 7:
            capture_files.append(from_file + 1)
        
        for to_file in capture_files:
            to_square = chess.square(to_file, black_promotion_rank)
            promo_piece_names = promotion_pieces
            for promo_piece in promo_piece_names:
                uci_move = chess.square_name(from_square) + chess.square_name(to_square) + promo_piece
                moves["black_pawn"].append(uci_move)
                all_moves.add(uci_move)

# --------------------------------------------------------

# Verification: Check if 'b2a1q' is included
if 'b2a1q' in all_moves:
    print("'b2a1q' is included in the move list.")
else:
    print("'b2a1q' is missing from the move list.")

# Display the count of moves for each piece
total_moves = 0
move_counts = []
for piece_name, move_list in moves.items():
    move_counts.append((piece_name, len(move_list)))
    total_moves += len(move_list)

move_counts_df = pd.DataFrame(move_counts, columns=['Piece', 'Move Count'])
move_counts_df.loc['Total'] = ['Total moves', total_moves]

print(move_counts_df)

# Save the moves to JSON files
with open('rules_based_mapping/chess_move_indices.json', 'w') as f:
    json.dump(moves, f, indent=4)

with open('rules_based_mapping/all_legal_moves.json', 'w') as f:
    sorted_all_moves = sorted(all_moves)
    print('All unique moves:', len(sorted_all_moves))
    json.dump(sorted_all_moves, f, indent=4)
