import os
import chess
import chess.pgn
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

# Piece to integer encoding
PIECE_TO_INT = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,   # White pieces
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,  # Black pieces
    None: 0  # Empty square
}

# Global variable to store legal_moves in worker processes
LEGAL_MOVES = []

def initializer(legal_moves_json_path):
    """
    Initializer function for each worker process.
    Loads the legal_moves JSON into a global variable.
    """
    global LEGAL_MOVES
    try:
        with open(legal_moves_json_path, 'r') as f:
            LEGAL_MOVES = json.load(f)
    except Exception as e:
        print(f"Error loading legal_moves JSON: {e}")
        LEGAL_MOVES = []

def load_legal_moves(json_path):
    """
    Load the JSON file containing all legal UCI moves and return it as a list.
    """
    with open(json_path, 'r') as f:
        return json.load(f)

def board_to_int_list(board):
    """
    Convert the current board state to a list of 64 integers.
    Each square is represented by the piece on it according to PIECE_TO_INT.
    """
    int_board = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        piece_str = piece.symbol() if piece else None
        int_board.append(PIECE_TO_INT.get(piece_str, 0))
    return int_board

def process_line(line):
    """
    Process a single line of UCI moves and convert it into the enhanced format:
    'move_index|<board state>|<rights>'
    
    Parameters:
    - line: A string containing space-separated UCI moves.
    
    Returns:
    - A string with enhanced move information.
    """
    global LEGAL_MOVES
    enhanced_moves = []
    uci_moves = line.strip().split()
    board = chess.Board()

    for uci_move in uci_moves:
        # Find the index of the UCI move in the legal_moves list
        try:
            move_index = LEGAL_MOVES.index(uci_move)
        except ValueError:
            # Handle the case where the move is not found
            # You can choose to skip the move, assign a special index, or raise an error
            # Here, we'll assign -1 to indicate an invalid move
            move_index = -1

        # Convert the current board state to a list of integers
        board_state = board_to_int_list(board)
        board_state_str = ','.join(map(str, board_state))

        # Get current rights and turn info before the move
        rights = [
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
            int(board.ep_square is not None),
            int(board.turn)  # 1 for white to move, 0 for black
        ]

        # Create the rights string by joining the rights without spaces
        rights_str = ''.join(map(str, rights))

        # Append the move index, board state, and rights in the format:
        # 'move_index|<board state>|<rights>'
        move_with_metadata = f"{move_index}|{board_state_str}|{rights_str}"
        enhanced_moves.append(move_with_metadata)

        # Apply the move to the board
        try:
            board.push_uci(uci_move)
        except ValueError:
            # Handle invalid UCI move formats
            # You can choose to skip the move or stop processing
            # Here, we'll skip the move
            continue

    return ' '.join(enhanced_moves)

def add_metadata_and_board_state_with_index_parallel(
    uci_dataset_path,
    output_file_path,
    legal_moves_json_path,
    num_workers=None,
    chunksize=1000
):
    """
    Parallelized version of add_metadata_and_board_state_with_index.
    Processes the UCI moves dataset and appends metadata and board state information.
    
    Parameters:
    - uci_dataset_path: Path to the input UCI moves dataset file.
    - output_file_path: Path to the output file with enhanced move information.
    - legal_moves_json_path: Path to the JSON file containing legal UCI moves.
    - num_workers: Number of parallel worker processes. Defaults to the number of CPU cores.
    - chunksize: Number of lines to send to each worker at a time.
    """
    # Check if legal_moves JSON exists
    if not os.path.isfile(legal_moves_json_path):
        print(f"Legal moves JSON file '{legal_moves_json_path}' not found.")
        return

    # Count total lines for the progress bar
    with open(uci_dataset_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    if total_lines == 0:
        print("Input UCI dataset is empty.")
        return

    print(f"Total games to process: {total_lines}")

    # Initialize the ProcessPoolExecutor with initializer
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=initializer,
        initargs=(legal_moves_json_path,)
    ) as executor:
        with open(uci_dataset_path, 'r', encoding='utf-8') as input_file, \
             open(output_file_path, 'w', encoding='utf-8') as output_file:

            # Submit all lines to the executor
            # Using generator expression for memory efficiency
            futures = []
            for line in input_file:
                futures.append(executor.submit(process_line, line))

            # Initialize the progress bar
            with tqdm(total=total_lines, desc="Processing Games") as pbar:
                for future in as_completed(futures):
                    try:
                        enhanced_line = future.result()
                        output_file.write(enhanced_line + '\n')
                    except Exception as e:
                        # Handle exceptions from worker processes
                        print(f"Error processing a game: {e}")
                    finally:
                        pbar.update(1)

    print(f"Processing complete. Enhanced dataset saved to '{output_file_path}'.")

if __name__ == '__main__':
    # Define input and output file paths
    uci_dataset_path = 'dataset/2_uci_moves_dataset.txt'
    output_file_path = 'dataset/3_uci_moves_with_board_state_and_index.txt'
    legal_moves_json_path = 'rules_based_mapping/all_legal_moves.json'

    # Ensure the input dataset exists
    if not os.path.isfile(uci_dataset_path):
        print(f"Input UCI dataset file '{uci_dataset_path}' not found.")
    elif not os.path.isfile(legal_moves_json_path):
        print(f"Legal moves JSON file '{legal_moves_json_path}' not found.")
    else:
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_file_path)
        os.makedirs(output_dir, exist_ok=True)

        # Call the parallelized function
        add_metadata_and_board_state_with_index_parallel(
            uci_dataset_path=uci_dataset_path,
            output_file_path=output_file_path,
            legal_moves_json_path=legal_moves_json_path,
            num_workers=None,   # Defaults to the number of CPU cores
            chunksize=1000      # Adjust based on memory and performance considerations
        )
