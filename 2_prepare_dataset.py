import os
import chess.pgn
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm  # Optional: for progress bar

def extract_pgn_file_to_uci(pgn_file_path):
    """
    Extract UCI move sequences from a single PGN file.

    Parameters:
    - pgn_file_path: Path to the PGN file.

    Returns:
    - List of UCI move sequences, each as a space-separated string.
    """
    uci_move_lines = []
    try:
        with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  # No more games to read
                
                # Extract the moves of the game in UCI format
                uci_moves = [move.uci() for move in game.mainline_moves()]
                
                if uci_moves:  # Ensure there are moves to write
                    uci_move_line = ' '.join(uci_moves)
                    uci_move_lines.append(uci_move_line)
    except Exception as e:
        print(f"Error processing file {pgn_file_path}: {e}")
    
    return uci_move_lines

def extract_pgn_to_uci_parallel(
    source_dir,
    dataset_file_path,
    num_workers=None,
    chunksize=10
):
    """
    Parallelized extraction of UCI move sequences from PGN files.

    Parameters:
    - source_dir: Directory containing PGN files.
    - dataset_file_path: Path to the output dataset file.
    - num_workers: Number of parallel worker processes. Defaults to CPU cores.
    - chunksize: Number of PGN files to process per chunk.
    """
    # Gather all PGN file paths
    pgn_file_paths = [
        os.path.join(source_dir, file_name)
        for file_name in os.listdir(source_dir)
        if file_name.lower().endswith('.pgn')
    ]
    
    total_files = len(pgn_file_paths)
    if total_files == 0:
        print("No PGN files found in the specified directory.")
        return
    
    print(f"Found {total_files} PGN files. Starting extraction...")

    # Open the output file in write mode
    with open(dataset_file_path, 'w', encoding='utf-8') as dataset_file:
        # Initialize the ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all PGN files to the executor
            future_to_file = {
                executor.submit(extract_pgn_file_to_uci, pgn_file_path): pgn_file_path
                for pgn_file_path in pgn_file_paths
            }
            
            # Initialize the progress bar
            with tqdm(total=total_files, desc="Processing PGN Files") as pbar:
                for future in as_completed(future_to_file):
                    pgn_file_path = future_to_file[future]
                    try:
                        uci_move_lines = future.result()
                        # Write each UCI move sequence to the dataset file
                        for uci_move_line in uci_move_lines:
                            dataset_file.write(uci_move_line + '\n')
                    except Exception as e:
                        print(f"Error processing file {pgn_file_path}: {e}")
                    finally:
                        pbar.update(1)

    print(f"Extraction complete. UCI move sequences saved to {dataset_file_path}")

if __name__ == '__main__':
    # Define input and output paths
    source_directory = './pgn'  # Directory containing PGN files
    output_dataset = './dataset/2_uci_moves_dataset.txt'  # Output dataset file path
    
    # Ensure the source directory exists
    if not os.path.isdir(source_directory):
        print(f"Source directory '{source_directory}' does not exist.")
    else:
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_dataset)
        os.makedirs(output_dir, exist_ok=True)
        
        # Call the parallelized extraction function
        extract_pgn_to_uci_parallel(
            source_dir=source_directory,
            dataset_file_path=output_dataset,
            num_workers=None,   # Defaults to the number of CPU cores
            chunksize=10        # Number of files to process per chunk; adjust as needed
        )
