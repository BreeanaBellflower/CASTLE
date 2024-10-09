import os
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import logging
from model.ChessMovePredictionModel import ChessMovePredictionModel;

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', filename='training_log.txt')

# Enable CUDA Launch Blocking and Anomaly Detection for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants
SUBSEQUENCE_MAX = 8
BATCH_SIZE = 2000
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # Prevents issues with tqdm and multiprocessing
CHUNK_SIZE = 20000  # Number of lines per chunk
MIN_TURNS = 8  # Minimum number of turns required to process a row

# Load the legal moves mapping
with open('rules_based_mapping/all_legal_moves.json', 'r') as f:
    legal_moves = json.load(f)
    # Create index to move mapping (index shifted by +1 to reserve 0 for padding)
    index_to_move = {idx + 1: move for idx, move in enumerate(legal_moves)}
    move_to_index = {move: idx + 1 for idx, move in enumerate(legal_moves)}

MOVE_VOCAB_SIZE = len(legal_moves) + 1  # Total number of unique moves + 1 for padding index 0
BOARD_VOCAB_SIZE = 13  # Values from 0 to 12 after shifting (-6 to 6)
BOARD_STATE_SIZE = 64  # Number of squares on the chessboard
GAME_RIGHTS_SIZE = 6   # Castling rights, en passant, turn indicator

# Custom Iterable Dataset
class ChunkDataset(IterableDataset):
    def __init__(self, chunk_lines):
        self.chunk_lines = chunk_lines

    def __iter__(self):
        return self.data_generator()

    def data_generator(self):
        max_allowed_move_index = MOVE_VOCAB_SIZE - 1  # Valid indices are from 0 to MOVE_VOCAB_SIZE - 1
        for line in self.chunk_lines:
            blocks = line.strip().split()
            move_data = []
            for block in blocks:
                try:
                    move_index_str, board_state_str, rights_str = block.strip().split('|')
                    move_index = int(move_index_str)

                    # Validate move_index before shifting
                    if move_index < 0 or move_index >= MOVE_VOCAB_SIZE - 1:
                        logging.warning(f"Invalid move_index {move_index} before shifting. Skipping.")
                        continue

                    move_index += 1  # Shift index by 1

                    if move_index > max_allowed_move_index:
                        logging.warning(f"Invalid move_index {move_index} after shifting. Skipping.")
                        continue

                    # Shift board_state indices from -6..6 to 0..12
                    board_state = [int(sq) + 6 for sq in board_state_str.split(',')]

                    # Validate board_state values
                    max_board_value = max(board_state)
                    min_board_value = min(board_state)
                    if max_board_value > 12 or min_board_value < 0:
                        logging.warning(f"Invalid board_state values detected. Min: {min_board_value}, Max: {max_board_value}. Skipping.")
                        continue

                    # Validate game_rights
                    game_rights = [int(c) for c in rights_str]
                    if len(game_rights) != GAME_RIGHTS_SIZE:
                        logging.warning(f"Invalid game_rights length {len(game_rights)}. Skipping.")
                        continue

                    move_data.append((move_index, board_state, game_rights))
                except ValueError as e:
                    logging.warning(f"ValueError: {e}. Skipping block.")
                    continue

            # Skip rows with fewer than MIN_TURNS
            if len(move_data) < MIN_TURNS:
                logging.info(f"Row with {len(move_data)} turns is skipped (requires at least {MIN_TURNS} turns).")
                continue

            # Generate subsequences
            subsequences = generate_subsequences_with_restrictions(move_data)
            for subseq in subsequences:
                if len(subseq) < 2:
                    continue

                input_sequence = subseq[:-1]
                target_move = subseq[-1]

                input_move_indices = [item[0] for item in input_sequence]
                board_state = input_sequence[-1][1]
                game_rights = input_sequence[-1][2]
                target_move_index = target_move[0]

                # Additional validation
                if any(idx < 0 or idx >= MOVE_VOCAB_SIZE for idx in input_move_indices):
                    logging.warning(f"Invalid input_move_indices {input_move_indices}. Skipping.")
                    continue

                if target_move_index < 0 or target_move_index >= MOVE_VOCAB_SIZE:
                    logging.warning(f"Invalid target_move_index {target_move_index}. Skipping.")
                    continue

                yield {
                    'move_sequence': torch.tensor(input_move_indices, dtype=torch.long),
                    'board_state': torch.tensor(board_state, dtype=torch.long),
                    'game_rights': torch.tensor(game_rights, dtype=torch.float),
                    'target': torch.tensor(target_move_index, dtype=torch.long)
                }

def collate_fn(batch):
    # Pad sequences and create attention masks
    move_sequences = [item['move_sequence'] for item in batch]
    board_states = [item['board_state'] for item in batch]
    game_rights = [item['game_rights'] for item in batch]
    targets = [item['target'] for item in batch]

    move_sequences_padded = nn.utils.rnn.pad_sequence(move_sequences, batch_first=True, padding_value=0)
    attention_masks = (move_sequences_padded != 0).long()
    board_states_tensor = torch.stack(board_states)
    game_rights_tensor = torch.stack(game_rights)
    targets_tensor = torch.stack(targets)

    return {
        'move_sequence': move_sequences_padded.to(device),
        'attention_mask': attention_masks.to(device),
        'board_state': board_states_tensor.to(device),
        'game_rights': game_rights_tensor.to(device),
        'target': targets_tensor.to(device)
    }

# Subsequence Generation Functions
def generate_subsequences_with_restrictions(move_data, max_length=20, subsequence_max=SUBSEQUENCE_MAX):
    """
    Generate subsequences from move_data with specific restrictions.
    Each item in move_data is a tuple: (move_index, board_state, game_rights)
    """
    subsequences = []
    num_moves = len(move_data)

    if num_moves < MIN_TURNS:
        # Already handled in data_generator, but added as a safety net
        return []

    # Generate at least one subsequence starting from the beginning
    start_subsequence_length = random.randint(2, min(max_length, num_moves))
    start_subsequence = move_data[:start_subsequence_length]
    subsequences.append(start_subsequence)

    # Generate at least one subsequence ending at the end
    end_subsequence_length = random.randint(2, min(max_length, num_moves))
    end_subsequence = move_data[-end_subsequence_length:]
    subsequences.append(end_subsequence)

    # Generate random subsequences
    num_random_subsequences = min(subsequence_max, num_moves)
    for _ in range(num_random_subsequences):
        if num_moves <= 2:
            break
        start_index = random.randint(0, num_moves - 2)
        length = random.randint(2, min(max_length, num_moves - start_index))
        random_subsequence = move_data[start_index:start_index + length]
        subsequences.append(random_subsequence)

    return subsequences

# Training Loop with Enhanced Progress Bars
def train_model():
    model = ChessMovePredictionModel().to(device)
    selected_epoch = 0
    # checkpoint = torch.load(f'checkpoints/model_epoch_{selected_epoch}.pt', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Get list of split files
    split_dir = 'dataset/train/'
    split_files = [os.path.join(split_dir, fname) for fname in os.listdir(split_dir) if fname.startswith('castle_dataset_split_')]
    split_files.sort()  # Ensure consistent order
    total_split_files = len(split_files)

    for epoch in range(selected_epoch, NUM_EPOCHS):
        with tqdm(total=total_split_files, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", position=0, leave=True) as epoch_bar:
            model.train()

            random.shuffle(split_files)  # Shuffle split files each epoch

            for split_idx, split_file in enumerate(split_files, 1):
                split_file_name = os.path.basename(split_file)
                # Initialize loss and accuracy counters for this split file
                total_loss = 0
                total_batches = 0
                total_samples = 0
                top1_correct = 0
                top3_correct = 0
                top5_correct = 0
                top10_correct = 0

                # File Processing Progress Bar
                with open(split_file, 'r') as f:
                    chunk_lines = []
                    # Count total lines in file for the progress bar
                    total_lines_in_file = sum(1 for _ in f)
                    f.seek(0)  # Reset file pointer

                    with tqdm(total=total_lines_in_file, desc=f"File {split_idx}/{total_split_files}: {split_file_name}", position=1, leave=False) as file_bar:
                        for line in f:
                            chunk_lines.append(line.strip())
                            file_bar.update(1)

                            if len(chunk_lines) >= CHUNK_SIZE:
                                # Create dataset and dataloader
                                dataset = ChunkDataset(chunk_lines)
                                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

                                # Estimate total batches (approximate)
                                estimated_total_batches = CHUNK_SIZE // BATCH_SIZE + 1

                                # Training on this chunk
                                with tqdm(total=estimated_total_batches, desc="Training", position=2, leave=False) as train_bar:
                                    for batch in dataloader:
                                        optimizer.zero_grad()
                                        outputs = model(
                                            batch['move_sequence'],
                                            batch['attention_mask'],
                                            batch['board_state'],
                                            batch['game_rights']
                                        )
                                        loss = criterion(outputs, batch['target'])
                                        loss.backward()
                                        optimizer.step()

                                        # Update loss and batch count
                                        total_loss += loss.item()
                                        total_batches += 1

                                        # Compute top-k accuracies
                                        with torch.no_grad():
                                            total_samples += batch['target'].size(0)

                                            # Get the top 5 predictions
                                            _, pred_top = outputs.topk(10, dim=1)

                                            # Top-1 accuracy
                                            top1_correct += (pred_top[:, 0] == batch['target']).sum().item()

                                            # Top-3 accuracy
                                            top3_correct += sum([
                                                batch['target'][i].item() in pred_top[i, :3]
                                                for i in range(batch['target'].size(0))
                                            ])

                                            # Top-5 accuracy
                                            top5_correct += sum([
                                                batch['target'][i].item() in pred_top[i, :5]
                                                for i in range(batch['target'].size(0))
                                            ])

                                            # Top-10 accuracy
                                            top10_correct += sum([
                                                batch['target'][i].item() in pred_top[i, :]
                                                for i in range(batch['target'].size(0))
                                            ])

                                        train_bar.update(1)

                                chunk_lines = []  # Reset chunk lines

                        # Process remaining lines in the last chunk
                        if chunk_lines:
                            dataset = ChunkDataset(chunk_lines)
                            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

                            # Estimate total batches (approximate)
                            remaining_data_points = len(chunk_lines)
                            estimated_total_batches = remaining_data_points // BATCH_SIZE + 1

                            with tqdm(total=estimated_total_batches, desc="Training", position=2, leave=False) as train_bar:
                                for batch in dataloader:
                                    optimizer.zero_grad()
                                    outputs = model(
                                        batch['move_sequence'],
                                        batch['attention_mask'],
                                        batch['board_state'],
                                        batch['game_rights']
                                    )
                                    loss = criterion(outputs, batch['target'])
                                    loss.backward()
                                    optimizer.step()

                                    # Update loss and batch count
                                    total_loss += loss.item()
                                    total_batches += 1

                                    # Compute top-k accuracies
                                    with torch.no_grad():
                                        total_samples += batch['target'].size(0)

                                        # Get the top 5 predictions
                                        _, pred_top = outputs.topk(10, dim=1)

                                        # Top-1 accuracy
                                        top1_correct += (pred_top[:, 0] == batch['target']).sum().item()

                                        # Top-3 accuracy
                                        top3_correct += sum([
                                            batch['target'][i].item() in pred_top[i, :3]
                                            for i in range(batch['target'].size(0))
                                        ])

                                        # Top-5 accuracy
                                        top5_correct += sum([
                                            batch['target'][i].item() in pred_top[i, :5]
                                            for i in range(batch['target'].size(0))
                                        ])

                                        # Top-10 accuracy
                                        top10_correct += sum([
                                            batch['target'][i].item() in pred_top[i, :]
                                            for i in range(batch['target'].size(0))
                                        ])

                                    train_bar.update(1)

                            chunk_lines = []  # Reset chunk lines

                # Calculate average metrics for this split file
                avg_loss = total_loss / total_batches if total_batches > 0 else 0
                top1_acc = top1_correct / total_samples if total_samples > 0 else 0
                top3_acc = top3_correct / total_samples if total_samples > 0 else 0
                top5_acc = top5_correct / total_samples if total_samples > 0 else 0
                top10_acc = top10_correct / total_samples if total_samples > 0 else 0

                # Log the metrics at the same location as before
                epoch_bar.update(1)  # Update epoch progress bar
                msg = (f"Chunk {epoch + 1}:{split_idx} completed. "
                       f"Average Loss: {avg_loss:.4f}, "
                       f"Top-1 Acc: {top1_acc:.4f}, "
                       f"Top-3 Acc: {top3_acc:.4f}, "
                       f"Top-5 Acc: {top5_acc:.4f}, "
                       f"Top-10 Acc: {top10_acc:.4f}")
                print(msg)
                logging.info(msg)
                checkpoint_path = f'checkpoints/model_latest_chunk.pt'
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Model checkpoint saved at {checkpoint_path}")

            # Step the scheduler after each epoch
            scheduler.step()

            # Optionally, you can log epoch-level metrics if desired
            # avg_epoch_loss = ...
            # msg = f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}"
            # print(msg)
            # logging.info(msg)

    print("Training completed.")

if __name__ == '__main__':
    train_model()
