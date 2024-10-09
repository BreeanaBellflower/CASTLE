**Title:**
Developing a Transformer-Based Chess Move Prediction Model Leveraging Game Sequences and Board State Representations

**Abstract:**
We present a comprehensive approach to building a chess move prediction model utilizing transformer architectures. The model integrates sequences of historical moves and explicit board state representations to predict subsequent moves, aiming to capture the temporal dependencies and spatial relationships inherent in chess games. This document outlines the data preprocessing methods, move index encoding, model architecture, training procedures, and implementation details essential for replicating and understanding the proposed system.

**Introduction:**
Predicting chess moves requires understanding both the historical sequence of moves and the current board state. Traditional models often struggle to capture the complex dependencies in chess. We propose a transformer-based model that processes sequences of moves and explicit board states, enabling it to learn temporal patterns and spatial relationships effectively.

**Methods:**

### Data Preprocessing:

1. **Data Source:**
   - A collection of Portable Game Notation (PGN) files containing approximately 2.2 million professional chess games.

2. **Extraction and Encoding:**
   - **PGN Parsing:** Games are parsed using the `python-chess` library to extract move sequences in Universal Chess Interface (UCI) notation.
   - **UCI Move Generation:** Each game is converted into a sequence of UCI moves.
   - **Metadata Addition:** For each move, additional metadata including board state, castling rights, en passant availability, and turn indicator are appended. Each record follows the format: `move_index|<board state>|<rights>`.

3. **Move Index Encoding:**
   
   To facilitate efficient move prediction, each unique chess move is encoded as a unique integer index. This encoding process ensures that the model can handle move representations consistently and effectively.

   **Encoding Process:**
   
   - **Move Categorization:** Moves are categorized based on the piece type and color, such as `white_pawn`, `black_rook`, etc.
   - **Move Generation:** For each piece type and position on the board, all possible legal moves are generated, including special moves like castling and pawn promotions.
   - **Promotion Handling:** Promotion moves are explicitly handled by considering all possible promotion pieces (`queen`, `rook`, `bishop`, `knight`) and capturing moves.
   - **Move Collection:** All unique UCI moves are collected and stored in a structured format, mapping each move to its corresponding piece type.

   **Implementation Details:**
   
   - **Script Functionality:** The provided Python script iterates over each piece type and board square to generate all possible legal UCI moves. It accounts for special moves such as castling and pawn promotions, ensuring comprehensive move coverage.
   - **Data Structures:**
     - **`moves` Dictionary:** A dictionary mapping each piece type (e.g., `white_pawn`, `black_rook`) to a list of corresponding UCI moves.
     - **`all_moves` Set:** A set containing all unique UCI moves across all piece types.
   
   **Output Files:**
   
   - **Move Indices JSON (`chess_move_indices.json`):**
     - **Location:** Stored in the `rules_based_mapping/` directory.
     - **Format:** A JSON file where each key is a piece type, and the value is a list of UCI moves corresponding to that piece. For example:
       ```json
       {
           "white_pawn": ["e2e4", "e2e3", ...],
           "black_rook": ["a8a6", "a8a5", ...],
           ...
       }
       ```
   
   - **All Legal Moves JSON (`all_legal_moves.json`):**
     - **Location:** Also stored in the `rules_based_mapping/` directory.
     - **Format:** A sorted JSON array containing all unique UCI moves. For example:
       ```json
       [
           "a2a3",
           "a2a4",
           "a7a6",
           "a7a5",
           ...
       ]
       ```
   
   **Usage in Model Training:**
   
   - **Move Index Mapping:** The `all_legal_moves.json` file is used to map each UCI move to a unique integer index (0 to 4,139). This mapping facilitates the translation of move sequences into numerical representations suitable for input into the transformer model.
   - **Consistent Encoding/Decoding:** By referencing the `all_legal_moves.json` file during both encoding and decoding phases, the system ensures that each move is consistently represented by its unique index across different datasets and model training sessions.

4. **Dataset Splitting:**
   - **Initial Dataset:** The complete dataset is compiled into a single large file named `3_uci_moves_with_board_state_and_index.txt`.
   - **Splitting into Chunks:** To manage memory constraints and facilitate parallel processing, the large dataset is split into multiple smaller files:
     - **Location:** All split files are stored within the `dataset/split/` directory.
     - **Naming Convention:** Each split file is named following the pattern `castle_dataset_split_{n}.txt`, where `{n}` is a sequential integer starting from 1 (e.g., `castle_dataset_split_1.txt`, `castle_dataset_split_2.txt`, ..., `castle_dataset_split_100.txt`).
     - **File Size:** Each split file contains up to 20,000 lines, ensuring manageable file sizes for subsequent processing steps.
     - **Data Format:** Each line within the split files adheres to the format `move_index|<board state>|<rights>`, where:
       - **`move_index`**: An integer representing the index of the move within the `legal_moves` list.
       - **`<board state>`**: A comma-separated string of 64 integers representing the board state, where each integer corresponds to a specific piece or an empty square.
       - **`<rights>`**: A 6-character string encoding castling rights, en passant availability, and turn information.

5. **Board State Representation:**
   - Each board state is encoded as a 64-length vector, where each element represents a square on the board. Pieces are mapped to integers (-6 to 6) corresponding to different piece types and colors, with `0` representing an empty square.

6. **Game Rights Encoding:**
   - Additional features include castling rights (kingside and queenside for both white and black), en passant availability, and a turn indicator. These are encoded as a 6-element vector to provide contextual information about the game state.

7. **Attention Masks:**
   - For handling variable-length sequences, attention masks are created to differentiate between actual moves and padding tokens, facilitating efficient processing by the transformer model.

8. **Data Storage:**
   - **Split Files:** The processed and split data is stored as multiple text files within the `dataset/split/` directory, each containing up to 20,000 lines in the specified format.
   - **Future Integration:** These split files are designed to be easily loaded and further processed into formats like HDF5 or directly fed into data loaders for model training.

### Model Architecture:

1. **Input Layers:**
   - **Move Sequence Input:** Accepts sequences of move indices with positional encodings added to capture the order of moves.
   - **Board State Input:** Processes the encoded board state through an embedding layer.
   - **Game Rights Input:** Processes the game rights through a linear layer to create embeddings.

2. **Transformer Encoders:**
   - **Move Sequence Transformer:** Processes the move sequence using multiple transformer encoder layers, leveraging attention mechanisms to capture temporal dependencies.
   - **Board State Transformer:** Processes the embedded board state to capture spatial relationships between pieces.

3. **Feature Extraction:**
   - **Sequence Features:** The output corresponding to the last move in the sequence represents the sequence features.
   - **Board Features:** The output from the board state transformer is flattened and passed through a multi-layer perceptron (MLP) to extract high-level features.

4. **Feature Fusion and Output:**
   - **Concatenation:** Sequence features, board features, and game rights embeddings are concatenated.
   - **Final MLP Layers:** The combined features are passed through additional MLP layers.
   - **Output Layer:** Produces logits for all possible moves (size 4,140), without explicit masking of illegal moves, trusting the model to learn move legality from data.

### Training Procedure:

1. **Data Loading and Chunk Processing:**
   - **Epochs:** The model is trained for a predefined number of epochs, with each epoch encompassing a single full pass through the entire training dataset.
   - **Chunks:** The training data is divided into chunks of 20,000 data points each to manage memory usage effectively. These chunks are processed sequentially within each epoch.
   - **In-Memory Expansion:**
     - For each chunk of 20,000 data points, the following in-memory expansion process is applied:
       - **Subsequence Generation:** Utilizing logic similar to the provided `generate_subsequences_with_restrictions` function, each UCI move sequence within the chunk is expanded into multiple subsequences with specific restrictions:
         - **Start Subsequence:** At least one subsequence starting from the beginning of the game.
         - **End Subsequence:** At least one subsequence ending at the end of the game.
         - **Random Subsequences:** Up to a specified maximum number of random subsequences (e.g., `subsequence_max=5`).
       - **Parallel Processing:** The expansion of subsequences is performed in parallel using multiple worker processes to expedite the preprocessing of large chunks.
   
2. **Batch Processing:**
   - **Batch Size:** Set to 64, adjustable based on memory constraints.
   - **Batch Formation:** After expanding a chunk into multiple subsequences, the data is further divided into batches of size `BATCH_SIZE`. Each batch is prepared for input into the transformer model.
   - **Parallel Execution:** Batches within a chunk are processed in parallel, leveraging multi-processing or multi-threading to optimize training speed and resource utilization.

3. **Loss Function:**
   - Cross-entropy loss is used to compare the predicted move probabilities with the actual next move.

4. **Optimizer and Learning Rate Scheduler:**
   - **Optimizer:** Adam optimizer with an initial learning rate of \(1 \times 10^{-4}\).
   - **Scheduler:** Learning rate decays by a factor of 0.1 every 5 epochs.

5. **Training Loop:**
   - **Epochs:** The model is trained for 20 epochs.
   - **Checkpointing:** Model checkpoints are saved after each epoch, including epoch number, training accuracy, and validation accuracy.
   - **Logging:** Training and validation accuracies are logged, and progress is tracked every 5% of an epoch.
   - **Data Shuffling:** Within each epoch, the order of data chunks is shuffled to ensure varied exposure to data patterns and prevent the model from learning order-specific biases.

### Implementation Details:

1. **Programming Language and Libraries:**
   - Python with PyTorch for neural network implementation, and `python-chess` library for parsing PGN files and handling chess-specific functionalities.

2. **Code Structure:**
   - **Data Preprocessing Scripts:** Handle parsing, encoding, sequence generation, dataset splitting, and data storage with checkpointing for fault tolerance.
   - **Move Index Encoding Scripts:** Generate and store move index mappings (`chess_move_indices.json` and `all_legal_moves.json`) for consistent encoding and decoding.
   - **Training Scripts:** Contain functions for loading data chunks, in-memory expansion of subsequences, batching, model training loops, validation steps, checkpointing, and logging.
   - **Model Definition:** Encapsulated in a class inheriting from `nn.Module`, defining all layers and the forward pass.
   - **Utilities:** Include scripts for handling data loading, preprocessing, and other auxiliary tasks.

3. **Hardware Considerations:**
   - Training is conducted on GPUs to accelerate computation, with considerations for memory management due to the model's complexity and dataset size.
   - Efficient utilization of CPU resources for parallel processing during in-memory data expansion.

4. **Data Organization:**
   - **Raw Data:** Stored in the `dataset/` directory.
     - **PGN Files:** Located in `dataset/pgn/`.
     - **UCI Moves Dataset:** The initial extracted UCI moves are stored as `dataset/2_uci_moves_dataset.txt`.
     - **Enhanced Dataset:** After adding metadata and board state, the dataset is saved as `dataset/3_uci_moves_with_board_state_and_index.txt`.
   - **Move Indices:**
     - **Move Indices JSON:** Stored as `rules_based_mapping/chess_move_indices.json`.
     - **All Legal Moves JSON:** Stored as `rules_based_mapping/all_legal_moves.json`.
   - **Split Datasets:**
     - **Location:** All split files are stored within the `dataset/split/` directory.
     - **Naming Convention:** Each split file follows the pattern `castle_dataset_split_{n}.txt`, where `{n}` is a sequential integer starting from 1 (e.g., `castle_dataset_split_1.txt`, `castle_dataset_split_2.txt`, ..., `castle_dataset_split_100.txt`).
     - **Content Format:** Each split file contains up to 20,000 lines, with each line formatted as `move_index|<board state>|<rights>`, where:
       - **`move_index`**: An integer representing the index of the move within the `legal_moves` list.
       - **`<board state>`**: A comma-separated string of 64 integers representing the board state, where each integer corresponds to a specific piece or an empty square.
       - **`<rights>`**: A 6-character string encoding castling rights, en passant availability, and turn information.

5. **Data Loading for Training:**
   - The split files within `dataset/split/` are loaded sequentially during each epoch. For each chunk of 20,000 data points:
     - **In-Memory Expansion:** Subsequence generation is performed in-memory to create multiple training examples from each original game sequence.
     - **Batch Formation:** The expanded data is divided into batches of `BATCH_SIZE` (e.g., 64) for parallel training.
     - **Efficient Memory Usage:** By processing data in chunks and expanding them on-the-fly, the system avoids loading the entire dataset into memory, ensuring scalability and efficiency.

**Conclusion:**
This approach leverages the strengths of transformer architectures in handling sequential data and captures the intricate dependencies present in chess. By integrating both move sequences and explicit board state representations, the model is poised to learn complex patterns and make accurate move predictions. The detailed methods, move index encoding, and implementation provided offer a foundation for further research and development in AI-based chess modeling.

### Additional Notes:

- **Data Integrity:** Ensure that each split file (`castle_dataset_split_{n}.txt`) maintains the correct format and that all lines adhere to the `move_index|<board state>|<rights>` structure.
  
- **Scalability:** The splitting mechanism allows the system to handle datasets larger than the available memory by processing and loading data in manageable chunks.
  
- **Future Enhancements:** Consider automating the data splitting process further or integrating it into a pipeline that seamlessly feeds data into the model training process.

---

### Detailed Breakdown of Move Index Encoding

**Purpose:**
The move index encoding system is designed to map every possible legal chess move to a unique integer index. This numerical representation is crucial for efficiently training the transformer model, allowing it to process and predict moves based on their indices rather than raw UCI strings.

**Encoding Process:**

1. **Move Categorization:**
   - **Piece Types:** Moves are categorized based on the piece type and color, such as `white_pawn`, `black_rook`, etc.
   - **Special Moves:** Special moves like castling and pawn promotions are handled separately to ensure comprehensive coverage.

2. **Legal Move Generation:**
   - **Board Setup:** For each piece type and position on the board, the board is set up with only that piece placed on a specific square.
   - **Move Generation:** Using the `python-chess` library, all legal moves for the placed piece are generated.
   - **Promotion Moves:** For pawns on promotion ranks, all possible promotion options (queen, rook, bishop, knight) are considered, including capture promotions.

3. **Move Collection:**
   - **`moves` Dictionary:** Each piece type is mapped to a list of its possible UCI moves.
   - **`all_moves` Set:** A comprehensive set of all unique UCI moves across all piece types is maintained to ensure uniqueness.

4. **JSON Output:**
   - **`chess_move_indices.json`:** Contains a mapping from each piece type to its list of UCI moves.
   - **`all_legal_moves.json`:** A sorted list of all unique UCI moves, facilitating the creation of a consistent move index mapping.

**Implementation Highlights:**

- **Handling Castling:**
  - Castling moves (`e1g1`, `e1c1`, `e8g8`, `e8c8`) are manually added to the respective king's move list to ensure they are included.

- **Promotion Captures:**
  - Capture moves leading to pawn promotions are explicitly generated and added to the move lists to account for all possible promotion scenarios.

- **Data Verification:**
  - After move generation, the script verifies the inclusion of specific moves (e.g., `'b2a1q'`) to ensure completeness.

- **Move Counts:**
  - The script outputs the count of moves per piece type and the total number of unique moves, providing insights into the move index coverage.

**Usage in Data Preprocessing:**

- **Mapping Moves to Indices:**
  - The sorted list in `all_legal_moves.json` serves as the basis for assigning each move a unique integer index. For instance, the first move in the sorted list is assigned index `0`, the second move index `1`, `"a2a3"` index `0`, `"a2a4"` index `1`, and so forth, up to `4,139`.

- **Consistent Encoding:**
  - By storing the move indices in JSON files, the encoding and decoding processes remain consistent across different stages of data processing and model training.

**Example Entry in `all_legal_moves.json`:**

```json
[
    "a2a3",
    "a2a4",
    "a7a6",
    "a7a5",
    ...
]
```

In this example, `"a2a3"` is assigned index `0`, `"a2a4"` index `1`, `"a7a6"` index `2`, and so forth.

**Example Entry in `chess_move_indices.json`:**

```json
{
    "white_pawn": ["a2a3", "a2a4", "b2b3", "b2b4", ...],
    "black_rook": ["a8a6", "a8a5", ...],
    ...
}
```

Each piece type has its own list of possible moves, facilitating targeted move generation and indexing.

**Conclusion:**
The move index encoding system ensures that every legal chess move is uniquely and consistently represented by an integer index. This systematic encoding is fundamental for the transformer-based model to effectively learn and predict chess moves based on historical game data and current board states.
