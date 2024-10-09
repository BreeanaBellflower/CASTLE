import os
from tqdm import tqdm

def split_large_dataset(
    input_file_path,
    output_dir,
    lines_per_file=100000,
    output_filename_pattern="castle_dataset_split_{n}.txt"
):
    """
    Splits a large text file into smaller chunks, each containing a specified number of lines.
    
    Parameters:
    - input_file_path: Path to the input large text file.
    - output_dir: Directory where the split files will be saved.
    - lines_per_file: Number of lines each split file should contain.
    - output_filename_pattern: Naming pattern for the split files. Use '{n}' as a placeholder for the file index.
    
    Example:
    split_large_dataset(
        input_file_path='dataset/3_uci_moves_with_board_state_and_index.txt',
        output_dir='dataset/train/',
        lines_per_file=100000,
        output_filename_pattern="castle_dataset_split_{n}.txt"
    )
    """
    
    # Check if the input file exists
    if not os.path.isfile(input_file_path):
        print(f"Input file '{input_file_path}' does not exist.")
        return
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # First, count the total number of lines for the progress bar
    print("Counting total number of lines in the input file...")
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)
    
    print(f"Total lines to process: {total_lines}")
    
    # Initialize variables
    file_index = 1
    current_line = 0
    current_file = None
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            with tqdm(total=total_lines, desc="Splitting File", unit="lines") as pbar:
                for line in infile:
                    # If this is the first line or we've reached the lines_per_file limit, open a new file
                    if current_line % lines_per_file == 0:
                        if current_file:
                            current_file.close()
                        output_filename = output_filename_pattern.format(n=file_index)
                        output_path = os.path.join(output_dir, output_filename)
                        current_file = open(output_path, 'w', encoding='utf-8')
                        file_index += 1
                        print(f"Creating file: {output_path}")
                    
                    # Write the current line to the current output file
                    current_file.write(line)
                    current_line += 1
                    pbar.update(1)
        
        # Close the last file if it's open
        if current_file:
            current_file.close()
    
    except Exception as e:
        print(f"An error occurred during splitting: {e}")
        if current_file:
            current_file.close()
        return
    
    print(f"Splitting complete. {file_index - 1} files created in '{output_dir}'.")

if __name__ == '__main__':
    # Define input and output paths
    input_file = 'dataset/3_uci_moves_with_board_state_and_index.txt'
    output_directory = 'dataset/train/'
    
    # Call the splitting function
    split_large_dataset(
        input_file_path=input_file,
        output_dir=output_directory,
        lines_per_file=20000,
        output_filename_pattern="castle_dataset_split_{n}.txt"
    )
