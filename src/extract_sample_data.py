#!/usr/bin/env python3
"""
Script to extract first 100 lines from each relations file to create sample data.
"""

import os
import shutil

def extract_first_lines(input_dir, output_dir, num_lines=100):
    """
    Extract first num_lines from each file in input_dir and save to output_dir.
    
    Args:
        input_dir (str): Path to directory containing original relation files
        output_dir (str): Path to directory where sample files will be saved
        num_lines (int): Number of lines to extract from each file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file in input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Only process files (not directories)
        if os.path.isfile(input_path):
            print(f"Processing {filename}...")
            with open(input_path, 'r', encoding='utf-8') as infile, \
                 open(output_path, 'w', encoding='utf-8') as outfile:
                # Read and write first num_lines
                for i, line in enumerate(infile):
                    if i >= num_lines:
                        break
                    outfile.write(line)
            print(f"Saved first {num_lines} lines to {output_path}")

if __name__ == "__main__":
    # Define paths
    input_directory = "../../relations"  # Relative path to original relations directory
    output_directory = "../sample_data"  # Relative path to sample data directory
    
    # Extract first 100 lines from each file
    extract_first_lines(input_directory, output_directory, 100)
    print("Sample data extraction completed.")