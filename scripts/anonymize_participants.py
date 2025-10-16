#!/usr/bin/env python3
"""
Anonymize participant filenames in user_experiments/results
Renames files like 'ParticipantName.csv' to 'P001.csv'
Creates a mapping file for reference.
"""

import os
import sys

# Get script directory and navigate to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
results_dir = os.path.join(project_root, 'src', 'user_experiments', 'results')

def anonymize_filenames():
    # Get all CSV files
    files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    files.sort()  # Sort alphabetically for consistent ordering

    # Create mapping
    mapping = {}
    for i, filename in enumerate(files, start=1):
        new_name = f'P{i:03d}.csv'
        mapping[filename] = new_name

    # Save mapping file
    mapping_file = os.path.join(project_root, 'participant_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("Original -> Anonymized\n")
        f.write("=" * 40 + "\n")
        for original, anonymized in mapping.items():
            f.write(f"{original} -> {anonymized}\n")

    print(f"Mapping saved to: {mapping_file}")
    print(f"\nWill rename {len(mapping)} files:")
    print("-" * 40)
    for original, anonymized in mapping.items():
        print(f"{original} -> {anonymized}")

    # Ask for confirmation
    print("\n" + "=" * 40)
    response = input("Proceed with renaming? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        for original, anonymized in mapping.items():
            old_path = os.path.join(results_dir, original)
            new_path = os.path.join(results_dir, anonymized)
            os.rename(old_path, new_path)
        print(f"\nSuccessfully renamed {len(mapping)} files!")
    else:
        print("\nAborted. No files were renamed.")

if __name__ == '__main__':
    anonymize_filenames()
