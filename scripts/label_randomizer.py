import os
import random

def create_labels_file(data_dir, output_file):
    # Get all .py files
    sequence_files = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.py')]
    # Create labels file
    with open(output_file, 'w') as f:
        for seq_id in sequence_files:
            # Randomly assign 0 or 1
            label = random.randint(0, 1)
            f.write(f"{seq_id},{label}\n")
          
# Create train labels
create_labels_file('data/train', 'data/train/labels.txt')
# Create test labels
create_labels_file('data/heldout', 'data/heldout/labels.txt')
