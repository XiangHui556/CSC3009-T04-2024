import os
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold


def create_kfold_splits(data_dir, n_splits):
    # Get the list of subdirectories (class labels)
    class_labels = sorted(os.listdir(data_dir))

    # Create a list to hold the file paths and corresponding labels
    file_paths = []
    labels = []

    # Iterate over each class label directory
    for i, label in enumerate(class_labels):
        label_dir = os.path.join(data_dir, label)
        # Get the list of file paths in the current class label directory
        files = [os.path.join(label_dir, f) for f in os.listdir(label_dir)]
        # Add file paths and corresponding labels to the lists
        file_paths.extend(files)
        labels.extend([i] * len(files))

    # Convert lists to arrays
    file_paths = np.array(file_paths)
    labels = np.array(labels)

    # Initialize StratifiedKFold with specified number of splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Iterate over the splits
    for fold, (train_index, val_index) in enumerate(skf.split(file_paths, labels)):
        # Create directories for train and validation sets for the current fold
        train_dir = os.path.join(data_dir, f"train_fold_{fold}")
        val_dir = os.path.join(data_dir, f"val_fold_{fold}")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Copy files to train and validation directories, preserving subdirectories
        for idx in train_index:
            src_file = file_paths[idx]
            dest_file = os.path.join(train_dir, os.path.relpath(src_file, data_dir))
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            shutil.copy(src_file, dest_file)
        for idx in val_index:
            src_file = file_paths[idx]
            dest_file = os.path.join(val_dir, os.path.relpath(src_file, data_dir))
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            shutil.copy(src_file, dest_file)


# Example usage
data_dir = "kfold_dataset_128"
n_splits = 6
create_kfold_splits(data_dir, n_splits)
