import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import math
import os

from torch import nn, optim
from sklearn.model_selection import ParameterGrid

# Define path to your dataset
data_dir_train = (
    "/content/drive/MyDrive/Colab Notebooks/MRI_Images/resized_dataset_128/train"
)
data_dir_val = (
    "/content/drive/MyDrive/Colab Notebooks/MRI_Images/resized_dataset_128/val"
)

# Initialize lists to store image paths and corresponding labels
image_paths_train = []
labels_train = []

image_paths_val = []
labels_val = []

# Define a dictionary to map folder names to class labels
class_map = {"glioma": 0, "meningioma": 1, "pituitary": 2, "notumor": 3}

# Traverse the root directory and its subdirectories
for root, dirs, files in os.walk(data_dir_train):
    # Iterate over files in each directory
    for file in files:
        # Check if the file is an image file
        if file.endswith(".jpg") or file.endswith(".png"):
            # Get the label from the parent directory
            label = os.path.basename(root)
            # Map the label to its corresponding class index
            label_index = class_map[label]
            # Construct the image path
            image_path = os.path.join(root, file)
            # Add the image path and label to the lists
            image_paths_train.append(image_path)
            labels_train.append(label_index)

# Traverse the root directory and its subdirectories
for root, dirs, files in os.walk(data_dir_val):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            label = os.path.basename(root)
            label_index = class_map[label]
            image_path = os.path.join(root, file)
            image_paths_val.append(image_path)
            labels_val.append(label_index)


# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


# Example usage:

# Define transformations for image preprocessing (e.g., resizing, normalization)
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # Resize images to 28x28 (adjust as needed)
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,)),  # Normalize images for grayscale
    ]
)

batch_size = 12


# Create dataset and data loaders
dataset_train = CustomDataset(image_paths_train, labels_train, transform=transform)
dataset_val = CustomDataset(image_paths_val, labels_val, transform=transform)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

# Check if a GPU is available and set the device accordingly
device = T.device("cuda" if T.cuda.is_available() else "cpu")

# Output the device being used
print(f"Using device: {device}")


class LSTM(nn.Module):
    def __init__(self, input_len, hidden_size, num_classes, n_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_len, hidden_size, n_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        hidden_states = torch.zeros(self.n_layers, X.size(0), self.hidden_size).to(
            device
        )
        cell_states = torch.zeros(self.n_layers, X.size(0), self.hidden_size).to(device)
        output, _ = self.lstm(X, (hidden_states, cell_states))
        output = self.output_layer(output[:, -1, :])
        return output


results = []

# Hyperparameter tuning
param_grid = {
    "learning_rate": [0.001, 0.0001, 0.01],
    "hidden_layers": [128, 256],
    "n_layers": [1, 2, 3],
}

best_val_accuracy = 0
best_params = {}
best_model_state = None

for params in ParameterGrid(param_grid):
    learning_rate = params["learning_rate"]
    hidden_layers = params["hidden_layers"]
    n_layers = params["n_layers"]

    lstm_class_model = LSTM(128, hidden_layers, 4, n_layers).to(device)

    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(lstm_class_model.parameters(), lr=learning_rate)

    max_epoch = 100  # Use a smaller number of epochs for grid search
    best_val_loss = float("inf")

    for epoch in range(max_epoch):
        ep_loss = 0
        lstm_class_model.train()  # Ensure model is in training mode
        for batch, (image, label) in enumerate(train_loader):
            image = image.reshape(-1, 128, 128).to(
                device
            )  # reshape the image so that it can fit into the LSTM model
            label = label.to(device)
            out = lstm_class_model(image)  # LSTM model expects 3 dimension input

            lossfunction = loss(out, label)

            ep_loss += lossfunction.item()  # summing losses

            opt.zero_grad()
            lossfunction.backward()
            opt.step()

        # Validation
        lstm_class_model.eval()  # set model to evaluation mode
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_image, val_label in test_loader:
                val_image = val_image.reshape(-1, 128, 128).to(device)
                val_label = val_label.to(device)
                val_out = lstm_class_model(val_image)
                val_loss += loss(val_out, val_label).item()

                _, predicted = torch.max(val_out.data, 1)
                total += val_label.size(0)
                correct += (predicted == val_label).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = correct / total

        if val_accuracy > best_val_loss:
            best_val_loss = val_accuracy
            best_params = params
            best_model_state = lstm_class_model.state_dict()

        print(
            f"Epoch {epoch + 1}, LR: {learning_rate}, hidden_layers: {hidden_layers}, n_layers: {n_layers}, Validation Accuracy: {val_accuracy * 100:.2f}%, Validation Loss: {avg_val_loss:.6f}"
        )

    results.append(
        {"params": params, "val_accuracy": val_accuracy, "val_loss": avg_val_loss}
    )

# Sort results by validation accuracy
results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)

# Print top 3 hyperparameter combinations
print("Top 3 Hyperparameter Combinations:")
for i in range(3):
    print(f"Rank {i+1}:")
    print(f"Params: {results[i]['params']}")
    print(f"Validation Accuracy: {results[i]['val_accuracy'] * 100:.2f}%")
    print(f"Validation Loss: {results[i]['val_loss']:.6f}")

print("Best Validation Accuracy: {:.2f}%".format(results[0]["val_accuracy"] * 100))
print("Best Hyperparameters: ", results[0]["params"])

# Save the best model state
if best_model_state is not None:
    torch.save(
        best_model_state,
        "/content/drive/MyDrive/Colab Notebooks/MRI_Images/best_lstm_model.pth",
    )
