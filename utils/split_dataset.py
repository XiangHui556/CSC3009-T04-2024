import shutil
import os
from sklearn.model_selection import train_test_split
from utils.get_categories import get_categories

# Split data into train, validation, and test sets, with 70%, 20%, and 10% respectively
def split_data(data_dir, train_dir, val_dir, test_dir, val_ratio=0.2, test_ratio=0.1):
    categories = os.listdir(data_dir)
    for category in categories:
        category_path = os.path.join(data_dir, category)
        images = os.listdir(category_path)
        train_and_val, test = train_test_split(
            images, test_size=test_ratio, random_state=42
        )
        train, val = train_test_split(
            train_and_val, test_size=val_ratio / (1 - test_ratio), random_state=42
        )

        for image in train:
            shutil.move(
                os.path.join(category_path, image),
                os.path.join(train_dir, category, image),
            )
        for image in val:
            shutil.move(
                os.path.join(category_path, image),
                os.path.join(val_dir, category, image),
            )
        for image in test:
            shutil.move(
                os.path.join(category_path, image),
                os.path.join(test_dir, category, image),
            )


data_dir = "./dataset_4"
train_dir = "split_dataset/train"
val_dir = "split_dataset/val"
test_dir = "split_dataset/test"

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

categories = get_categories(data_dir)
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

split_data(data_dir, train_dir, val_dir, test_dir)
