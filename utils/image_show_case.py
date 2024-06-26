import os
import cv2
import matplotlib.pyplot as plt
from get_categories import get_categories


def show_sample_images(data_dir, categories):
    fig, axes = plt.subplots(1, len(categories), figsize=(15, 10))
    for ax, category in zip(axes, categories):
        category_path = os.path.join(data_dir, category)
        sample_image_path = os.path.join(category_path, os.listdir(category_path)[33])
        sample_image = cv2.imread(sample_image_path)
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
        ax.imshow(sample_image)
        ax.set_title(category)
        ax.axis("off")
    plt.show()


data_dir = "../DATASETS/dataset_4"
categories = get_categories(data_dir)
show_sample_images(data_dir, categories)
