import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)

# Define ImageDataGenerator with transformations
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)


# Function to plot images
def plot_images(original, transformed):
    plt.figure(figsize=(12, 6))

    for i in range(4):
        # Original images
        plt.subplot(2, 4, i + 1)
        plt.imshow(original[i], cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Transformed images
        plt.subplot(2, 4, i + 5)
        plt.imshow(transformed[i], cmap="gray")
        plt.title("Transformed")
        plt.axis("off")

    plt.show()


# Load sample images
sample_images = []
image_paths = [
    "../DATASETS/resized_dataset_256/train/glioma/Tr-gl_0025.jpg",
    "../DATASETS/resized_dataset_256/train/meningioma/Te-me_0171.jpg",
    "../DATASETS/resized_dataset_256/train/notumor/Te-no_0201.jpg",
    "../DATASETS/resized_dataset_256/train/pituitary/Te-pi_0196.jpg",
]

for image_path in image_paths:
    img = load_img(image_path, target_size=(256, 256), color_mode="grayscale")
    img_array = img_to_array(img)
    sample_images.append(img_array)

# Convert to numpy array
sample_images = np.array(sample_images)

# Generate transformed images
transformed_images = []
for img in sample_images:
    img = np.expand_dims(img, axis=0)
    iterator = datagen.flow(img, batch_size=1)
    transformed_img = iterator.next()[0]
    transformed_images.append(transformed_img)

# Convert to numpy array
transformed_images = np.array(transformed_images)

# Plot the images
plot_images(sample_images, transformed_images)
