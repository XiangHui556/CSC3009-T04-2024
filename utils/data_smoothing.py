import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your MRI image
image_path = "../DATASETS/dataset_4/glioma/Te-gl_0101.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply GaussianBlur with a specified kernel size and standard deviation (sigma)
smoothed_image1 = cv2.GaussianBlur(image, (7, 7), sigmaX=1.5)
smoothed_image2 = cv2.GaussianBlur(image, (5, 5), sigmaX=1)
smoothed_image3 = cv2.GaussianBlur(image, (3, 3), sigmaX=0.5)

# Display the original image and the smoothed images using matplotlib
plt.figure(figsize=(15, 10))

# Plot the original image
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original MRI Image")
plt.axis("off")

# Plot the first smoothed image
plt.subplot(2, 2, 2)
plt.imshow(smoothed_image1, cmap="gray")
plt.title("Smoothed Image (3x3 kernel, sigma=0.5)")
plt.axis("off")

# Plot the second smoothed image
plt.subplot(2, 2, 3)
plt.imshow(smoothed_image2, cmap="gray")
plt.title("Smoothed Image (5x5 kernel, sigma=1.0)")
plt.axis("off")

# Plot the third smoothed image
plt.subplot(2, 2, 4)
plt.imshow(smoothed_image3, cmap="gray")
plt.title("Smoothed Image (7x7 kernel, sigma=1.5)")
plt.axis("off")

plt.tight_layout()
plt.show()
