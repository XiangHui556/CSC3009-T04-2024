import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_binary_mask(image_path, threshold_value=10):
    # Read the image
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create binary mask
    _, binary_mask = cv2.threshold(
        grayscale_image, threshold_value, 255, cv2.THRESH_BINARY
    )

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contours[0])
        for contour in contours:
            x0, y0, w0, h0 = cv2.boundingRect(contour)
            x = min(x, x0)
            y = min(y, y0)
            w = max(x + w, x0 + w0) - x
            h = max(y + h, y0 + h0) - y

        # Draw bounding box on original image
        image_with_bbox = image.copy()
        cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Crop the image using bounding box coordinates
        cropped_image = grayscale_image[y : y + h, x : x + w]
    else:
        image_with_bbox = image
        cropped_image = grayscale_image

    # Plotting the images
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Binary Mask")
    plt.imshow(binary_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Bounding Box")
    plt.imshow(cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Cropped Image")
    plt.imshow(cropped_image, cmap="gray")
    plt.axis("off")

    plt.show()


# Example usage
image_path = "../DATASETS/dataset_4/notumor/Te-no_0201.jpg"
visualize_binary_mask(image_path, threshold_value=10)
