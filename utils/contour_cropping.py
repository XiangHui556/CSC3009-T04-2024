import cv2
import numpy as np
import os


def crop_black_borders(image, threshold_value=10):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours or bounding box of non-black regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contours[0])
        for contour in contours:
            x0, y0, w0, h0 = cv2.boundingRect(contour)
            x = min(x, x0)
            y = min(y, y0)
            w = max(x + w, x0 + w0) - x
            h = max(y + h, y0 + h0) - y

        # Crop the image using bounding box coordinates
        cropped_image = image[y : y + h, x : x + w]
    else:
        cropped_image = image

    return cropped_image


def make_square(image):
    h, w = image.shape[:2]
    if h == w:
        return image
    if h > w:
        pad_width = (h - w) // 2
        padded_image = cv2.copyMakeBorder(
            image,
            0,
            0,
            pad_width,
            h - w - pad_width,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
    else:
        pad_height = (w - h) // 2
        padded_image = cv2.copyMakeBorder(
            image,
            pad_height,
            w - h - pad_height,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

    return padded_image


# def process_images(input_dir, output_dir, threshold_value=10):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for filename in os.listdir(input_dir):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(input_dir, filename)
#             image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#             cropped_image = crop_black_borders(image, threshold_value)
#             output_path = os.path.join(output_dir, filename)
#             cv2.imwrite(output_path, cropped_image)

# # Example usage
# input_dir = '/test'
# output_dir = '/out'
# process_images(input_dir, output_dir, threshold_value=10)
