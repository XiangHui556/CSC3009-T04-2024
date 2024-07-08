import os
import cv2
from utils.contour_cropping import crop_black_borders
from utils.contour_cropping import make_square
from utils.resize_image import resize_and_convert_to_grayscale

def process_images(input_dir, output_dir, threshold_value=10, resize_value=512):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each image in the dataset directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check if the file is an image (you can add more file extensions if needed)
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                # Read the image
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Crop, Resize and convert to grayscale
                cropped_image = crop_black_borders(image, threshold_value)
                square_image = make_square(cropped_image)
                resized_image = resize_and_convert_to_grayscale(
                    square_image, resize_value
                )

                # Get the relative path of the image from the dataset directory
                relative_path = os.path.relpath(root, input_dir)

                # Create the corresponding output subdirectory in the output directory
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                # Save the resized and converted image to the output directory with the original filename
                output_path = os.path.join(output_subdir, file)
                cv2.imwrite(output_path, resized_image)


# Example usage
input_dir = "dataset_4"
output_dir = "dataset_4_256"
resize_value = 256
process_images(input_dir, output_dir, threshold_value=10, resize_value=resize_value)
