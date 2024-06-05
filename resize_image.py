import cv2
import os


def resize_image(image, size=(512, 512)):
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized_image


def resize_dataset(dataset_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each image in the dataset directory
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            # Check if the file is an image (you can add more file extensions if needed)
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                # Read the image
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                # Resize the image
                resized_image = resize_image(image, size=(128, 128))

                # Get the relative path of the image from the dataset directory
                relative_path = os.path.relpath(root, dataset_dir)

                # Create the corresponding output subdirectory in the output directory
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                # Save the resized image to the output directory with the original filename
                output_path = os.path.join(output_subdir, file)
                cv2.imwrite(output_path, resized_image)


# Define the paths
dataset_dir = "split_dataset"
output_dir = "resized_dataset"

# Resize the dataset
resize_dataset(dataset_dir, output_dir)
