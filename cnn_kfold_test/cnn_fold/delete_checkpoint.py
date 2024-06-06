import os

def delete_checkpoint_files(source_folder):
    # Iterate through all files and folders in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if the file matches the pattern "checkpoint.data-xxxxx"
            if file.startswith("checkpoint.data-"):
                # Construct the full path of the file
                file_path = os.path.join(root, file)
                # Delete the file
                os.remove(file_path)

# Replace 'source_folder_path' with the path to your folder containing subfolders
source_folder_path = './'

# Call the function to delete the files
delete_checkpoint_files(source_folder_path)