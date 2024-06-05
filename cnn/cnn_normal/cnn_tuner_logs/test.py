import os


# Delete tuning checkpoint files to save disk space
def delete_checkpoint_files(source_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.startswith("checkpoint.data-"):
                file_path = os.path.join(root, file)
                os.remove(file_path)


source_folder_path = "cnn_hyperparam_tuning"
delete_checkpoint_files(source_folder_path)
