import subprocess

# List of scripts to run sequentially
SCRIPTS = [
    "./densenet121_cnn/cnn_densenet121_model_train_hyper_kfold.py",
    "./resnet50_cnn/cnn_resnet50_model_train_hyper_kfold.py",
    "./inceptionV3_cnn/cnn_inceptionV3_model_train_hyper_kfold.py",
    "./vgg16_cnn/cnn_vgg16_model_train_hyper_kfold.py",
]

# Loop through each script
for script in SCRIPTS:
    print(f"Running {script}...")

    # Run the script using subprocess
    process = subprocess.Popen(
        ["python", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()  # Capture stdout and stderr

    # Print the stdout and stderr
    if stdout:
        print("STDOUT:")
        print(stdout.decode())  # Decode bytes to string
    if stderr:
        print("STDERR:")
        print(stderr.decode())  # Decode bytes to string

    # Check the exit status
    if process.returncode == 0:
        print(f"{script} completed successfully.")
    else:
        print(f"Error: {script} failed.")
        exit(1)

print("All scripts completed successfully.")
