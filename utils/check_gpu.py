import tensorflow as tf

print("TensorFlow version:", tf.__version__)

print(tf.config.list_physical_devices("GPU"))

# Check if GPU is available
if tf.test.is_gpu_available():
    print("GPU is available")
    # Check if CUDA is installed
    if tf.test.is_built_with_cuda():
        print("CUDA is installed")
    else:
        print("CUDA is not installed")
else:
    print("GPU is not available")
