import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=128)],
            )
    except RuntimeError as e:
        print(e)

# Load the saved model
model = load_model("cnn_mri_classifier_acc_0.833_loss_0.482_top_1.h5")

# Recompile the model with the correct metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(),  # or any other optimizer you want to use
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Define the paths
test_dir = "resized_dataset/test"

# ImageDataGenerator for normalization
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Data generator for test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(
        128,
        128,
    ),  # Adjust if your model was trained with a different target size
    batch_size=12,  # Adjust based on your batch size during training
    class_mode="categorical",
    shuffle=False,
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_generator)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")

# Make predictions (optional)
predictions = model.predict(test_generator)
