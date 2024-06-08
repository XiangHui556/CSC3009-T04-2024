import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

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

# Define the paths
train_dir = "../DATASETS/resized_dataset_128/train"
val_dir = "../DATASETS/resized_dataset_128/val"

# ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Data generators
# Batch size is set to 12, due to the dataset split amount, it is a multiple of 12
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(512, 512), batch_size=12, class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(512, 512), batch_size=12, class_mode="categorical"
)

# Define the CNN model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(512, 512, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(4, activation="softmax"),  # Assuming 4 categories
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10,
)

# Save the trained model to a file, for ease of use
model.save("cnn_mri_classifier.h5")
