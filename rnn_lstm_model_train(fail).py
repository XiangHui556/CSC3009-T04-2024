import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Conv2D,
    MaxPooling2D,
    Flatten,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

sequence_length = 12
image_height = 128
image_width = 128
num_channels = 3

# Define model parameters
input_shape = (
    sequence_length,
    image_height,
    image_width,
    num_channels,
)  # Define input shape for RNN

num_classes = 4  # Number of output classes
learning_rate = 0.001
batch_size = 32
epochs = 20

# Define RNN model
model = Sequential(
    [
        TimeDistributed(
            Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            input_shape=input_shape,
        ),
        TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
        TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation="relu")),
        TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
        TimeDistributed(Flatten()),
        LSTM(128),  # You can adjust the number of LSTM units as needed
        Dense(num_classes, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Print model summary
model.summary()

# Define the paths
train_dir = "resized_dataset/train"
val_dir = "resized_dataset/val"
test_dir = "resized_dataset/test"

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
    train_dir, target_size=(128, 128), batch_size=12, class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=12, class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(128, 128), batch_size=12, class_mode="categorical"
)


# Define callbacks (optional)
callbacks = [
    ModelCheckpoint(
        filepath="rnn_model.h5", save_best_only=True
    ),  # Save the best model during training
    EarlyStopping(
        patience=5, restore_best_weights=True
    ),  # Stop training early if validation loss stops improving
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=epochs,
    callbacks=callbacks,
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
