import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Flatten,
    Dense,
    Input,
)

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


def unet_for_classification(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, kernel_size=3, activation="relu", padding="same")(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    conv3 = Conv2D(128, kernel_size=3, activation="relu", padding="same")(pool2)
    up1 = UpSampling2D(size=(2, 2))(conv3)
    concat1 = Concatenate()([conv2, up1])
    conv4 = Conv2D(64, kernel_size=3, activation="relu", padding="same")(concat1)
    up2 = UpSampling2D(size=(2, 2))(conv4)
    concat2 = Concatenate()([conv1, up2])

    # Classification
    flatten = Flatten()(concat2)
    dense1 = Dense(256, activation="relu")(flatten)
    output = Dense(num_classes, activation="softmax")(dense1)

    model = Model(inputs=inputs, outputs=output)
    return model


# Define the paths
train_dir = "resized_dataset2/train"
val_dir = "resized_dataset2/val"
test_dir = "resized_dataset2/test"

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

image_size = 64

# Data generators
# Batch size is set to 12, due to the dataset split amount, it is a multiple of 12
# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=1,
    class_mode="categorical",
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(image_size, image_size),
    batch_size=1,
    class_mode="categorical",
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=1,
    class_mode="categorical",
)

# Example usage
input_shape = (image_size, image_size, 1)
model = unet_for_classification(input_shape, 4)
# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Model summary
model.summary()

# Assuming you have train_images and train_labels
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10,
)

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
