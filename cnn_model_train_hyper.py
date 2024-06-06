import tensorflow as tf
import json
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping


print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define the paths
train_dir = "resized_dataset_128/train"
val_dir = "resized_dataset_128/val"
test_dir = "resized_dataset_128/test"

# Define image statistics
image_size = 128
image_color = 3


# Define a function to build the model for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(
        Conv2D(
            hp.Int("conv1_units", min_value=32, max_value=256, step=32),
            (3, 3),
            activation="relu",
            input_shape=(image_size, image_size, image_color),  # 1 for grayscale, 3 for RGB
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Additional convolutional layers
    for i in range(hp.Int("conv_layers", min_value=1, max_value=3)):
        model.add(
            Conv2D(
                hp.Int(f"conv_units_{i}", min_value=32, max_value=256, step=32),
                (3, 3),
                activation="relu",
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Dense layers
    for i in range(hp.Int("dense_layers", min_value=1, max_value=3)):
        model.add(
            Dense(
                hp.Int(f"dense_units_{i}", min_value=128, max_value=1024, step=128),
                activation="relu",
            )
        )
        model.add(Dropout(0.5))

    model.add(Dense(4, activation="softmax"))  # 4 categories

    # Define learning rate
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


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
    train_dir,
    target_size=(image_size, image_size),
    batch_size=28,
    class_mode="categorical",
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(image_size, image_size),
    batch_size=12,
    class_mode="categorical",
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=12,
    class_mode="categorical",
)


# Setup tuner
tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=20,  # Maximum number of trials to run
    executions_per_trial=1,  # Number of executions per trial
    directory="cnn_tuner_logs",  # Directory to save results
    project_name="cnn_hyperparam_tuning",
)

# Used to stop training if there is no improvement in the validation loss, improves training speed
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=25,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,
)

# Search for the best hyperparameter configuration
tuner.search(
    train_generator,
    epochs=1000,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stopping],
)

# Get the best model
best_models = tuner.get_best_models(num_models=3)

# Get the best hyperparameters
best_hyperparameters = tuner.oracle.get_best_trials(1)[0].hyperparameters.values

# Save the best hyperparameters to a JSON file
with open("best_hyperparameters.json", "w") as f:
    json.dump(best_hyperparameters, f)

# Evaluate the best model on the test set
for model in best_models:
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

# Save the top 3 best models
for i, model in enumerate(best_models):
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # Save the training history
    try:
        history = model.history.history
        with open(f"training_history_fold_{i}.pkl", "wb") as f:
            pickle.dump(history, f)
    except:
        pass

    # Save the model with accuracy and loss in the name
    model_name = f"cnn_mri_classifier_acc_{accuracy:.3f}_loss_{loss:.3f}_top_{i+1}.h5"

    # Include the best hyperparameters in the model name
    model_name_with_hyperparameters = f"{model_name[:-3]}_hyperparameters.json"
    with open(model_name_with_hyperparameters, "w") as f:
        json.dump(best_hyperparameters, f)

    model.save(model_name)
