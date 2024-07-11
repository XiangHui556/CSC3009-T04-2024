import tensorflow as tf
import json
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import ParameterGrid

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Define a function to build the model for hyperparameter tuning
def build_VGG16_model(hp):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    # Freeze the base model layers
    base_model.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())

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


# Define the paths
data_dir = "../DATASETS/kfold_dataset_128"  # Directory containing train, val, and test directories

test_dir = "../DATASETS/resized_dataset_128/test"
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=12,
    class_mode="categorical",
)


def unfreeze_layers(base_model, num_layers):
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True


# Iterate over the splits
for i in range(6):
    print(f"Fold {i + 1}")
    train_dir = data_dir + f"/train_fold_{i}"
    val_dir = data_dir + f"/val_fold_{i}"

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

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=18,
        class_mode="categorical",
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(128, 128),
        batch_size=18,
        class_mode="categorical",
    )

    # Setup tuner
    tuner = RandomSearch(
        build_VGG16_model,
        objective="val_accuracy",
        max_trials=12,  # Maximum number of trials to run
        executions_per_trial=1,  # Number of executions per trial
        directory=f"./cnn_fold_tuner_logs/cnn_tuner_logs_fold_{i}",  # Directory to save results
        project_name=f"cnn_hyperparam_tuning_fold_{i}",
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

    # Define variables to keep track of the best model
    best_model_tune2 = None
    best_accuracy = 0
    best_loss = 0.0


    param_grid = {
        "learning_rate": [0.001, 0.0001, 0.01, 0.00001],
        "unfreeze_layers": [5, 10, 20, 30],
    }

    for params in ParameterGrid(param_grid):
        # Get the best model
        best_model = tuner.get_best_models(num_models=1)[0]
        learning_rate = params["learning_rate"]
        unfreeze_layer = params["unfreeze_layers"]
        print(f"Unfreezing the last {unfreeze_layer} layers")

        unfreeze_layers(best_model, unfreeze_layer)

        # Compile the model
        best_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train the model
        history = best_model.fit(
            train_generator,
            epochs=1000,
            validation_data=val_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_steps=val_generator.samples // val_generator.batch_size,
            callbacks=[early_stopping],
        )

        # Evaluate the model on the test set
        test_loss, test_accuracy = best_model.evaluate(test_generator)
        print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

        # Check if this model is better than the previous best model
        if test_accuracy >= best_accuracy:
            if test_accuracy == best_accuracy:
                if test_loss < best_loss:
                    best_accuracy = test_accuracy
                    best_loss = test_loss
                    best_model_tune2 = best_model
            else:
                best_model_tune2 = best_model
                best_accuracy = test_accuracy
                best_loss = test_loss

    # Print the accuracy of the best model
    print(f"Best model accuracy: {best_accuracy}")

    # Evaluate the best model on the test set
    test_loss, test_accuracy = best_model_tune2.evaluate(test_generator)
    print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

    # Save the model with accuracy and loss in the name
    model_name = (
        f"cnn_mri_classifier_acc_{test_accuracy:.3f}_loss_{test_loss:.3f}_top_{i+1}.h5"
    )

    best_model_tune2.save(model_name)

    # Save the best hyperparameters
    with open(f"best_hyperparameters_fold_{i}.json", "w") as f:
        json.dump(tuner.oracle.get_best_trials(1)[0].hyperparameters.values, f)
        json.dump(params, f)
