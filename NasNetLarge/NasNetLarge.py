import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dropout,
    Dense,
    GlobalAveragePooling2D,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define paths
data_dir = "../split_dataset"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

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

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Generators for training and validation and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(331, 331),
    batch_size=32,
    class_mode="categorical",
)

val_generator = valid_datagen.flow_from_directory(
    val_dir,
    target_size=(331, 331),
    batch_size=32,
    class_mode="categorical",
)

test_generator = valid_datagen.flow_from_directory(
    test_dir,
    target_size=(331, 331),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)

# Load NASNetLarge model without top layers
base_model = NASNetLarge(
    weights="imagenet", include_top=False, input_shape=(331, 331, 3)
)

# Freeze all layers of NASNetLarge
base_model.trainable = False

# Building Model with Dropout and Batch Normalization
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(4, activation="softmax"))

# Model Summary
model.summary()

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Define callbacks
checkpoint_callback = ModelCheckpoint(
    filepath="best_model.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

early_stopping_callback = EarlyStopping(
    monitor="val_accuracy",  # Monitor validation accuracy
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Restore weights to the best observed during training
)

# Train the model with callbacks to save the best model based on validation accuracy
history = model.fit(
    train_generator,
    epochs=100, 
    validation_data=val_generator,
    callbacks=[checkpoint_callback, early_stopping_callback],
    verbose=1,
)

# Load the best saved model
best_model = load_model("best_model.h5")

# Print and save best results
best_params = best_model.optimizer.get_config()
best_loss, best_accuracy = best_model.evaluate(val_generator)
print(f"Best Parameters: {best_params}")
print(f"Val Accuracy: {best_accuracy}")
print(f"Val Loss: {best_loss}")

# Plot best model training history
best_history = history.history
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(best_history["accuracy"], label="Training Accuracy")
plt.plot(best_history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_history["loss"], label="Training Loss")
plt.plot(best_history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
# Save the plot as an image
plt.savefig("training_validation_accuracy_loss.png")

# Evaluate on the test set
test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predict the labels for the test set
categories = ["glioma", "meningioma", "pituitary", "notumor"]
y_pred = best_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Compute the classification report
report = classification_report(y_true, y_pred_classes, target_names=categories)
print("Classification Report:\n")
print(report)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=categories,
    yticklabels=categories,
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
