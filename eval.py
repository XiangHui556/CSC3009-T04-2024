from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the saved model
mode_name = "inceptionResNetV2_model.h5"
model = load_model(mode_name)

# Recompile the model with the correct metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(),  # or any other optimizer you want to use
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Define the paths
test_dir = "./DATASETS/dataset_7_256"

# ImageDataGenerator for normalization
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Data generator for test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(
        256,
        256,
    ),  # Adjust if your model was trained with a different target size
    batch_size=12,  # Adjust based on your batch size during training
    class_mode="categorical",
    shuffle=False,
)
categories = ["glioma", "meningioma", "pituitary", "notumor"]

# Predict the labels for the test set
y_pred = model.predict(test_generator)
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
plt.show()
