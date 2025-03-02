from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from get_datasets import get_datasets
import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt

# Display all rows and columns in pandas
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Dataset paths
train_path_2 = "dataset2/Data/train"
val_path_2 = "chest-xray-pneumonia/chest_xray/val"  # Keeping this the same, as new set has no validation folder
test_path_2 = "dataset2/Data/test"

# Load datasets
train_dataset_2, _, test_dataset_2 = get_datasets(train_path_2, val_path_2, test_path_2)

# Load the pre-trained model
model = load_model("updated_model.h5")

# Predict on the test dataset
y_pred_prob = model.predict(test_dataset_2)
y_pred = (y_pred_prob > 0.5).astype(int)

# Extract the true labels
y_true = np.concatenate([label for _, label in test_dataset_2], axis=0)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
disp.plot(cmap='Blues')
plt.show()

# Calculate Precision and Recall
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
