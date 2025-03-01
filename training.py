import os
from simple_model import simple_model, train, class_weights
from keras.models import load_model
from get_datasets import get_datasets
from keras.optimizers import Adam

# Initialize the base model with a simple CNN architecture
untrain_model, call_backs = simple_model()

# Define paths to training, validation, and test datasets
train_path = "chest-xray-pneumonia/chest_xray/train"
val_path = "chest-xray-pneumonia/chest_xray/val"
test_path = "chest-xray-pneumonia/chest_xray/test"

# Load the datasets
# train_dataset, val_dataset, test_dataset = get_datasets(train_path, val_path, test_path)

# Calculate class weights to address class imbalance
# class_weights_dict = class_weights(train_path, train_dataset)

# Set a lower learning rate for fine-tuning
new_learning_rate = 1e-5  # Lower learning rate for improved training
optimizer = Adam(learning_rate=new_learning_rate)

# Load the previously trained model for improvement (fine-tuning)
model = load_model('Model.h5')

# Define paths for a new dataset (dataset2)
train_path_2 = "dataset2/Data/train"
val_path_2 = "chest-xray-pneumonia/chest_xray/val"  # Using the same validation set, no changes needed
test_path_2 = "dataset2/Data/test"

# Load new dataset
train_dataset_2, val_dataset_2, test_dataset_2 = get_datasets(train_path_2, val_path_2, test_path_2)

# Calculate class weights for the new dataset to handle class imbalance
class_weights_dict = class_weights(train_path_2, train_dataset_2)

# Freeze all layers except the last one to fine-tune the model
for layer in model.layers[:-1]:  # Freeze all layers except the output layer
    layer.trainable = False

# Recompile the model with the new learning rate and the updated optimizer
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Continue training (fine-tuning) the model with the new dataset
trained_model = train(model, train_dataset_2, val_dataset_2, test_dataset_2, callbacks=call_backs, class_weights=class_weights_dict)
