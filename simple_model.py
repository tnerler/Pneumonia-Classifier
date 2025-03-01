from keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten
from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os

def simple_model(input_shape=(224, 224, 3)): 
    model = Sequential()

    # Block 1: First Convolutional Layer followed by MaxPooling
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Block 2: Second Convolutional Layer with L2 regularization and MaxPooling
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Block 3: Third Convolutional Layer with L2 regularization and MaxPooling
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Fully Connected Layers
    model.add(Flatten())  # Flatten the 3D output to 1D
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))  # Fully connected layer
    model.add(Dropout(0.6))  # Dropout for regularization
    
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Callbacks to manage training
    call_backs = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min'),  # Stop early if no improvement
        ModelCheckpoint('updated_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),  # Save best model
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)  # Reduce learning rate if no improvement
    ]

    # Compile the model with Adam optimizer and binary crossentropy loss
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=["accuracy"])

    return model, call_backs


def class_weights(train_path, train_dataset):  
    normal_images = os.path.join(train_path, 'NORMAL')  # Path to normal images
    pne_images = os.path.join(train_path, 'PNEUMONIA')  # Path to pneumonia images

    # Count the number of images in each class
    class_counts = {
        "NORMAL": len(os.listdir(normal_images)), 
        "PNEUMONIA": len(os.listdir(pne_images))
    }

    total_samples = sum(class_counts.values())  # Total number of samples
    num_classes = len(class_counts)  # Number of classes

    # Calculate class weights to balance the dataset
    class_weights = {train_dataset.class_indices[class_name]: total_samples / (num_classes * count)
                     for class_name, count in class_counts.items()}
    
    return class_weights


def train(model, train_dataset, val_dataset, test_dataset, callbacks, class_weights, epochs=20):
    # Train the model with the provided datasets and callbacks
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks, class_weight=class_weights)

    # Evaluate the model on the test dataset and print accuracy
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    return model
