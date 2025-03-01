# Pneumonia Classifier %85 Test Accuracy

This repository contains a deep learning model for classifying pneumonia in chest X-ray images with an accuracy of around **85%**. The model is trained using two datasets for pneumonia detection. The classifier uses convolutional neural networks (CNN) to achieve accurate classification.

## Datasets

1. **Chest X-ray Pneumonia Dataset**:  
   - This dataset contains chest X-ray images labeled for pneumonia detection. You can find it on Kaggle at the following link:  
   [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

2. **Chest X-ray Dataset (COVID-19 and Pneumonia)**:  
   - Although this dataset includes images labeled for both COVID-19 and pneumonia, **only the pneumonia and normal labels were used** during the training. The COVID-19 labels were not utilized in the current model.  
   [Chest X-ray COVID-19 Pneumonia Dataset](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using the Keras library. The architecture consists of the following layers:

- **Conv2D** layers for feature extraction
- **MaxPooling2D** layers for downsampling
- **Dense** layers for fully connected output
- **Dropout** for regularization
- **Sigmoid activation** for binary classification

## Requirements

To run this project, you'll need the following Python libraries:

- TensorFlow / Keras
- NumPy
- Matplotlib
- pandas
- os
- sklearn

You can install the required libraries using `pip`:

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn
