# Cancer_Classifier_MLP

## Overview
This repository contains a simple program for classifying breast cancer cells as malignant (1) or benign (0) using a neural network implemented with TensorFlow and Keras.

## Requirements
Make sure you have the following dependencies installed:

+ TensorFlow
+ scikit-learn
+ pandas

You can install these packages using the following:
 ```bash
     pip install tensorflow scikit-learn pandas
 ```

## About dataset 
The program uses the Breast Cancer Wisconsin (Diagnostic) dataset, which is loaded from the 'Cancer_Data.csv' file. The dataset contains various features related to cell nuclei and a target variable indicating the diagnosis as malignant (M) or benign (B).
The dataset was extracted from the following publication on Kaggle: https://www.kaggle.com/datasets/erdemtaha/cancer-data/data

### License of dataset
+ Released under MIT License
+ Copyright (c) 2013 Mark Otto.
+ Copyright (c) 2017 Andrew Fong.


## Data Preprocessing
The 'diagnosis' column in the dataset is converted to numeric values, where 'M' is replaced with 1 and 'B' is replaced with 0. The data is then split into input features (X) and target labels (y). Further, the dataset is divided into training and testing sets using the **train_test_split** function.

## Model Architecture
The neural network model consists of two dense layers. The first layer has 12 units with a ReLU activation function, and the input dimension is set to the number of features (30). The second layer has 1 unit with a sigmoid activation function, suitable for binary classification.

Model Training
The model is compiled with the Adam optimizer and binary crossentropy loss function. The training is carried out for 50 epochs with a batch size of 32. The training progress is validated on the test set. After training, the model's accuracy on the test set is displayed.

**If the test accuracy is equal to or greater than 95%, the model is saved as 'model_pretrained_bcc.h5'.**

## How to Run
**1. Clone this repository:**
 ```bash
     git clone https://github.com/mat490/Cancer_Classifier_MLP.git
 ```
**2. Navigate to the repository:**
 ```bash
     cd your-repo
 ```
**3. Run the program:**
 ```bash
     python MLP_Cells.py
 ```

> [!WARNING]
> Ensure you have the necessary dataset file ('Cancer_Data.csv') in the correct location before running the program.
