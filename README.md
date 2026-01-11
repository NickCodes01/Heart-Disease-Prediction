Heart Disease Prediction – Machine Learning Pipeline
Author: Nicholas Chludzinski


This project implements a complete supervised machine learning workflow for predicting heart disease using a structured clinical dataset. It demonstrates data preprocessing, model training, hyperparameter tuning, and evaluation using scikit‑learn tools. The dataset used in this project is licensed under CC0 Public Domain and may be freely redistributed.


Project Overview
This repository contains a full machine learning pipeline built around a Gradient Boosting Classifier with GridSearchCV hyperparameter tuning. It is designed as a reusable template for binary classification tasks involving tabular data.


The workflow includes:
 Data cleaning
 One‑hot encoding of categorical variables
 Mapping Yes and No fields to binary
 Train and test split with stratification
 Feature scaling
 Hyperparameter tuning
 Model evaluation
 ROC and AUC visualization


Files
 Heart Disease.ipynb - main script containing the full machine learning pipeline
 heart_disease_dataset.csv - dataset used for training (CC0 Public Domain)
 README.md - project documentation


Methods Used
 GradientBoostingClassifier
 GridSearchCV with five‑fold cross‑validation
 StandardScaler
 One‑hot encoding
 Binary mapping
 ROC and AUC evaluation


Model Evaluation
The script outputs accuracy, a classification report, a confusion matrix, an ROC curve, and an AUC score. These metrics provide a complete view of model performance.


How to Run
Install dependencies:
pip install numpy pandas scikit-learn matplotlib
Ensure the dataset file heart_disease_dataset.csv is in the project directory.
Update the CSV path in the script if needed:
df = pd.read_csv('heart_disease_dataset.csv')
Run the script


Purpose
This project demonstrates a clean, end‑to‑end machine learning workflow for predicting whether a patient has heart disease based on clinical features. It serves as a reusable template for binary classification problems and a practical example of applied machine learning in healthcare analytics.

