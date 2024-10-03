
---

# Earthquake Detection Model Code Explanation

## Table of Contents
1. [Overview](#overview)
2. [Code Structure](#code-structure)
3. [Code Explanation](#code-explanation)
   - [Importing Libraries](#importing-libraries)
   - [Loading the Dataset](#loading-the-dataset)
   - [Data Preprocessing](#data-preprocessing)
   - [Splitting the Data](#splitting-the-data)
   - [Feature Scaling](#feature-scaling)
   - [Handling Class Imbalance with SMOTE](#handling-class-imbalance-with-smote)
   - [Defining Hyperparameter Grid for Random Forest](#defining-hyperparameter-grid-for-random-forest)
   - [Grid Search for Hyperparameter Tuning](#grid-search-for-hyperparameter-tuning)
   - [Model Training](#model-training)
   - [Model Validation and Evaluation](#model-validation-and-evaluation)
   - [Generating Synthetic Data](#generating-synthetic-data)
   - [Testing the Model](#testing-the-model)
4. [Conclusion](#conclusion)

## Overview
This document explains the code used to build an earthquake detection model using machine learning. The model classifies seismic events based on features extracted from accelerometer and seismograph readings. It employs a Random Forest Classifier and utilizes techniques for data balancing and hyperparameter tuning to enhance performance.

## Code Structure
The code consists of several sections, each focusing on a specific aspect of model development:
1. **Importing Libraries**: Necessary Python libraries for data handling, visualization, machine learning, and model persistence.
2. **Loading the Dataset**: Reading the earthquake detection dataset from an Excel file.
3. **Data Preprocessing**: Preparing the data for analysis, including label encoding and feature selection.
4. **Splitting the Data**: Dividing the dataset into training, validation, and test sets.
5. **Feature Scaling**: Standardizing the feature values to improve model performance.
6. **Handling Class Imbalance with SMOTE**: Applying SMOTE to create a balanced training dataset.
7. **Defining Hyperparameter Grid for Random Forest**: Setting up the parameter grid for hyperparameter tuning.
8. **Grid Search for Hyperparameter Tuning**: Using GridSearchCV to find the best model parameters.
9. **Model Training**: Training the Random Forest model with the best parameters.
10. **Model Validation and Evaluation**: Evaluating the model's performance using validation and test datasets.
11. **Generating Synthetic Data**: Creating synthetic data for additional testing.
12. **Testing the Model**: Using synthetic data to test the trained model.

## Code Explanation

### Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from imblearn.over_sampling import SMOTE
```
This section imports the necessary libraries:
- **pandas**: For data manipulation.
- **numpy**: For numerical operations.
- **matplotlib & seaborn**: For data visualization.
- **scikit-learn**: For model building and evaluation.
- **imbalanced-learn**: For handling class imbalance.
- **joblib**: For saving and loading the model.

### Loading the Dataset
```python
data = pd.read_excel('balanced_earthquake_detection_dataset.xlsx', sheet_name=None)
train_data = data['Train']
test_data = data['Test']
```
This code reads an Excel file containing the earthquake detection dataset and separates it into training and testing datasets.

### Data Preprocessing
```python
combined_data = pd.concat([train_data, test_data], ignore_index=True)
combined_data['Label'] = combined_data['Label'].map({'normal': 0, 'earthquake': 1})
```
Here, we combine the training and testing datasets for preprocessing. The labels are encoded into binary values where `normal` is 0 and `earthquake` is 1.

### Splitting the Data
```python
X = combined_data[['Time', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Seismo_Reading']]
y = combined_data['Label']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```
The dataset is split into features (`X`) and target labels (`y`). Then, the data is further split into training (60%), validation (20%), and test sets (20%).

### Feature Scaling
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```
StandardScaler is used to standardize the feature values by removing the mean and scaling to unit variance, which helps in improving model performance.

### Handling Class Imbalance with SMOTE
```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
```
SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training dataset to address class imbalance by creating synthetic samples of the minority class.

### Defining Hyperparameter Grid for Random Forest
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]  # Adjust class weights
}
```
A grid of hyperparameters is defined for tuning the Random Forest model, allowing the model to find the best configuration during training.

### Grid Search for Hyperparameter Tuning
```python
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)
```
GridSearchCV is employed to perform an exhaustive search over the specified parameter grid. It uses cross-validation to evaluate the performance of different parameter combinations.

### Model Training
```python
best_model = grid_search.best_estimator_
best_model.fit(X_train_resampled, y_train_resampled)
```
The best model found during grid search is trained on the resampled training dataset.

### Model Validation and Evaluation
```python
y_val_pred = best_model.predict(X_val_scaled)
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))

confusion_mtx = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Earthquake'], 
            yticklabels=['Normal', 'Earthquake'])
plt.title('Confusion Matrix (Validation Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```
The model's performance is validated using the validation dataset. A classification report and confusion matrix are generated to evaluate precision, recall, F1-score, and overall accuracy.

### Generating Synthetic Data
```python
def generate_synthetic_data(num_samples=5):
    synthetic_time = np.random.uniform(0, 1000, num_samples)
    synthetic_accel_x = np.random.uniform(-2, 2, num_samples)
    synthetic_accel_y = np.random.uniform(-2, 2, num_samples)
    synthetic_accel_z = np.random.uniform(-2, 2, num_samples)
    synthetic_seismo_reading = np.random.uniform(0, 10, num_samples)

    synthetic_data = pd.DataFrame({
        'Time': synthetic_time,
        'Accel_X': synthetic_accel_x,
        'Accel_Y': synthetic_accel_y,
        'Accel_Z': synthetic_accel_z,
        'Seismo_Reading': synthetic_seismo_reading
    })

    synthetic_data_scaled = scaler.transform(synthetic_data)
    synthetic_predictions = best_model.predict(synthetic_data_scaled)
    
    return synthetic_data, synthetic_predictions
```
A function is defined to generate synthetic earthquake data for testing the model. The generated data is scaled and predictions are made using the trained model.

### Testing the Model
```python
synthetic_data, synthetic_predictions = generate_synthetic_data(num_samples=10)
synthetic_data['Predicted_Label'] = synthetic_predictions

print("Synthetic Data Predictions:")
print(synthetic_data)
```
Finally, the synthetic data is generated and predictions are made. The predicted labels are then appended to the synthetic data for review.

## Conclusion
This code provides a complete framework for developing an earthquake detection model using machine learning techniques. It covers all essential steps, from data loading and preprocessing to model training and evaluation. By understanding each section, you can modify and adapt the code for different datasets or modeling approaches.

---
