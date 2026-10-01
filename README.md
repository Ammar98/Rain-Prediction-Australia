# Australian Rain Prediction

A machine learning classification project for predicting whether it will rain tomorrow in Australia based on historical weather observations.

The project covers exploratory data analysis, data cleaning, feature engineering, class balancing, categorical encoding, feature scaling, model comparison, evaluation with multiple classification metrics, and hyperparameter optimization using GridSearchCV.

## Project Overview

Weather prediction is a binary classification problem in which historical atmospheric measurements can be used to estimate whether rain will occur on the following day.

This project uses Australian weather observations to predict the target variable:

`RainTomorrow`

Several machine learning algorithms are trained and compared:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest

The models are evaluated using Accuracy, Precision, Recall, F1 Score, ROC curves, and ROC-AUC.

Hyperparameter optimization is additionally performed for the SVM model using `GridSearchCV`.

## Dataset

The original Australian weather dataset contains:

- **145,460 daily weather observations**
- **23 original features**

The dataset includes weather information such as:

- Minimum and maximum temperature
- Rainfall
- Wind direction and speed
- Humidity
- Atmospheric pressure
- Cloud coverage
- Temperature measurements
- Rain occurrence on the current day
- Rain occurrence on the following day

The prediction target is:

`RainTomorrow`

with two classes:

- `No` — No rain tomorrow
- `Yes` — Rain tomorrow

## Data Preprocessing

The dataset contains missing values across several weather variables.

The preprocessing workflow includes:

- Analysis of missing values
- Removal of `Evaporation` and `Sunshine` because of their high number of missing observations
- Removal of observations containing remaining missing values
- Exploratory analysis of weather variables
- Removal of selected features
- Transformation of the date into a seasonal representation
- Class balancing
- One-hot encoding of categorical variables
- Conversion of the target variable into binary labels
- Feature scaling using `MinMaxScaler`

After preprocessing and class balancing, the dataset used for model training contains:

**33,582 observations**

Categorical variables including:

- `WindGustDir`
- `WindDir3pm`
- `RainToday`

are transformed using One-Hot Encoding.

After encoding, the final feature matrix contains:

**49 input features**

## Class Balancing

The original target distribution is imbalanced, with substantially more `No` observations than `Yes` observations.

To reduce this imbalance, a random subset of the majority class is removed before training.

This results in a more balanced dataset for comparing classification models.

## Train-Test Split

The processed dataset is divided into training and test sets using `train_test_split`.

The features are then scaled with `MinMaxScaler`.

The scaler is fitted only on the training data and subsequently applied to the test data.

## Models and Results

### Logistic Regression

Logistic Regression is used as a baseline classification model.

**Results:**

- Training Accuracy: `79.5%`
- Test Accuracy: `79.5%`
- Precision: `0.81`
- Recall: `0.78`
- F1 Score: `0.79`

A confusion matrix and ROC curve are also generated to analyze classification performance.

### Support Vector Machine (SVM)

An SVM with an RBF kernel is trained using optimized hyperparameters.

Parameters:

```text
kernel = rbf
C = 10
gamma = 0.1
```

**Results:**

- Training Accuracy: `82.4%`
- Test Accuracy: `80.5%`
- Precision: `0.82`
- Recall: `0.80`
- F1 Score: `0.81`

Among the evaluated models, the SVM achieved the strongest overall test performance in the recorded experiments.

### Decision Tree

A Decision Tree classifier is trained as another non-linear classification approach.

**Results:**

- Training Accuracy: `100%`
- Test Accuracy: `72.6%`
- Precision: `0.72`
- Recall: `0.73`
- F1 Score: `0.73`
- ROC-AUC: `0.726`

The difference between training and test accuracy indicates substantial overfitting in the unrestricted Decision Tree.

### K-Nearest Neighbors (KNN)

A KNN classifier is evaluated on the scaled weather features.

**Results:**

- Training Accuracy: `82.4%`
- Test Accuracy: `74.4%`
- Precision: `0.73`
- Recall: `0.76`
- F1 Score: `0.74`
- ROC-AUC: `0.744`

### Random Forest

A Random Forest classifier is also evaluated.

**Results:**

- Training Accuracy: `98.8%`
- Test Accuracy: `78.4%`
- Precision: `0.82`
- Recall: `0.74`
- F1 Score: `0.78`

The model achieves relatively high precision, although the difference between training and test accuracy suggests some degree of overfitting.

## Model Comparison

| Model | Test Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 79.5% | 0.81 | 0.78 | 0.79 |
| SVM | **80.5%** | **0.82** | **0.80** | **0.81** |
| Decision Tree | 72.6% | 0.72 | 0.73 | 0.73 |
| KNN | 74.4% | 0.73 | 0.76 | 0.74 |
| Random Forest | 78.4% | 0.82 | 0.74 | 0.78 |

The SVM achieved the highest test accuracy and F1 Score among the evaluated models.

The comparison also demonstrates why relying only on accuracy is insufficient. Precision, Recall, and F1 Score provide additional insight into the behavior of each classifier.

## Hyperparameter Optimization

`GridSearchCV` is used to optimize the Support Vector Machine.

The parameter grid explores:

```python
{
    "kernel": ["linear", "rbf"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.05, 0.1, 1, 10, 100]
}
```

The best parameters found are:

```text
C = 10
gamma = 0.1
kernel = rbf
```

The best cross-validation accuracy reported by GridSearchCV is approximately:

**80.09%**

These optimized parameters are then used for the final SVM experiment.

## Model Evaluation

The project uses several classification metrics rather than relying on accuracy alone:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- ROC-AUC

This makes it possible to compare the models from different perspectives and better understand their classification behavior.

## Machine Learning Pipeline

The overall workflow can be summarized as:

```text
Australian Weather Dataset
        ↓
Exploratory Data Analysis
        ↓
Missing Value Analysis
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Class Balancing
        ↓
Categorical Encoding
        ↓
Min-Max Scaling
        ↓
Train / Test Split
        ↓
Model Training
        ├── Logistic Regression
        ├── SVM
        ├── Decision Tree
        ├── KNN
        └── Random Forest
        ↓
Model Evaluation
        ├── Accuracy
        ├── Precision
        ├── Recall
        ├── F1 Score
        ├── Confusion Matrix
        └── ROC / AUC
        ↓
SVM Hyperparameter Optimization
        ↓
GridSearchCV
```

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Key Learnings

This project demonstrates a complete classical machine learning classification workflow, including:

- Exploring a large real-world weather dataset
- Handling missing data
- Analyzing class imbalance
- Encoding categorical variables
- Scaling numerical features
- Training multiple classification algorithms
- Comparing models using multiple evaluation metrics
- Detecting overfitting through train-test performance differences
- Evaluating classifiers with confusion matrices and ROC curves
- Optimizing model hyperparameters using GridSearchCV

## Possible Improvements

The project could be extended by:

- Replacing random undersampling with more systematic class-imbalance techniques
- Using stratified train-test splitting
- Implementing preprocessing with `Pipeline` and `ColumnTransformer`
- Applying cross-validation consistently across all models
- Performing hyperparameter optimization for Random Forest, KNN, and Decision Tree
- Using additional ensemble methods such as Gradient Boosting or XGBoost
- Improving feature engineering for date and seasonal information
- Comparing performance before and after class balancing
- Adding model explainability using permutation importance or SHAP
- Packaging the best model for inference through a REST API

## Author

**Ammar Abouazan**

M.Sc. Applied Artificial Intelligence  
B.Sc. Computer Science
