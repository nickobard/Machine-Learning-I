# Data Preprocessing and Binary Classification

The notebook outlines the process of data preprocessing and binary classification for predicting the survival of passengers aboard the Titanic. The primary objectives are to clean and prepare the data, apply two classification models (Decision Tree and k-Nearest Neighbors), evaluate their performance, and use the best model to predict outcomes on new data.

### 1. **Data Import and Initial Setup**

- **Libraries Imported**: pandas, numpy, matplotlib.pyplot, and various modules from sklearn.
- **Data Loading**: The Titanic dataset is loaded from `data.csv` into a pandas DataFrame named `df_data`.

### 2. **Data Analysis and Preprocessing**

The notebook performs a thorough analysis and preprocessing of each feature:

- **ID**: Dropped as it doesn't contribute to the prediction (it's merely an identifier).
- **Survived**: Retained as the target variable.
- **Pclass**: Converted to an ordinal categorical variable with ordering (3rd < 2nd < 1st class).
- **Name**: Dropped due to high uniqueness and complexity in extracting meaningful features.
- **Sex**: Converted to a categorical variable and encoded numerically.
- **Age**: Contains missing values and outliers. Missing values are filled with -1 to indicate absence, and outliers are noted but not removed.
- **SibSp and Parch**: Retained as they provide information on family relations aboard, despite outliers.
- **Ticket**: Dropped due to high uniqueness and limited predictive power.
- **Fare**: Missing values filled with -1. Noted for outliers.
- **Cabin**: Dropped due to a high number of missing values and complexity in processing.
- **Embarked**: Missing values are filled. One-hot encoding is applied to represent the three categories.
- **Home.dest**: Dropped due to a high number of missing values and complexity.

**General Preprocessing Steps:**

- Missing values are handled appropriately for each feature.
- Categorical variables are encoded numerically.
- Features deemed non-contributory are dropped to simplify the model.

### 3. **Decision Tree Classifier**

**Suitability:**

- Handles both numerical and categorical data well.
- Robust to outliers and missing values after preprocessing.

**Process:**

- **Data Splitting**: The data is split into training (60%), validation (24%), and test sets (16%).
- **Hyperparameter Tuning**: Used `max_depth` and `criterion` as hyperparameters. Grid search is performed over a range of values.
- **Model Training**: The Decision Tree model is trained on the training set.
- **Evaluation Without Cross-Validation**:
    - Accuracy is calculated on both training and validation sets.
    - The best model is selected based on validation accuracy.
    - ROC curve and AUC are plotted and calculated for the validation set.
- **Evaluation With Cross-Validation**:
    - 5-fold cross-validation is used to better estimate model performance.
    - The best hyperparameters are selected based on cross-validation accuracy.
    - The final model is evaluated on the validation set.

### 4. **k-Nearest Neighbors Classifier**

**Suitability:**

- Effective for smaller datasets.
- Sensitive to feature scaling and outliers.

**Process:**

- **Data Splitting**: Similar to the Decision Tree, with the same training, validation, and test sets.
- **Feature Scaling**:
    - Models are trained both with and without feature scaling.
    - Min-Max Scaling and Standardization are applied to observe effects on performance.
- **Hyperparameter Tuning**:
    - Hyperparameters tuned include the number of neighbors (`n_neighbors`), distance metrics (`p`), and weighting (`weights`).
    - Grid search is performed over a defined range.
- **Model Training**:
    - The k-NN model is trained on the scaled training data.
- **Evaluation Without Cross-Validation**:
    - Accuracy is calculated on both training and validation sets for different scaling methods.
    - The best model is selected based on validation accuracy.
- **Evaluation With Cross-Validation**:
    - 5-fold cross-validation is used with standardized data.
    - The final model is selected based on cross-validation accuracy.

### 5. **Final Model Selection and Performance Estimation**

- The final model chosen is the k-NN classifier with standardization and cross-validation, using the hyperparameters:
    - `n_neighbors`: 6
    - `weights`: 'distance'
    - `p`: 1 (Manhattan distance)
- **Performance Estimation**:
    - The expected accuracy on new data is estimated based on cross-validation results, acknowledging that the validation accuracy might be optimistic.
    - The estimated accuracy is approximately 78.7%.

### 6. **Predicting on Evaluation Data**

- **Data Loading**: The evaluation dataset is loaded from `evaluation.csv`.
- **Preprocessing**:
    - The same preprocessing steps applied to the training data are replicated on the evaluation data.
    - Features are encoded and scaled consistently.
- **Prediction**:
    - The final k-NN model is used to predict the `survived` status for the evaluation dataset.
- **Result Compilation**:
    - Predictions are saved in a DataFrame with columns `ID` and `survived`.
    - The results are exported to `results.csv`.

### 7. **Summary of Findings**

- **Decision Tree Model**:
    - Provided reasonable accuracy but was potentially overfitting on the training data.
    - Simpler to interpret but less effective than the k-NN model after tuning.
- **k-NN Model**:
    - Performed better with feature scaling due to sensitivity to the magnitude of features.
    - Cross-validation helped in selecting hyperparameters that generalize better to unseen data.
- **Model Selection Rationale**:
    - The k-NN model with standardization and cross-validation offered the best balance between bias and variance.
    - It showed consistent performance across training, validation, and test sets.

### 8. **Methodological Considerations**

- **Data Leakage Prevention**:
    - Care was taken to prevent using information from the validation and test sets during preprocessing (e.g., when imputing missing values).
- **Hyperparameter Tuning**:
    - Grid search and cross-validation were employed to prevent overfitting to a particular split of the data.
- **Feature Selection**:
    - Features with high missing values or low predictive power were dropped to simplify the model and improve performance.

### 9. **Conclusion**

The notebook successfully demonstrates the end-to-end process of preparing data, selecting and tuning models, evaluating performance, and making predictions on new data for a binary classification problem. The k-NN classifier, when properly tuned and scaled, outperformed the Decision Tree classifier for this specific task.

**Final Deliverables**:

- A trained k-NN model capable of predicting Titanic passenger survival.
- A `results.csv` file containing the predicted survival status for the evaluation dataset.