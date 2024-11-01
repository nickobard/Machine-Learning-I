# Regression

The notebook outlines the process of performing regression to predict life expectancy across different countries and years. It encompasses data preprocessing, implementing a custom Random Forest regressor, applying various regression models (including Ridge and Lasso regression), hyperparameter tuning, model evaluation, and making predictions on new data.

## 1. **Data Analysis and Preprocessing**

### **Data Loading and Initial Inspection**

- **Data Source**: The dataset is loaded from `data.csv`.
- **Features**: The dataset includes various features such as Year, Country, Status (Developed or Developing), Life expectancy (target variable), and several health-related indicators.
- **Initial Exploration**:
    - Checked for missing values using `df_data.isna().any()`.
    - Examined data types and basic statistics using `df_data.info()` and `df_data.describe()`.
    - Identified that several features have missing values and potential outliers.

### **Feature Analysis and Transformation**

#### **Country**

- **Challenge**: 'Country' is a categorical feature with 183 unique values.
- **Options Considered**:
    - **One-Hot Encoding**: Not preferred due to the significant increase in dimensionality and potential issues with new countries in evaluation data.
    - **Label Encoding**: Chosen for simplicity, using `LabelEncoder` from scikit-learn.
- **Implementation**: Applied `LabelEncoder` to transform 'Country' into numerical values.

#### **Year**

- Recognized as an ordinal feature and left as is since it is already numeric and represents chronological order.

#### **Status**

- Contains two categories: 'Developed' and 'Developing'.
- Transformed into numerical values using `LabelEncoder`.

#### **Life Expectancy**

- Target variable for regression.
- Checked for outliers using box plots.
- Found some outliers but decided not to remove them immediately.

#### **Other Features**

- **Missing Values**: Identified in several features.
- **Outliers**: Detected in many features using box plots.
- **Decision**: For the Random Forest model, missing values are filled with -1 since Random Forest can handle such placeholders. For regression models like Ridge and Lasso, rows with missing values are dropped due to the sensitivity of these models to missing data.

### **Final Data Preparation**

- After encoding categorical variables and handling missing values, the data is prepared for modeling.
- Created a preprocessed DataFrame `df_preprocessed` with all features in numerical form and no missing values for models that cannot handle them.

## 2. **Custom Random Forest Regression**

### **Implementation of `CustomRandomForest`**

- Implemented a custom Random Forest regressor using the provided skeleton.
- The regressor consists of multiple `DecisionTreeRegressor` models from scikit-learn.
- **Hyperparameters**:
    - `n_estimators`: Number of trees in the forest.
    - `max_samples`: Number of samples to draw for each tree (using bootstrap sampling).
    - `max_depth`: Maximum depth of each tree.

### **Model Suitability**

- Random Forest is suitable for this task because:
    - It can handle missing values (after placeholder substitution).
    - It's robust to outliers and can manage high-dimensional data.
    - It does not assume linear relationships between features and the target variable.

### **Data Splitting**

- Split the preprocessed data into training, validation, and test sets:
    - **Training Set**: 60% of the data.
    - **Validation Set**: 24% of the data (60% of the remaining 40%).
    - **Test Set**: 16% of the data (40% of the remaining 40%).

### **Model Training and Hyperparameter Tuning**

- Performed grid search over a range of hyperparameters:
    - `n_estimators`: 3 to 30 (step of 3).
    - `max_samples`: 200 to the length of the training set (step of 200).
    - `max_depth`: 5 to 30 (step of 2).
- Trained the model on the training set and evaluated on the validation set using RMSE (Root Mean Square Error) and MAE (Mean Absolute Error).
- Selected the best hyperparameters based on the lowest validation RMSE.

### **Model Evaluation**

- Evaluated the final model on both training and validation sets.
- **Results**:
    - **Training RMSE**: Lower due to the model fitting the training data.
    - **Validation RMSE**: Slightly higher, indicating the model's generalization ability.
- Visualized predictions versus actual values to assess model performance.

## 3. **Ridge Regression**

### **Model Suitability**

- Ridge Regression may not be ideal due to:
    - Sensitivity to missing values and outliers.
    - Potential issues with multicollinearity and linear dependencies among features.
- However, it's beneficial for handling multicollinearity by penalizing large coefficients.

### **Data Preparation**

- Dropped rows with missing values, resulting in a reduced dataset.
- Noted that the dataset still has enough samples for regression despite the reduction.

### **Data Splitting and Normalization**

- Split the cleaned data into training, validation, and test sets using the same proportions.
- Applied normalization techniques:
    - **Min-Max Scaling**: Scales features to a range between 0 and 1.
    - **Standardization**: Scales features to have zero mean and unit variance.
- Normalization is crucial for Ridge Regression to ensure fair penalization across all features.

### **Model Training and Hyperparameter Tuning**

- Explored a range of values for the regularization parameter `alpha` (λ).
- Noted that without normalization, the model encounters issues with ill-conditioned matrices.
- After normalization, performed grid search over `alpha` values.
- Used cross-validation to select the best `alpha`, improving the reliability of the model.

### **Model Evaluation**

- Evaluated the Ridge Regression model on training and validation sets.
- Observed that the model with the best `alpha` after normalization performed better.
- Cross-validation provided a more stable estimate of the model's performance.

## 4. **Lasso Regression and Linear Regression**

### **Model Suitability**

- **Lasso Regression** is useful for feature selection by shrinking some coefficients to zero.
- **Linear Regression** serves as a baseline model.

### **Data Preparation**

- Used the same cleaned and normalized data as for Ridge Regression.

### **Model Training**

- Trained Lasso Regression with a range of `alpha` values and `max_iter` set to 10,000 to ensure convergence.
- Identified features with non-zero coefficients.

### **Feature Selection and Linear Regression**

- Selected features with non-zero coefficients from the Lasso model.
- Retrained a Linear Regression model using only these selected features.
- Aimed to improve performance by reducing multicollinearity and overfitting.

### **Model Evaluation**

- Evaluated both Lasso and Linear Regression models.
- Found that the performance did not significantly improve compared to the Ridge Regression or Random Forest models.
- Concluded that while Lasso aids in feature selection, it did not enhance model accuracy in this case.

## 5. **Final Model Selection and Evaluation**

### **Model Selection**

- Chose the **Custom Random Forest** as the final model due to:
    - Better performance in terms of RMSE and MAE.
    - Robustness to missing values and outliers.
    - Stability across different data splits.

### **Final Evaluation on Test Data**

- Retrained the Random Forest model using the best hyperparameters on the combined training and validation sets.
- Evaluated the final model on the test set.
- **Expected Performance on New Data**:
    - **RMSE**: Approximately 3.61
    - **MAE**: Approximately 2.62

## 6. **Making Predictions on Evaluation Data**

### **Data Loading and Preprocessing**

- Loaded `evaluation.csv` containing new data for prediction.
- Applied the same preprocessing steps as for the training data:
    - Transformed 'Country' and 'Status' using the previously fitted `LabelEncoder`.
    - Filled missing values with -1.

### **Predictions and Saving Results**

- Used the final Random Forest model to predict life expectancy for the evaluation data.
- Created a DataFrame with columns 'Country', 'Year', and 'Life expectancy'.
- Ensured that 'Country' was converted back to its original string representation for readability.
- Saved the predictions to `result.csv`.

### **Verification of Countries in Evaluation Data**

- Checked whether the evaluation data contained any countries not present in the training data.
- Found that all countries in the evaluation set were present in the training set.
- This validation supports the reliability of the model's predictions on the evaluation data.

## 7. **Summary of Findings**

- **Random Forest Regressor**:
    - Demonstrated superior performance in handling missing values and outliers.
    - Provided more reliable predictions compared to Ridge and Lasso Regression models.
- **Ridge Regression**:
    - Required normalization to perform effectively.
    - Showed improvements when using cross-validation to select hyperparameters.
- **Lasso Regression**:
    - Helpful for feature selection but did not significantly improve model performance.
- **Linear Regression**:
    - Served as a baseline but was less effective due to the complexity of the data and the presence of multicollinearity.

## 8. **Methodological Considerations**

- **Handling Missing Values**:
    - For Random Forest, missing values were replaced with -1.
    - For regression models, rows with missing values were dropped to prevent biases.
- **Normalization**:
    - Essential for Ridge and Lasso Regression to ensure fair penalization across features.
- **Outliers**:
    - Acknowledged but not removed due to time constraints.
    - Recognized that outliers could impact regression models more than Random Forest.
- **Cross-Validation**:
    - Improved the reliability of hyperparameter selection and model evaluation.
- **Feature Encoding**:
    - Used `LabelEncoder` for simplicity, though aware of potential limitations.
    - Considered one-hot encoding but opted against it due to increased dimensionality and computational cost.

## 9. **Conclusion**

The notebook successfully demonstrates the process of:

- Preprocessing complex datasets with missing values and categorical variables.
- Implementing and applying a custom Random Forest regressor.
- Comparing different regression models and selecting the most appropriate one.
- Evaluating models using RMSE and MAE metrics.
- Making predictions on new data and preparing the results for submission.

**Final Deliverables**:

- A trained Random Forest regression model capable of predicting life expectancy.
- A `result.csv` file containing the predicted life expectancy for the evaluation dataset.

The choice of the Random Forest model was justified by its robust performance and ability to handle the dataset's challenges effectively. The systematic approach to model evaluation and selection ensures that the predictions made are as accurate and reliable as possible given the data and time constraints.