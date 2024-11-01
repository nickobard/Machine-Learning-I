**[Data Preprocessing and Binary Classification](/01 - Data Preprocessing and Binary Classification)**


[file](01 - Data Preprocessing and Binary Classification) 

The notebook preprocesses Titanic passenger data to predict survival using Decision Tree and k-Nearest Neighbors (k-NN) classifiers. Irrelevant features are dropped, categorical variables are encoded, and missing values are handled. After hyperparameter tuning and evaluation, the k-NN classifier with standardization and cross-validation is selected as the final model, achieving approximately 78.7% accuracy. Predictions on new data are made and saved to `results.csv`.

---

**Second Notebook: Regression**

The notebook predicts life expectancy across countries and years using regression models. Data preprocessing includes handling missing values, encoding categorical variables, and addressing outliers. A custom Random Forest regressor is implemented alongside Ridge and Lasso regression models. After hyperparameter tuning and evaluation, the Random Forest model is chosen for its superior performance and robustness. It is used to make predictions on new data, which are saved to `result.csv`.