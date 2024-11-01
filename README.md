**[Data Preprocessing and Binary Classification](https://github.com/nickobard/Machine-Learning-I/tree/master/01%20-%20Data%20Preprocessing%20and%20Binary%20Classification)**


The notebook preprocesses Titanic passenger data to predict survival using Decision Tree and k-Nearest Neighbors (k-NN) classifiers. Irrelevant features are dropped, categorical variables are encoded, and missing values are handled. After hyperparameter tuning and evaluation, the k-NN classifier with standardization and cross-validation is selected as the final model, achieving approximately 78.7% accuracy. Predictions on new data are made and saved to `results.csv`.

---

**[Regression](https://github.com/nickobard/Machine-Learning-I/tree/master/02%20-%20Regression)**

The notebook predicts life expectancy across countries and years using regression models. Data preprocessing includes handling missing values, encoding categorical variables, and addressing outliers. A custom Random Forest regressor is implemented alongside Ridge and Lasso regression models. After hyperparameter tuning and evaluation, the Random Forest model is chosen for its superior performance and robustness. It is used to make predictions on new data, which are saved to `result.csv`.


---

Repository structure:

```
.
├── 01 - Data Preprocessing and Binary Classification
│   ├── data.csv
│   ├── evaluation.csv
│   ├── homework_01.ipynb
│   ├── README.md
│   └── results.csv
├── 02 - Regression
│   ├── data.csv
│   ├── evaluation.csv
│   ├── homework_02.ipynb
│   ├── README.md
│   ├── result.csv
│   └── standardization.ipynb
└── README.md

3 directories, 12 files
```
