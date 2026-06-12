# Bank Fraud Detection — End-to-End Machine Learning Pipeline

## Overview

This project builds an end-to-end machine learning pipeline for detecting fraudulent bank transactions using a large-scale financial dataset containing over **1,000,000 transactions and 26 features**.

The objective is to accurately identify fraudulent transactions in a highly imbalanced dataset while minimizing false negatives, which carry significant financial and business risk.

The project covers the complete machine learning workflow, including exploratory data analysis (EDA), preprocessing, feature engineering, class imbalance handling, model training, evaluation, and feature importance analysis.

---

## Problem Statement

Fraudulent financial transactions typically represent only a small fraction of total transactions, making fraud detection a challenging classification problem.

This project aims to:

* Detect fraudulent transactions using supervised machine learning.
* Handle severe class imbalance effectively.
* Compare multiple classification algorithms.
* Evaluate models using business-relevant metrics rather than accuracy alone.

---

## Dataset

**Dataset Size:** 1,000,000+ transactions

**Target Variable:** `is_fraud`

| Class | Description            |
| ----- | ---------------------- |
| 0     | Legitimate Transaction |
| 1     | Fraudulent Transaction |

### Features

The dataset contains transaction-level, customer-level, and behavioral attributes, including:

* Transaction amount
* Account balance
* Customer age
* Credit score
* Account age
* Merchant category
* Payment method
* Device type
* Geographic information
* Transaction frequency
* Distance from home
* Time since last transaction
* Failed login attempts
* PIN change history
* Temporal transaction features

---

## Exploratory Data Analysis (EDA)

Performed extensive exploratory analysis across all 26 features.

### Key Analyses

* Class imbalance analysis
* Fraud-rate breakdown by:

  * Merchant category
  * Payment method
  * Device type
  * Geography
* Transaction amount distribution
* Credit score distribution
* Distance from home analysis
* Time-based fraud patterns
* Correlation matrix analysis
* Outlier detection using boxplots

### Key Findings

* Fraud transactions exhibit distinct behavioral patterns compared to legitimate transactions.
* Certain merchant categories and payment methods demonstrate elevated fraud rates.
* Transaction timing and customer behavior features show predictive value.
* Significant class imbalance required specialized handling.

---

## Data Preprocessing

### Data Leakage Prevention

The following columns were removed before training:

* `fraud_type`
* `customer_id`
* Other non-predictive identifiers

All preprocessing transformers were fitted exclusively on training data and subsequently applied to the test set.

### Feature Engineering

Created and utilized features including:

* Hour of day
* Weekend indicator
* Night transaction indicator
* Transaction frequency metrics
* Behavioral transaction features

### Encoding & Scaling

Implemented using `ColumnTransformer`:

* One-Hot Encoding for categorical features
* StandardScaler for numerical features

---

## Handling Class Imbalance

The dataset contained approximately a **17:1 class imbalance**.

To address this issue:

* Applied SMOTE (Synthetic Minority Oversampling Technique) on the training data only.
* Generated synthetic fraud samples to improve minority-class representation.
* Reduced model bias toward the majority class.

---

## Models Trained

### Logistic Regression

Used as a baseline linear classifier.

### Random Forest Classifier

Tree-based ensemble model capable of capturing nonlinear relationships.

### XGBoost Classifier

Gradient boosting framework optimized for high predictive performance on tabular datasets.

---

## Model Evaluation

Accuracy alone is insufficient for fraud detection due to class imbalance.

Models were evaluated using:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC (Precision-Recall AUC)

### Why These Metrics?

* **Precision:** Measures false positive control.
* **Recall:** Measures ability to capture fraudulent transactions.
* **F1 Score:** Balances precision and recall.
* **ROC-AUC:** Evaluates ranking capability.
* **PR-AUC:** More informative for imbalanced classification problems.

---



### Best Performing Model

XGBoost achieved the highest overall performance on the held-out test set, outperforming both Logistic Regression and Random Forest across key evaluation metrics.

---

## Feature Importance

Feature importance analysis was performed to identify the strongest fraud predictors.

Examples of influential features include:

* Transaction amount
* Transaction frequency
* Time since last transaction
* Distance from home
* Failed attempts
* Device type
* Merchant category

---

## Tech Stack

### Languages

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* XGBoost

---

## How to Run

### Clone Repository

```bash
git clone https://github.com/your-username/bank-fraud-detection.git

cd bank-fraud-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Notebook

```bash
jupyter notebook
```

---

## Future Improvements

* Hyperparameter tuning using Optuna or GridSearchCV
* Cost-sensitive learning
* Real-time fraud detection API
* Model deployment using FastAPI
* Drift monitoring and retraining pipeline
* Explainability using SHAP

---

## Author

**Areesh Arafat**

LinkedIn: https://www.linkedin.com/in/areesh-arafat/

X: https://x.com/areesharafat
