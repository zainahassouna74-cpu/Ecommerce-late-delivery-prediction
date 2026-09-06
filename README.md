# E-Commerce Late Delivery Prediction

End-to-end machine learning project for predicting whether an e-commerce order will be delivered late using the Olist Brazilian e-commerce dataset.

## Project Overview

The goal of this project is to predict late deliveries before the final delivery outcome is known.

The workflow follows a leakage-aware machine learning pipeline:

- Load and join relational tables from PostgreSQL
- Create the late-delivery target
- Split the data chronologically into train, validation, and test sets
- Perform exploratory data analysis on the training set only
- Engineer features and preprocess the data
- Train, compare, tune, and evaluate classification models

## Target Definition

The binary target is:

- **is_late = 1** — actual customer delivery date is later than the estimated delivery date
- **is_late = 0** — order is delivered on or before the estimated delivery date

Orders with missing actual or estimated delivery dates are excluded because the target cannot be determined reliably.

## Repository Structure

```text
.
├── 1_read join tables.ipynb
├── 2_create labels.ipynb
├── 3_train val test split.ipynb
├── 4_EDA.ipynb
├── 5_preprocessing.ipynb
├── 6_train tune evaluate.ipynb
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Notebook Workflow
01 — Data Loading & Order-Level Table Construction

Loads Olist tables from PostgreSQL, inspects keys and duplicates, aggregates one-to-many tables before joining, and creates a final order-level dataset with one row per order.

Final dataset:

99,441 orders
23 columns
0 duplicate order_id values

Output: ml_table.csv

02 — Target Creation

Creates the is_late target using actual and estimated delivery dates.

Final labeled dataset:

96,476 orders
91.89% on-time orders
8.11% late orders

Output: labeled_table.csv

03 — Train / Validation / Test Split

Uses a chronological 70% / 15% / 15% split to better simulate a real-world production setting.

Train: 67,533 orders
Validation: 14,471 orders
Test: 14,472 orders

The late-delivery rate changes over time, which supports the use of a time-based split instead of a random split.

Outputs:

train.csv
validation.csv
test.csv
04 — Exploratory Data Analysis

EDA is performed only on the training set to avoid leakage.

Key findings:

The target is highly imbalanced.
Late orders tend to have higher price, freight, and payment values.
Late-delivery rates vary significantly across months.
March, November, and February show higher late-delivery rates.
Delivery performance varies by customer state.
Customer city has high cardinality.
Future information such as actual delivery dates and review scores should not be used as prediction features.
05 — Feature Engineering & Preprocessing

The preprocessing stage:

Removes identifiers and leakage-prone variables
Separates the target from model features
Creates time-based features:
Purchase year
Purchase month
Purchase day of week
Purchase hour
Estimated delivery days
Handles missing payment values
One-hot encodes customer_city and customer_state
Fits preprocessing on the training set only

Final feature space: 3,789 model-ready features

Saved artifacts include:

preprocessor.pkl
X_train_processed.npz
X_val_processed.npz
X_test_processed.npz
y_train.csv
y_val.csv
y_test.csv
feature_list.json
06 — Model Training, Tuning & Evaluation

Two models are compared on the validation set.

Logistic Regression
Metric	Validation
Accuracy	68.22%
Precision	10.18%
Recall	63.26%
F1 Score	17.54%
ROC-AUC	70.27%
Random Forest
Metric	Validation
Accuracy	62.59%
Precision	8.29%
Recall	59.64%
F1 Score	14.55%
ROC-AUC	66.03%

Logistic Regression performs better on Recall, F1 Score, and ROC-AUC, so it is selected for further tuning.

Threshold Tuning

The Logistic Regression classification threshold is tuned on the validation set.

Best threshold: 0.75

Validation performance at the selected threshold:

Precision: 18.09%
Recall: 23.29%
F1 Score: 20.36%
Final Test Results

The selected Logistic Regression model is evaluated once on the unseen test set using the tuned threshold.

Metric	Test
Accuracy	84.09%
Precision	10.04%
Recall	17.66%
F1 Score	12.80%
ROC-AUC	63.15%

The lower test performance compared with validation suggests limited generalization to the more recent test period and possible temporal distribution shift.

No additional model tuning is performed using the test set.

Tech Stack
Python
Pandas
NumPy
Matplotlib
Scikit-learn
SciPy
SQLAlchemy
PostgreSQL
psycopg2
python-dotenv
Joblib
Jupyter Notebook
Database Configuration

Database credentials are not stored directly in the notebooks.

Create a local .env file using .env.example as a template:

DB_USER=postgres
DB_PASSWORD=your_password_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=olist

The .env file is excluded from Git through .gitignore.

How to Run
Set up PostgreSQL and load the Olist dataset tables.
Create a .env file with your database credentials.
Install the project dependencies:
pip install -r requirements.txt
Run the notebooks in order from 01 to 06.

Each notebook saves the artifacts required by the next stage.
