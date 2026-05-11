# 💳 Credit Card Fraud Detection System

> An intelligent fraud detection system built with Python, AI & Machine Learning to identify fraudulent credit card transactions accurately and efficiently.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Methodology](#-methodology)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Interactive Predictor](#-interactive-predictor)
- [Technologies Used](#-technologies-used)
- [Key Findings](#-key-findings)

---

## 📖 Overview

With the rapid growth of online transactions and digital payment systems, credit card fraud has become a major concern for financial institutions and customers. This project develops an **intelligent fraud detection system** using AI and ML techniques to identify fraudulent transactions accurately and efficiently.

Key challenges addressed:
- **Highly imbalanced dataset** — fraud represents only ~0.17% of all transactions
- **High false positive cost** — legitimate transactions being flagged incorrectly
- **High false negative cost** — fraudulent transactions going undetected

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Total transactions** | 284,807 |
| **Fraudulent transactions** | 492 (0.173%) |
| **Legitimate transactions** | 284,315 (99.827%) |
| **Features** | 31 (Time, V1–V28, Amount, Class) |
| **File** | `creditcard.csv` (~143 MB) |

> **Note:** Features V1–V28 are the result of PCA (Principal Component Analysis) transformation applied to protect customer confidentiality. `Time` and `Amount` are the only original features.

---

## 📁 Project Structure

```
credit_card_fraud_detection/          ← root folder (upload everything here)
│
├── 📓 Credit_Card_Fraud_Detection.ipynb   # Main notebook
├── 📄 README.md                           # This file
├── 📦 creditcard.csv                      # Dataset (download from Kaggle)
│
├── 🤖 models/
│   ├── xgboost_fraud_model.pkl            # Trained XGBoost model
│   ├── random_forest_fraud_model.pkl      # Trained Random Forest model
│   ├── scaler_amount.pkl                  # StandardScaler for Amount
│   └── scaler_time.pkl                    # StandardScaler for Time
│
└── 📊 images/                             # ← ALL plot images go here
    ├── eda_overview.png
    ├── correlation_heatmap.png
    ├── class_balancing.png
    ├── confusion_matrices.png
    ├── model_comparison.png
    ├── roc_pr_curves.png
    ├── feature_importance.png
    ├── cross_validation.png
    └── threshold_tuning.png
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### Step 2 — Create a virtual environment
```bash
# Create
python -m venv fraud_env

# Activate (Windows)
fraud_env\Scripts\activate

# Activate (Mac/Linux)
source fraud_env/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib ipywidgets ipykernel notebook
```

### Step 4 — Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root folder.

---

## ▶️ How to Run

### In VS Code
```bash
# Register your virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=fraud_env --display-name "Python (fraud_env)"

# Open the notebook
code Credit_Card_Fraud_Detection.ipynb
```
Then select `Python (fraud_env)` as the kernel and click **Run All**.

### In Google Colab
1. Upload `Credit_Card_Fraud_Detection.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload `creditcard.csv` when prompted
3. Click `Runtime → Run all`

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Class distribution analysis (99.827% legitimate vs 0.173% fraud)
- Transaction amount and time distribution
- Feature correlation heatmap
- PCA feature distributions (V1, V4, V14)

### 2. Data Preprocessing
- Removed duplicate rows
- Applied `StandardScaler` to `Amount` and `Time` features
- V1–V28 features are already PCA-scaled

### 3. Class Imbalance Handling

| Technique | Before | After (Fraud) |
|---|---|---|
| **Original** | 226,602 vs 378 | — |
| **SMOTE** | — | 226,602 (balanced) |
| **Random Undersampling** | — | 378 (balanced) |
| **SMOTETomek** | — | 226,602 (balanced) |

### 4. Models Trained

| Model | Type | Imbalance Strategy |
|---|---|---|
| **Isolation Forest** | Anomaly Detection | Contamination rate |
| **Logistic Regression** | Supervised | SMOTE + class_weight |
| **Random Forest** | Supervised | SMOTE + class_weight |
| **XGBoost** | Supervised | SMOTE + scale_pos_weight |

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Average Precision Score
- Confusion Matrix
- 5-Fold Stratified Cross-Validation
- Decision Threshold Tuning

---

## 📈 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Isolation Forest | 0.938 | 0.237 | 0.242 | 0.240 | 0.938 |
| Logistic Regression | 0.969 | 0.053 | 0.874 | 0.101 | 0.962 |
| **Random Forest** | **0.930** | **0.830** | **0.768** | **0.798** | **0.970** |
| **XGBoost** | **1.000** | **0.576** | **0.800** | **0.670** | **0.973** |

### 🏆 Best Model — XGBoost (ROC-AUC: 0.9747)

| Metric | Value |
|---|---|
| ROC-AUC | **0.9747** |
| Average Precision | **0.7981** |
| Optimal Threshold | **0.94** |
| Fraud Detection Rate | **80%** |
| False Alarm Rate | **0.10%** |

### Confusion Matrix — XGBoost

|  | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 56,595 (TN) | 56 (FP) |
| **Actual Fraud** | 19 (FN) | 76 (TP) |

### 5-Fold Cross-Validation ROC-AUC

| Model | Mean | Std |
|---|---|---|
| Logistic Regression | ~0.980 | ±0.004 |
| Random Forest | ~0.965 | ±0.018 |
| XGBoost | ~0.978 | ±0.010 |

---

## 📊 Visualizations

### Exploratory Data Analysis
![EDA Overview](eda_overview.png)

### Feature Correlation Heatmap
![Correlation Heatmap](correlation_heatmap.png)

### Class Balancing Techniques
![Class Balancing](class_balancing.png)

### Model Performance Comparison
![Model Comparison](model_comparison.png)

### ROC & Precision-Recall Curves
![ROC PR Curves](roc_pr_curves.png)

### Confusion Matrices — All Models
![Confusion Matrices](confusion_matrices.png)

### Feature Importance (Random Forest & XGBoost)
![Feature Importance](feature_importance.png)

> Both models agree: **V14** and **V10** are the strongest fraud indicators.

### 5-Fold Cross-Validation
![Cross Validation](cross_validation.png)

### XGBoost Threshold Tuning
![Threshold Tuning](threshold_tuning.png)

> Optimal decision threshold found at **0.94**, maximising F1-Score.

---

## 🔮 Interactive Predictor

The notebook includes a fully interactive fraud prediction widget at the end:

```
💳 Credit Card Fraud Detection — Interactive Predictor
┌────────────────────────────────────────────────────┐
│  Mode: Manual input | Pick from dataset | Batch    │
├────────────────────────────────────────────────────┤
│  Amount (€): [_______]   Time (s): [_______]       │
│  V1–V28: [input fields for all PCA features]       │
├────────────────────────────────────────────────────┤
│  [🔍 Predict] [⚠ Fraud sample] [✅ Legit sample]  │
│  [🎲 Random]  [🗑 Clear]                           │
├────────────────────────────────────────────────────┤
│  Result:                                           │
│  🔴 FRAUDULENT TRANSACTION                        │
│  Fraud Probability : 98.74%   [HIGH]              │
│  ████████████████████░░░░░░░░ 98.7%              │
└────────────────────────────────────────────────────┘
```

**Features:**
- Manual input of all 31 transaction features
- Load real fraud / legitimate samples from the dataset
- Pick any row by index using a slider
- Batch test N random transactions with a styled results table
- Live confusion matrix for batch results

---

## 🛠️ Technologies Used

| Category | Libraries |
|---|---|
| **Data manipulation** | `numpy`, `pandas` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` |
| **Imbalance handling** | `imbalanced-learn` (SMOTE, SMOTETomek) |
| **Gradient Boosting** | `xgboost` |
| **Model persistence** | `joblib` |
| **Interactive UI** | `ipywidgets` |
| **Notebook** | `jupyter`, `ipykernel` |

---

## 🔍 Key Findings

1. **V14 is the strongest fraud indicator** — ranked #1 in both Random Forest (0.1908) and XGBoost (0.4057) feature importance
2. **SMOTE significantly improved recall** — from near-zero to 80%+ for fraud class
3. **XGBoost achieved the best ROC-AUC** of 0.9747, correctly identifying 76 out of 95 fraudulent transactions in the test set
4. **Optimal threshold of 0.94** (not the default 0.5) maximises F1-Score by balancing precision and recall
5. **Isolation Forest** works well as an unsupervised baseline (ROC-AUC: 0.9377) without needing labels
6. **Fraudulent transactions** have a higher mean amount (€122.2) compared to legitimate ones (€88.3)
7. **False alarm rate of only 0.10%** — only 56 legitimate transactions out of 56,651 were incorrectly flagged

---


## 🙏 Acknowledgements

- Dataset provided by [Machine Learning Group — ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

<p align="center">Made with ❤️ using Python & Machine Learning</p>
