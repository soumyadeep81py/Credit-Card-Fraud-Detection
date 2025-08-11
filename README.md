# 🎉 Credit Card Fraud Detection Model

Welcome to the **Credit Card Fraud Detection Model**, a comprehensive machine learning project built for Google Colab! This repository contains a Python script to detect fraudulent credit card transactions using data preprocessing, feature engineering, model training, and an interactive prediction tool. 🚀

---

## 📋 Project Overview

This project leverages machine learning to identify fraudulent transactions in credit card data. It processes datasets (`fraudTrain.csv` and `fraudTest.csv`), engineers meaningful features, trains models like Random Forest and Logistic Regression, and allows users to input transaction details for real-time fraud predictions.

- **Goal**: Build a robust fraud detection system with high accuracy and interpretability.
- **Environment**: Designed for Google Colab with Google Drive integration.
- **Current Date**: *, Monday, August 11, 2025*

---

## 🚀 Getting Started

### Prerequisites
- A Google account with Google Drive.
- Google Colab (access at [colab.research.google.com](https://colab.research.google.com)).
- Python libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
- Dataset files: `fraudTrain.csv` (335 MB) and `fraudTest.csv` (143.4 MB).

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection

📂 Project Structure

main.py or .ipynb: Main script/notebook with the fraud detection pipeline.
datasets/: Folder for fraudTrain.csv and fraudTest.csv (upload to Google Drive).
README.md: This file!


⚙️ How It Works
The script is divided into modular functions, executed as Colab cells:
1. Data Loading and Validation (load_and_validate_data)

Purpose: Loads and checks fraudTrain.csv and fraudTest.csv for consistency.
Arguments: None (uses hardcoded file names; modify to use paths).
Output: DataFrames with summaries (shape, columns).

2. Feature Engineering (engineer_features_dynamic)

Purpose: Creates features like hour, amt_log, distance, and encoded categories.
Arguments: DataFrame, is_training flag.
Key Features: amt_log (log of amount), hour (0-23), age, gender_encoded, etc.

3. Feature Preparation (prepare_features_dynamic)

Purpose: Selects numerical features, handles missing values, prepares X and y.
Arguments: DataFrame, optional target column.
Output: X matrix, y labels, feature names list.

4. Class Imbalance Handling (handle_class_imbalance_dynamic)

Purpose: Undersamples majority class (non-fraud) based on imbalance ratio.
Arguments: X, y, strategy ('auto').
Output: Balanced dataset.

5. Model Training (train_models_adaptive)

Purpose: Trains Logistic Regression, Decision Tree, Random Forest (adaptive hyperparameters).
Arguments: Training/validation data, feature names.
Output: Trained models with metrics.

6. Model Evaluation (final_model_evaluation)

Purpose: Selects best model by AUC, evaluates on test set.
Arguments:

results: Dict of models and metrics (required).
X_test: Test features (optional).
y_test: Test labels (optional).
feature_names: Feature names list (optional, for importance plots).


Output: Scores (AUC, precision), confusion matrix, visualizations.

7. Interactive Prediction (predict_fraud_from_input)

Purpose: Allows user input for real-time fraud prediction.
Arguments:

model: Trained model (e.g., best_model).
feature_names: List of features.
scaler: Optional scaler for normalization.


Output: Probability, prediction (Yes/No), risk level.
