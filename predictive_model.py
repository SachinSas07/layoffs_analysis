# Phase 3 is the Predictive Model phase, where we will use the trained model to make predictions on new data.
# This script builds and evaluates two classification models to predict
# whether a stock will go up or down 30 days after a tech layoff announcement 

# MODELS
# Logistic Regression (baseline)
# Random Forest Classifier (advanced)

# Features Used 
# total_laid_off : raw number of employees laid off
# percentage_laid_off : percentage of workforce laid off
# funds_raised : total funding raised by the company
# industry_encoded : industry label encoded as a number
# stage_encoded : stage of the company label encoded as a number

# Output
# Accuracy, Precision, Recall, F1-Score for both models
# Confusion Matrices
# Feature Importance Chart (Random Forest)
# Save model_results.csv for use in streamlit app

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os
import warnings
warnings.filterwarnings("ignore")

# Load dataset produced by phase1_cleaning.py
df = pd.read_csv("layoffs_with_stocks.csv", parse_dates=["date"])
print(f"Loaded {len(df)} rows")

# Feature Engineering - Select the relevant features used to predict whether the stock will go up or down

# Encode categorical features
# We will encode the 'industry' and 'stage' features using Label Encoding, which converts each unique category into a number.
le_industry = LabelEncoder()
le_stage = LabelEncoder()

# Fills nulls with 'Unknown' before encoding so LabelEncoder doesn't crash
df["industry"] = df["industry"].fillna("Unknown")
df["stage"] = df["stage"].fillna("Unknown")

df["industry_encoded"] = le_industry.fit_transform(df["industry"])
df["stage_encoded"] = le_stage.fit_transform(df["stage"])

# Fill nulls in numerical features with 0 (or you could choose to fill with mean/median)
df["percentage_laid_off"] = df["percentage_laid_off"].fillna(df["percentage_laid_off"].median())
df["funds_raised"] = df["funds_raised"].fillna(df["funds_raised"].median())

# Define features and target variable
features = [
    "total_laid_off",
    "percentage_laid_off",
    "funds_raised",
    "industry_encoded",
    "stage_encoded",
]
TARGET = "target"

# Drop any remaining rows with nulls in features or target
model_df = df[features + [TARGET]].dropna()
print(f"Rows Available for Modeling: {len(model_df)}")

X = model_df[features]
y = model_df[TARGET]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training Rows: {len(X_train)} | Test Rows: {len(X_test)}")

# Model 1: Logistic Regression (Baseline)
# Models the probability of the stock going up (1) or down (0) based on the features.
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Evaluate using cross-validation for more reliable accuracy measurement 
lr_cv_scores = cross_val_score(lr, X, y, cv=5, scoring="accuracy")

print("\n--- Logistic Regression ------------------------------")
print(f"Test Accuracy       : {accuracy_score(y_test, lr_preds):.2%}")
print(f"Cross-Val Accuracy  : {lr_cv_scores.mean():.2%} ± {lr_cv_scores.std():.2%}")
print(f"Precision           : {precision_score(y_test, lr_preds):.2%}")
print(f"Recall              : {recall_score(y_test, lr_preds):.2%}")
print(f"F1-Score            : {f1_score(y_test, lr_preds):.2%}")

# Model 2: Random Forest Classifier (Advanced)
# Builds decision trees on random subsets of the data and features, then averages their predictions for better accuracy and robustness.
# Generally performs better than logistic regression on complex datasets with non-linear relationships.
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Evaluate using cross-validation for more reliable accuracy measurement
rf_cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")

print("\n--- Random Forest Classifier -----------------------")
print(f"Test Accuracy       : {accuracy_score(y_test, rf_preds):.2%}")
print(f"Cross-Val Accuracy  : {rf_cv_scores.mean():.2%} ± {rf_cv_scores.std():.2%}")
print(f"Precision           : {precision_score(y_test, rf_preds):.2%}")
print(f"Recall              : {recall_score(y_test, rf_preds):.2%}")
print(f"F1-Score            : {f1_score(y_test, rf_preds):.2%}")

# Visualize Confusion Matrices
os.makedirs("visuals", exist_ok=True)

# Chart 1: Confusion Matrices Side by Side for Logistic Regression and Random Forest
# Shows how many predictions were true positives, true negatives, false positives, and false negatives for both models.
# Rows = actual labels, Columns = predicted labels
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, preds, title in zip(
    axes,
    [lr_preds, rf_preds],
    ["Logistic Regression", "Random Forest"],
):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stock Down (0)", "Stock Up (1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13)

plt.suptitle("Confusion Matrices - Logistic Regression vs Random Forest", fontsize=16)
plt.tight_layout()
plt.savefig("visuals/chart7_confusion_matrices.png", dpi=150)
plt.show()
print("Chart 7 Saved!: Confusion Matrices for Logistic Regression and Random Forest")

# Chart 2: Feature Importance from Random Forest
# This tells us which inputs were most useful for predicting whether a stock went up or down
importances = pd.Series(rf.feature_importances_, index=features).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2a9d8f" if v >= importances.median() else "#457b9d" for v in importances.values]
ax.barh(importances.index, importances.values, color=colors, edgecolor="white")
ax.set_title("Feature Importance - Random Forest")
ax.set_xlabel("Importance Score")
ax.set_facecolor("#f9f9f9")
ax.grid(True, color="#dddddd")
ax.spines["top"].set_visible(False)
plt.tight_layout()
plt.savefig("visuals/chart8_rf_feature_importance.png", dpi=150)
plt.show()
print("Chart 8 Saved!: Feature Importance from Random Forest")

# Chart 3: Comparison of Model Performance Metrics (Bar Chart)
# Compares accuracy, precision, recall, and F1-score for both models in a single chart for easy comparison.

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
lr_scores = [
    accuracy_score(y_test, lr_preds),
    precision_score(y_test, lr_preds),
    recall_score(y_test, lr_preds),
    f1_score(y_test, lr_preds),
]
rf_scores = [
    accuracy_score(y_test, rf_preds),
    precision_score(y_test, rf_preds),
    recall_score(y_test, rf_preds),
    f1_score(y_test, rf_preds),
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width / 2, lr_scores, width, label="Logistic Regression", color="#457b9d", edgecolor="white")
bars2 = ax.bar(x + width / 2, rf_scores, width, label="Random Forest", color="#2a9d8f", edgecolor="white")

# Values on top of each bar 
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, 
            f"{bar.get_height():.2%}", ha="center", fontsize=9)

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, 
            f"{bar.get_height():.2%}", ha="center", fontsize=9)
    
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.15)
ax.set_title("Model Comparison - Logistic Regression vs Random Forest")
ax.legend()
ax.set_facecolor("#f9f9f9")
ax.grid(True, color="#dddddd", axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("visuals/chart9_model_comparison.png", dpi=150)
plt.show()
print("Chart 9 Saved!: Comparison of Model Performance Metrics")

# Save model results to CSV for use in Streamlit app
results_df = X_test.copy()
results_df["actual"] = y_test.values
results_df["lr_pred"] = lr_preds
results_df["rf_pred"] = rf_preds
results_df.to_csv("model_results.csv", index=False)
print("\nSaved to model_results.csv for Streamlit app!")

print("\nPredictive Modeling Phase Complete! Charts saved in 'visuals' folder and model results saved to 'model_results.csv'.")
