"""
modelling.py  (versi MLProject – tanpa start_run internal)
Digunakan dalam MLflow Project dan GitHub Actions CI.
"""

import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

TARGET_COL   = "quality_label"
TRAIN_PATH   = "winequality_preprocessing/train.csv"
TEST_PATH    = "winequality_preprocessing/test.csv"
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT   = "Wine_Quality_CI"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Wine Quality RF model")
    parser.add_argument("--n_estimators",     type=int,   default=200)
    parser.add_argument("--max_depth",        type=int,   default=12)
    parser.add_argument("--min_samples_split",type=int,   default=2)
    parser.add_argument("--max_features",     type=str,   default="sqrt")
    parser.add_argument("--random_state",     type=int,   default=42)
    return parser.parse_args()


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test  = test.drop(columns=[TARGET_COL])
    y_test  = test[TARGET_COL]
    return X_train, X_test, y_train, y_test


def main():
    args = parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Parameter model
    params = {
        "n_estimators":      args.n_estimators,
        "max_depth":         args.max_depth,
        "min_samples_split": args.min_samples_split,
        "max_features":      args.max_features,
        "random_state":      args.random_state,
    }
    mlflow.log_params(params)

    # Training
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("roc_auc",   auc)

    print(f"\nAccuracy: {acc:.4f} | Precision: {prec:.4f} | "
          f"Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # Log model MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["bad", "good"],
                yticklabels=["bad", "good"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Simpan model.pkl lokal
    model_file = "model.pkl"
    joblib.dump(model, model_file)
    mlflow.log_artifact(model_file)

    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    print("\n✓ Training selesai!")


if __name__ == "__main__":
    main()
