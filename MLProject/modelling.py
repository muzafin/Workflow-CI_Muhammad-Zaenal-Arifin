"""
modelling.py  (versi MLProject – dengan argparse)
Digunakan dalam MLflow Project dan GitHub Actions CI.

Author : Muhammad Zaenal Arifin
"""

import os
import contextlib
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
    f1_score, roc_auc_score,
    confusion_matrix
)
from mlflow.models.signature import infer_signature

TARGET_COL   = "quality_label"
TRAIN_PATH   = "winequality_preprocessing/train.csv"
TEST_PATH    = "winequality_preprocessing/test.csv"
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT   = "Wine_Quality_CI"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Wine Quality RF model")
    parser.add_argument("--n_estimators",      type=int, default=200)
    parser.add_argument("--max_depth",         type=int, default=12)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--max_features",      type=str, default="sqrt")
    parser.add_argument("--random_state",      type=int, default=42)
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

    # Saat dipanggil via `mlflow run .`, env var MLFLOW_RUN_ID sudah di-set
    # oleh MLflow Project runner — jangan panggil start_run() lagi.
    # Saat dijalankan langsung (python modelling.py), buat run baru.
    if os.environ.get("MLFLOW_RUN_ID"):
        # Dipanggil via mlflow run — run sudah dikelola oleh MLflow Project
        ctx = contextlib.nullcontext()
    else:
        # Dijalankan langsung
        mlflow.set_experiment(EXPERIMENT)
        ctx = mlflow.start_run(run_name="RF_CI_run")

    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    with ctx:
        params = {
            "n_estimators":      args.n_estimators,
            "max_depth":         args.max_depth,
            "min_samples_split": args.min_samples_split,
            "max_features":      args.max_features,
            "random_state":      args.random_state,
        }
        mlflow.log_params(params)

        # Train
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

        # Log model + signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            "random_forest_model",
            signature=signature,
            input_example=X_train.iloc[:5],
        )

        # Confusion matrix
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

        # Simpan model pkl
        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl")

        print(f"Run ID: {mlflow.active_run().info.run_id}")

    print("\n✓ Training selesai!")


if __name__ == "__main__":
    main()
