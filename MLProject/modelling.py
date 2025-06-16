import argparse
import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument("--x_train_path", type=str, default="heart_preprocessing/X_train.csv")
parser.add_argument("--x_test_path", type=str, default="heart_preprocessing/X_test.csv")
parser.add_argument("--y_train_path", type=str, default="heart_preprocessing/y_train.csv")
parser.add_argument("--y_test_path", type=str, default="heart_preprocessing/y_test.csv")
args = parser.parse_args()

mlflow.set_tracking_uri("https://dagshub.com/RiyZ411/msml-studi-kasus-heart.mlflow")

X_train = pd.read_csv(args.x_train_path)
X_test = pd.read_csv(args.x_test_path)
y_train = pd.read_csv(args.y_train_path).squeeze()
y_test = pd.read_csv(args.y_test_path).squeeze()

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("cohen_kappa", cohen_kappa_score(y_test, y_pred))

    # ROC AUC dengan handling biner/multikelas
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            mlflow.log_metric("roc_auc", auc)
        except Exception as e:
            print(f"ROC AUC tidak bisa dihitung: {e}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig("conf_matrix.png")
    mlflow.log_artifact("conf_matrix.png", artifact_path="plots")

    # Classification Report
    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact("classification_report.txt", artifact_path="reports")

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    with open("metric_info.json", "w") as f_json:
        json.dump(report_dict, f_json, indent=4)
    mlflow.log_artifact("metric_info.json", artifact_path="reports")

    # Model dan run ID
    mlflow.sklearn.log_model(model, "model")
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)
    mlflow.log_artifact("run_id.txt", artifact_path="metadata")