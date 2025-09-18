import os
import sys
import shutil
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# make sure Python can find utils.py
sys.path.append(os.path.dirname(__file__))
from utils import plot_confusion_matrix

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear", probability=True)
}

# Train & log with MLflow
mlflow.set_experiment("mlops-assignment-1")

results = {}  # to keep track of all models and accuracy

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        results[name] = acc  # store accuracy for comparison

        # Log parameters & metrics
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Save model to /models folder
        os.makedirs(os.path.join("..", "models"), exist_ok=True)
        model_dir = os.path.join("..", "models", name)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        mlflow.sklearn.save_model(model, model_dir)

        # Also save as .pkl
        pkl_path = os.path.join("..", "models", f"{name}.pkl")
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
        joblib.dump(model, pkl_path)

        # Log confusion matrix
        os.makedirs(os.path.join("..", "results"), exist_ok=True)
        cm_path = os.path.join("..", "results", f"{name}_cm.png")
        plot_confusion_matrix(y_test, y_pred, name, cm_path)
        mlflow.log_artifact(cm_path)

        # Log model into MLflow run (so it's tracked)
        mlflow.sklearn.log_model(model, name)

        print(f"{name} logged with accuracy: {acc:.4f}")

# After training all models → find best
best_model_name = max(results, key=results.get)
best_acc = results[best_model_name]
print(f"\n Best Model: {best_model_name} with accuracy {best_acc:.4f}")

# Register best model in MLflow Model Registry
with mlflow.start_run(run_name=f"{best_model_name}-registry"):
    best_model = models[best_model_name]
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="Best_Model_Assignment1"
    )
    mlflow.log_metric("best_accuracy", best_acc)

print("\n Best model registered in MLflow Model Registry as 'Best_Model_Assignment1'")
