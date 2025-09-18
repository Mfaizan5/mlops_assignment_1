import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils import plot_confusion_matrix

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel="linear", probability=True)
}

# Train & log with MLflow
mlflow.set_experiment("mlops-assignment-1")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

       
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # parameters & metrics
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Save themodel
        os.makedirs("../models", exist_ok=True)
        model_path = f"../models/{name}.pkl"
        mlflow.sklearn.save_model(model, model_path)

        # confusion matrix
        os.makedirs("../results", exist_ok=True)
        cm_path = f"../results/{name}_cm.png"
        plot_confusion_matrix(y_test, y_pred, name, cm_path)
        mlflow.log_artifact(cm_path)

        print(f"{name} logged with accuracy: {acc:.4f}")
