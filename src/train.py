import pandas as pd
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

DATA_PATH = Path("data/processed/readmission_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run: python src/data_prep.py")

    df = pd.read_csv(DATA_PATH)

    target = "readmitted_30d"
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    if "PATIENT" in X.columns:
        X = X.drop(columns=["PATIENT"])

    cat_features = ["encounter_class"]
    num_features = [
        "encounter_length_hours",
        "conditions_365d",
        "unique_conditions_365d",
        "meds_365d",
        "unique_meds_365d",
    ]
    # In case some columns are missing (if you don't have conditions/meds)
    num_features = [c for c in num_features if c in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ]
    )

    max_iter = 2000
    C = 1.0
    threshold = 0.5

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=max_iter, C=C))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("readmission-risk")
    with mlflow.start_run():
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("num_features", ",".join(num_features))
        mlflow.log_param("cat_features", ",".join(cat_features))
        mlflow.log_param("rows", len(df))

        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        auc = roc_auc_score(y_test, probs)
        ap = average_precision_score(y_test, probs)

        report = classification_report(y_test, preds)

        print(f"ROC-AUC: {auc:.4f}")
        print(f"Avg Precision: {ap:.4f}")
        print(report)

        mlflow.log_metric("roc_auc", float(auc))
        mlflow.log_metric("avg_precision", float(ap))

        # Save model locally (for FastAPI)
        model_path = MODEL_DIR / "readmission_model.joblib"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_local")

        # Also log model to MLflow
        mlflow.sklearn.log_model(clf, artifact_path="model")

        # Log a text report artifact
        report_path = MODEL_DIR / "classification_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        print(f"Saved model to: {model_path}")
        print("Logged run to MLflow.")

if __name__ == "__main__":
    main()