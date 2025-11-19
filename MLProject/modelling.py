import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

import mlflow
import mlflow.sklearn

PROJECT_ROOT = Path(__file__).resolve().parent


def load_data(data_path: str):
    path = PROJECT_ROOT / data_path
    print(f"[INFO] Load data: {path}")
    df = pd.read_csv(path)

    if "primary_genre" not in df.columns:
        raise ValueError("Kolom 'primary_genre' tidak ditemukan")

    if "text" not in df.columns:
        def combine_text(row):
            parts = []
            for col in ["title", "description", "tags"]:
                v = row.get(col, "")
                if isinstance(v, str):
                    parts.append(v)
            return " ".join(parts)
        df["text"] = df.apply(combine_text, axis=1)

    df = df.dropna(subset=["primary_genre"]).copy()
    X = df["text"]
    y = df["primary_genre"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main(data_path: str):
    X_train, X_test, y_train, y_test = load_data(data_path)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english",
        )),
        ("clf", LinearSVC()),
    ])

    with mlflow.start_run(run_name="ci_training_run"):
        mlflow.log_param("max_features", 30000)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("C", 1.0)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        print(f"[TEST] acc={acc:.4f}, f1_macro={f1_macro:.4f}")
        print(classification_report(y_test, y_pred))

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_macro", f1_macro)

        # artefak tambahan (advanced)
        report_txt = classification_report(y_test, y_pred)
        with open("test_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report_txt)
        mlflow.log_artifact("test_classification_report.txt")

        mlflow.sklearn.log_model(model, artifact_path="model")

        # simpan model ke file (optional)
        out_dir = PROJECT_ROOT / "models"
        out_dir.mkdir(exist_ok=True)
        joblib.dump(model, out_dir / "tfidf_svc_ci.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data_preprocessing/videos_preprocessed.csv")
    args = parser.parse_args()
    main(args.data_path)
