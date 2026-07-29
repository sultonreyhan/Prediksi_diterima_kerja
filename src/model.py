from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

try:
    from src.preprocessing import (
        TARGET_CLASSES,
        TARGET_COLUMN,
        clean_dataset,
        create_preprocessor,
        create_train_test_data,
        dataset_summary,
        filter_training_data,
        load_dataset,
        split_features_target,
    )
except ModuleNotFoundError:
    from preprocessing import (
        TARGET_CLASSES,
        TARGET_COLUMN,
        clean_dataset,
        create_preprocessor,
        create_train_test_data,
        dataset_summary,
        filter_training_data,
        load_dataset,
        split_features_target,
    )


MODEL_PATH = Path("saved_models/model.pkl")
RANDOM_STATE = 42


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(X)),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]
    )


def evaluate_multiclass(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    classes = pipeline.classes_

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes),
        "classification_report": report,
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_probability": y_prob,
        "classes": classes.tolist(),
    }
    return metrics


def train_models(df: pd.DataFrame) -> dict[str, Any]:
    summary = dataset_summary(df)
    print("\n" + "=" * 45)
    print("DATASET SUMMARY")
    print("=" * 45)
    print(f"  Original rows:                    {summary['original_rows']}")
    print(f"  Excluded unknown waiting time:    {summary['excluded_unknown']}")
    print(f"  Rows used for training:           {summary['training_rows']}")
    print(f"\n  Target distribution:")
    for cls, cnt in sorted(summary['target_distribution'].items()):
        print(f"    {cls}: {cnt}")
    print(f"\n  Features: {len(summary['features'])}")
    print(f"  Excluded leakage columns: {summary['excluded_leakage']}")
    print("=" * 45 + "\n")

    X, y = split_features_target(df)
    X, y, _ = filter_training_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    evaluation = evaluate_multiclass(pipeline, X_test, y_test)

    print("TEST SET EVALUATION")
    print(f"  Accuracy:         {evaluation['accuracy']:.3f}")
    print(f"  Balanced Accuracy: {evaluation['balanced_accuracy']:.3f}")
    print(f"  Macro F1:         {evaluation['macro_f1']:.3f}")
    print(f"  Weighted F1:      {evaluation['weighted_f1']:.3f}")
    print(f"\n{classification_report(y_test, pipeline.predict(X_test), zero_division=0)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
    cv_acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"  Cross-val Macro F1:  {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  Cross-val Accuracy:  {cv_acc.mean():.3f} (+/- {cv_acc.std():.3f})")

    final_pipeline = build_pipeline(X)
    final_pipeline.fit(X, y)

    return {
        "pipeline": final_pipeline,
        "primary_model_name": "Multiclass Logistic Regression",
        "evaluation": evaluation,
        "feature_columns": X.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "target_classes": TARGET_CLASSES,
        "model_type": "multiclass_logistic_regression",
        "model_version": 2,
        "cv_macro_f1_mean": float(cv_scores.mean()),
        "cv_macro_f1_std": float(cv_scores.std()),
        "cv_accuracy_mean": float(cv_acc.mean()),
        "cv_accuracy_std": float(cv_acc.std()),
        "dataset_summary": summary,
    }


def save_model(artifact: dict[str, Any], path: str | Path = MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def train_and_save_model(
    dataset_path: str | Path = "data/Kuesioner Faktor Akademik dan Aktivitas Organisasi terhadap Kecepatan Diterima Kerja Lulusan Mahasiswa  (Responses) - Form responses 1.csv",
    model_path: str | Path = MODEL_PATH,
) -> dict[str, Any]:
    df = load_dataset(dataset_path)
    artifact = train_models(df)
    save_model(artifact, model_path)
    return artifact


if __name__ == "__main__":
    artifact = train_and_save_model()
    print(f"\nPrimary model: {artifact['primary_model_name']}")
