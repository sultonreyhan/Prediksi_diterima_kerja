"""Training and evaluation logic for the employability prediction model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

try:
    from src.preprocessing import (
        POSITIVE_LABEL,
        create_preprocessor,
        create_train_test_data,
        load_dataset,
        split_features_target,
    )
except ModuleNotFoundError:
    from preprocessing import (
        POSITIVE_LABEL,
        create_preprocessor,
        create_train_test_data,
        load_dataset,
        split_features_target,
    )


MODEL_PATH = Path("saved_models/model.pkl")
RANDOM_STATE = 42


def get_candidate_models() -> dict[str, Any]:
    """Return baseline models suitable for a small tabular dataset."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def build_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    """Build a full preprocessing and model pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(X)),
            ("model", estimator),
        ]
    )


def _safe_roc_auc(y_true: pd.Series, y_probability: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_probability))


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute classification metrics and curve data."""
    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]
    labels = [0, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_test, y_probability),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_probability": y_probability,
    }

    if y_test.nunique() > 1:
        fpr, tpr, thresholds = roc_curve(y_test, y_probability)
        metrics["roc_curve"] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }
    else:
        metrics["roc_curve"] = None

    return metrics


def train_models(df: pd.DataFrame) -> dict[str, Any]:
    """Train several models and return artifact with Logistic Regression as primary model."""
    X_train, X_test, y_train, y_test = create_train_test_data(df)
    comparison_rows = []
    trained_models: dict[str, Pipeline] = {}
    evaluations: dict[str, dict[str, Any]] = {}

    for model_name, estimator in get_candidate_models().items():
        pipeline = build_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        evaluation = evaluate_pipeline(pipeline, X_test, y_test)

        trained_models[model_name] = pipeline
        evaluations[model_name] = evaluation
        comparison_rows.append(
            {
                "Model": model_name,
                "Accuracy": evaluation["accuracy"],
                "Precision": evaluation["precision"],
                "Recall": evaluation["recall"],
                "F1-score": evaluation["f1"],
                "ROC AUC": evaluation["roc_auc"],
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        by=["F1-score", "Accuracy"],
        ascending=False,
    )
    X, y = split_features_target(df)

    primary_model_name = "Logistic Regression"
    lr_estimator = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    final_pipeline = build_pipeline(X, lr_estimator)
    final_pipeline.fit(X, y)

    return {
        "pipeline": final_pipeline,
        "primary_model_name": primary_model_name,
        "comparison": comparison,
        "evaluation": evaluations[primary_model_name],
        "evaluations": evaluations,
        "feature_columns": X.columns.tolist(),
        "target_label": POSITIVE_LABEL,
    }


def save_model(artifact: dict[str, Any], path: str | Path = MODEL_PATH) -> None:
    """Persist the trained model artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Load a trained model artifact."""
    return joblib.load(path)


def train_and_save_model(
    dataset_path: str | Path = "data/Kuesioner Faktor Akademik dan Aktivitas Organisasi terhadap Kecepatan Diterima Kerja Lulusan Mahasiswa  (Responses) - Form responses 1.csv",
    model_path: str | Path = MODEL_PATH,
) -> dict[str, Any]:
    """Train models from the dataset and persist the best one."""
    df = load_dataset(dataset_path)
    artifact = train_models(df)
    save_model(artifact, model_path)
    return artifact


if __name__ == "__main__":
    model_artifact = train_and_save_model()
    print(f"Primary model: {model_artifact['primary_model_name']}")
    print(model_artifact["comparison"].to_string(index=False))
