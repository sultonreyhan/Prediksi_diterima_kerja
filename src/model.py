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

# Canonical class order for confusion matrix and reports
CLASS_ORDER = ["< 1 bulan", "1 - 3 bulan", "3 - 6 bulan", "> 6 bulan"]


def build_pipeline(X: pd.DataFrame, class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(X)),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    max_iter=2000,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_multiclass(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_order: list[str] = CLASS_ORDER,
) -> dict[str, Any]:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # Use canonical class order; only include classes present in pipeline
    pipeline_classes = pipeline.classes_.tolist()
    ordered_labels = [c for c in class_order if c in pipeline_classes]

    report = classification_report(
        y_test, y_pred, labels=ordered_labels, output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_precision": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=ordered_labels
        ),
        "classification_report": report,
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_probability": y_prob,
        "classes": ordered_labels,
    }
    return metrics


def _compare_class_weight(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> str:
    """Train both class_weight options and return which wins on balanced accuracy."""
    results = {}
    for cw in [None, "balanced"]:
        p = build_pipeline(X_train, class_weight=cw)
        p.fit(X_train, y_train)
        y_pred = p.predict(X_test)
        results[str(cw)] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

    print("\n  class_weight comparison (holdout):")
    print(f"  {'Metric':<22} {'None':>8} {'balanced':>10}")
    print(f"  {'-'*42}")
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        v_none = results["None"][metric]
        v_bal = results["balanced"][metric]
        print(f"  {metric:<22} {v_none:>8.3f} {v_bal:>10.3f}")

    # Pick winner by balanced accuracy (primary), macro_f1 as tiebreaker
    none_score = results["None"]["balanced_accuracy"] + results["None"]["macro_f1"] * 0.1
    bal_score = results["balanced"]["balanced_accuracy"] + results["balanced"]["macro_f1"] * 0.1
    winner = "balanced" if bal_score >= none_score else None
    print(f"\n  Selected class_weight: {winner!r}\n")
    return str(winner) if winner else "None"


def train_models(df: pd.DataFrame) -> dict[str, Any]:
    summary = dataset_summary(df)
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"  Total dataset rows:          {summary['original_rows']}")
    print(f"  Eligible ML rows (4 classes):{summary['training_rows']}")
    print(f"  Excluded (Belum bekerja):    {summary['excluded_unknown']}")
    print(f"\n  Target distribution (ML eligible):")
    for cls in CLASS_ORDER:
        cnt = summary["target_distribution"].get(cls, 0)
        print(f"    {cls}: {cnt}")
    print(f"\n  Features used: {len(summary['features'])}")
    print(f"  Excluded leakage columns: {summary['excluded_leakage']}")
    print("=" * 50 + "\n")

    X, y = split_features_target(df)
    X, y, _ = filter_training_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test  samples: {len(X_test)}")
    print(f"  Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"  Test  class distribution: {y_test.value_counts().to_dict()}\n")

    # Compare class_weight options, pick winner
    best_cw_str = _compare_class_weight(X_train, y_train, X_test, y_test)
    best_cw: str | None = None if best_cw_str == "None" else best_cw_str

    pipeline = build_pipeline(X_train, class_weight=best_cw)
    pipeline.fit(X_train, y_train)
    evaluation = evaluate_multiclass(pipeline, X_test, y_test)

    print("TEST SET EVALUATION")
    print(f"  Train size:        {len(X_train)}")
    print(f"  Test  size:        {len(X_test)}")
    print(f"  Accuracy:          {evaluation['accuracy']:.3f}")
    print(f"  Balanced Accuracy: {evaluation['balanced_accuracy']:.3f}")
    print(f"  Macro F1:          {evaluation['macro_f1']:.3f}")
    print(f"  Weighted F1:       {evaluation['weighted_f1']:.3f}")
    print(
        f"\n{classification_report(y_test, pipeline.predict(X_test), labels=CLASS_ORDER, zero_division=0)}"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
    cv_acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    cv_bal = cross_val_score(pipeline, X, y, cv=cv, scoring="balanced_accuracy")
    print(f"  Cross-val Macro F1:         {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Cross-val Accuracy:         {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  Cross-val Balanced Acc:     {cv_bal.mean():.3f} ± {cv_bal.std():.3f}")

    # Final model: retrain on ALL eligible data
    final_pipeline = build_pipeline(X, class_weight=best_cw)
    final_pipeline.fit(X, y)

    return {
        "pipeline": final_pipeline,
        "primary_model_name": "Multiclass Logistic Regression",
        "evaluation": evaluation,
        "feature_columns": X.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "target_classes": CLASS_ORDER,
        "model_type": "multiclass_logistic_regression",
        "model_version": 3,
        "class_weight_used": best_cw_str,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "eligible_ml_rows": len(X),
        "excluded_belum_bekerja": summary["excluded_unknown"],
        "cv_macro_f1_mean": float(cv_scores.mean()),
        "cv_macro_f1_std": float(cv_scores.std()),
        "cv_accuracy_mean": float(cv_acc.mean()),
        "cv_accuracy_std": float(cv_acc.std()),
        "cv_balanced_acc_mean": float(cv_bal.mean()),
        "cv_balanced_acc_std": float(cv_bal.std()),
        "dataset_summary": summary,
    }


def save_model(artifact: dict[str, Any], path: str | Path = MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def train_and_save_model(
    dataset_path: str | Path = DATA_PATH if "DATA_PATH" in dir() else "data/Kuesioner Faktor Akademik dan Aktivitas Organisasi terhadap Kecepatan Diterima Kerja Lulusan Mahasiswa  (Responses) - Form responses 1.csv",
    model_path: str | Path = MODEL_PATH,
) -> dict[str, Any]:
    df = load_dataset(dataset_path)
    artifact = train_models(df)
    save_model(artifact, model_path)
    return artifact


if __name__ == "__main__":
    artifact = train_and_save_model()
    print(f"\nPrimary model: {artifact['primary_model_name']}")
    print(f"class_weight used: {artifact['class_weight_used']}")
