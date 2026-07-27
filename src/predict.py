"""Prediction helpers for single and batch employability inference."""

from __future__ import annotations

import pandas as pd

from src.preprocessing import FEATURE_COLUMNS, build_candidate_dataframe


def _validate_candidate(candidate: dict) -> None:
    jumlah = candidate.get("Jumlah Organisasi", "0")
    if jumlah == "0":
        assert candidate.get("Jenis Organisasi") == "Tidak mengikuti organisasi", (
            "Jumlah Organisasi 0 tidak sesuai dengan Jenis Organisasi"
        )
        assert candidate.get("Jabatan Organisasi") == "Tidak pernah mengikuti organisasi", (
            "Jumlah Organisasi 0 tidak sesuai dengan Jabatan Organisasi"
        )
        assert candidate.get("Aktif Organisasi") == "Tidak", (
            "Jumlah Organisasi 0 harus memiliki Aktif Organisasi = Tidak"
        )
    aktif = candidate.get("Aktif Organisasi", "Tidak")
    tingkat = candidate.get("Tingkat Keaktifan Organisasi (1-5)", 1)
    if aktif == "Tidak":
        assert tingkat == 1, (
            "Aktif Organisasi = Tidak tidak sesuai dengan Tingkat Keaktifan"
        )


def predict_single(artifact: dict, candidate: dict) -> dict:
    """Predict employability for one candidate."""
    _validate_candidate(candidate)
    candidate_df = build_candidate_dataframe(candidate)
    probability = float(artifact["pipeline"].predict_proba(candidate_df)[:, 1][0])
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "label": "Berpotensi diterima kerja" if prediction == 1 else "Perlu penguatan profil",
        "probability": probability,
    }


def predict_batch(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Predict employability for uploaded candidate rows."""
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Kolom wajib belum ada: {', '.join(missing_columns)}")

    prediction_df = df.copy()
    probabilities = artifact["pipeline"].predict_proba(prediction_df[FEATURE_COLUMNS])[:, 1]
    prediction_df["Employability Probability"] = probabilities
    prediction_df["Prediction"] = [
        "Berpotensi diterima kerja" if probability >= 0.5 else "Perlu penguatan profil"
        for probability in probabilities
    ]
    return prediction_df
