from __future__ import annotations

import pandas as pd

from src.preprocessing import FEATURE_COLUMNS, build_candidate_dataframe


def _validate_candidate(candidate: dict) -> None:
    jumlah = str(candidate.get("Jumlah Organisasi", "0"))
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
    _validate_candidate(candidate)
    candidate_df = build_candidate_dataframe(candidate)

    pipeline = artifact["pipeline"]
    classes = pipeline.classes_
    probabilities = pipeline.predict_proba(candidate_df)[0]
    predicted_class = pipeline.predict(candidate_df)[0]

    probability_by_class = {
        str(cls): float(prob)
        for cls, prob in zip(classes, probabilities)
    }

    confidence = float(max(probabilities))

    return {
        "predicted_class": str(predicted_class),
        "confidence": confidence,
        "probabilities": probability_by_class,
    }


def predict_batch(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Kolom wajib belum ada: {', '.join(missing_columns)}")

    pipeline = artifact["pipeline"]
    classes = pipeline.classes_
    probs = pipeline.predict_proba(df[FEATURE_COLUMNS])

    result_df = df.copy()
    for i, cls in enumerate(classes):
        result_df[f"Prob_{cls}"] = probs[:, i]

    result_df["Predicted Class"] = pipeline.predict(df[FEATURE_COLUMNS])
    result_df["Confidence"] = probs.max(axis=1)
    return result_df
