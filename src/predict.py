from __future__ import annotations

import pandas as pd

from src.preprocessing import (
    FEATURE_COLUMNS,
    build_candidate_dataframe,
    split_jenis_string,
    validate_jumlah_jenis_binding,
)


def _validate_candidate(candidate: dict) -> str | None:
    jumlah = str(candidate.get("Jumlah Organisasi", "0"))
    jenis = str(candidate.get("Jenis Organisasi", "Tidak mengikuti organisasi"))
    aktif = str(candidate.get("Aktif Organisasi", "Tidak"))
    jabatan = str(candidate.get("Jabatan Organisasi", "Tidak pernah mengikuti organisasi"))
    tingkat = candidate.get("Tingkat Keaktifan Organisasi (1-5)", 1)

    valid, msg = validate_jumlah_jenis_binding(jumlah, jenis)
    if not valid:
        return msg

    if jumlah == "0":
        if jenis != "Tidak mengikuti organisasi":
            return "Jumlah Organisasi 0 harus memiliki Jenis Organisasi = Tidak mengikuti organisasi."
        if jabatan != "Tidak pernah mengikuti organisasi":
            return "Jumlah Organisasi 0 harus memiliki Jabatan Organisasi = Tidak pernah mengikuti organisasi."
        if aktif != "Tidak":
            return "Jumlah Organisasi 0 harus memiliki Aktif Organisasi = Tidak."

    if aktif == "Tidak" and jumlah != "0":
        selected = split_jenis_string(jenis)
        if len(selected) > 0:
            pass

    if aktif == "Tidak":
        if tingkat != 1:
            return "Aktif Organisasi = Tidak harus memiliki Tingkat Keaktifan = 1."

    if aktif == "Ya" and jumlah == "0":
        return "Jumlah Organisasi 0 tidak boleh memiliki Aktif Organisasi = Ya."

    return None


def predict_single(artifact: dict, candidate: dict) -> dict:
    error = _validate_candidate(candidate)
    if error:
        return {"error": error}

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
