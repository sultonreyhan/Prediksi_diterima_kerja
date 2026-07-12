"""Preprocessing utilities for the employability prediction project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/dummy_dataset_employability.xlsx")
TARGET_COLUMN = "Status Pekerjaan"
POSITIVE_LABEL = "Sudah bekerja"

LEAKAGE_COLUMNS = [
    "Waktu Tunggu Kerja",
    "Kesesuaian Pekerjaan",
    "Kisaran Gaji",
]

FEATURE_COLUMNS = [
    "Tahun Lulus",
    "Rumpun Jurusan",
    "Jenjang Pendidikan",
    "Kategori IPK",
    "Lama Masa Studi",
    "Pernah Magang",
    "Memiliki Sertifikasi",
    "Memiliki Prestasi",
    "Aktif Organisasi",
    "Jenis Organisasi",
    "Jabatan Organisasi",
    "Tingkat Keaktifan Organisasi (1-5)",
    "Jumlah Organisasi",
]


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the Excel dataset and normalize column names lightly."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    return df


def build_target(series: pd.Series) -> pd.Series:
    """Convert employment status into a binary target for employability."""
    return (series.astype(str).str.strip() == POSITIVE_LABEL).astype(int)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model-ready features and binary target."""
    available_features = [column for column in FEATURE_COLUMNS if column in df.columns]
    X = df[available_features].copy()
    y = build_target(df[TARGET_COLUMN])
    return X, y


def get_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical feature groups."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing transformer for mixed tabular data."""
    numeric_features, categorical_features = get_feature_groups(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def create_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets with stratification when possible."""
    X, y = split_features_target(df)
    stratify = y if y.nunique() > 1 else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def get_input_options(df: pd.DataFrame) -> dict[str, list]:
    """Collect sorted options for Streamlit form controls."""
    options: dict[str, list] = {}
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            continue

        values = df[column].dropna().unique().tolist()
        if pd.api.types.is_numeric_dtype(df[column]):
            options[column] = sorted(values)
        else:
            options[column] = sorted(values, key=lambda value: str(value))
    return options


def build_candidate_dataframe(candidate: dict) -> pd.DataFrame:
    """Create a one-row DataFrame from form input using the expected column order."""
    return pd.DataFrame([{column: candidate.get(column) for column in FEATURE_COLUMNS}])
