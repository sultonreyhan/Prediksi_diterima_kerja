from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


DATA_PATH = Path(
    "data/Kuesioner Faktor Akademik dan Aktivitas Organisasi terhadap Kecepatan Diterima Kerja Lulusan Mahasiswa  (Responses) - Form responses 1.csv"
)
TARGET_COLUMN = "Waktu Tunggu Kerja"
TARGET_CLASSES = ["< 1 bulan", "1 - 3 bulan", "3 - 6 bulan", "> 6 bulan"]

LEAKAGE_COLUMNS = [
    "Status Pekerjaan",
    "Kesesuaian Pekerjaan",
    "Kisaran Gaji",
]

# Jenis Organisasi multi-hot: canonical names exactly as they appear in the data
JENIS_ORGANISASI_LIST = [
    "BEM (Badan Eksekutif Mahasiswa)",
    "Himpunan Mahasiswa",
    "Kepanitaan Acara",
    "Komunitas Kampus",
    "TSR PMI",
    "UKM (Unit Kegiatan Mahasiswa)",
]

# Short feature name prefix for each org (for readability in coefficient chart)
JENIS_ORGANISASI_FEATURE_NAMES = [
    "org_BEM",
    "org_Himpunan",
    "org_Kepanitiaan",
    "org_Komunitas",
    "org_TSR_PMI",
    "org_UKM",
]

JUMLAH_ORGANISASI_OPTIONS = ["0", "1", "2", ">2"]
JENIS_SEPARATOR = ", "

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

ORDINAL_COLUMNS: dict[str, list] = {
    "Kategori IPK": ["< 3.00", "3.00 - 3.25", "3.26 - 3.50", "> 3.50"],
    "Lama Masa Studi": ["< 4 tahun", "4 - 5 tahun", "> 5 tahun"],
    # Tingkat Keaktifan is integer 1-5; handled as numeric (StandardScaler)
    "Jumlah Organisasi": ["0", "1", "2", ">2"],
}

# Binary (Ya/Tidak) + nominal features — use OneHotEncoder
NOMINAL_COLUMNS = [
    "Rumpun Jurusan",
    "Jenjang Pendidikan",
    "Pernah Magang",
    "Memiliki Sertifikasi",
    "Memiliki Prestasi",
    "Aktif Organisasi",
    "Jabatan Organisasi",
]

COLUMN_MAPPING = {
    "Tahun Lulus Kuliah": "Tahun Lulus",
    "Pilih Rumpun Jurusan": "Rumpun Jurusan",
    "Jenjang Pendidikan": "Jenjang Pendidikan",
    "Berapa kategori IPK Anda saat lulus?": "Kategori IPK",
    "Berapa lama masa studi Anda": "Lama Masa Studi",
    "Apakah Anda pernah mengikuti program magang": "Pernah Magang",
    "Apakah Anda memiliki sertifikasi kompetensi": "Memiliki Sertifikasi",
    "Apakah Anda pernah memiliki prestasi akademik/non-akademik selama kuliah?": "Memiliki Prestasi",
    "Apakah Anda aktif mengikuti organisasi selama kuliah?": "Aktif Organisasi",
    "Organisasi apa yang pernah Anda ikuti?": "Jenis Organisasi",
    "Jabatan tertinggi yang pernah Anda pegang di organisasi?": "Jabatan Organisasi",
    "Seberapa aktif Anda dalam kegiatan organisasi": "Tingkat Keaktifan Organisasi (1-5)",
    "Berapa jumlah organisasi yang pernah Anda ikuti": "Jumlah Organisasi",
    "Apakah Anda saat ini sudah bekerja": "Status Pekerjaan",
    "Berapa lama waktu yang Anda butuhkan untuk mendapatkan pekerjaan pertama setelah lulus?": "Waktu Tunggu Kerja",
    "Apakah pekerjaan Anda saat ini sesuai dengan bidang kuliah?": "Kesesuaian Pekerjaan",
    "Kisaran gaji pertama Anda setelah bekerja": "Kisaran Gaji",
}

DROP_COLUMNS = ["Timestamp", "Saran", "Email address"]


# ---------------------------------------------------------------------------
# Multi-hot encoder for Jenis Organisasi
# ---------------------------------------------------------------------------

class JenisOrganisasiMultiHotEncoder(BaseEstimator, TransformerMixin):
    """
    Converts the 'Jenis Organisasi' comma-separated string column into
    multi-hot binary indicator columns — one per canonical org type.

    Input column may contain strings like:
        "BEM (Badan Eksekutif Mahasiswa), Himpunan Mahasiswa"
    or the sentinel value:
        "Tidak mengikuti organisasi"

    Output: len(JENIS_ORGANISASI_LIST) binary columns, one per org type.
    """

    def __init__(
        self,
        org_list: list[str] = JENIS_ORGANISASI_LIST,
        feature_names: list[str] = JENIS_ORGANISASI_FEATURE_NAMES,
    ) -> None:
        self.org_list = org_list
        self.feature_names = feature_names

    def fit(self, X: Any, y: Any = None) -> "JenisOrganisasiMultiHotEncoder":
        return self  # stateless

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        else:
            series = pd.Series(X.ravel())

        result = np.zeros((len(series), len(self.org_list)), dtype=np.float64)
        for i, val in enumerate(series):
            memberships = _parse_jenis(str(val))
            for j, org in enumerate(self.org_list):
                if org in memberships:
                    result[i, j] = 1.0
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.feature_names)


def _parse_jenis(jenis_str: str) -> set[str]:
    """Return set of org names from a comma-separated Jenis Organisasi string."""
    if not jenis_str or jenis_str.strip() in ("Tidak mengikuti organisasi", "nan", ""):
        return set()
    parts = [p.strip() for p in jenis_str.split(",")]
    return {p for p in parts if p}


def normalize_jenis_string(selected_types: list[str]) -> str:
    """Normalize a list of selected org types to a canonical sorted string."""
    cleaned = [t.strip() for t in selected_types if t.strip()]
    deduped = sorted(set(cleaned))
    if not deduped:
        return "Tidak mengikuti organisasi"
    return JENIS_SEPARATOR.join(deduped)


def split_jenis_string(jenis_str: str) -> list[str]:
    if jenis_str == "Tidak mengikuti organisasi":
        return []
    return [t.strip() for t in jenis_str.split(JENIS_SEPARATOR) if t.strip()]


def validate_jumlah_jenis_binding(jumlah: str, jenis_str: str) -> tuple[bool, str]:
    selected = split_jenis_string(jenis_str)
    count = len(selected)

    if jumlah == "0":
        if count != 0:
            return False, "Jumlah organisasi 0 tidak boleh memiliki jenis organisasi."
        return True, ""
    elif jumlah == "1":
        if count < 1:
            return False, "Jumlah organisasi adalah 1. Pilih tepat 1 jenis organisasi."
        if count > 1:
            return False, f"Jumlah organisasi adalah 1. Pilih tepat 1 jenis organisasi (saat ini {count})."
        return True, ""
    elif jumlah == "2":
        if count < 2:
            return False, f"Jumlah organisasi adalah 2. Pilih tepat 2 jenis organisasi (saat ini {count})."
        if count > 2:
            return False, f"Jumlah organisasi adalah 2. Pilih tepat 2 jenis organisasi (saat ini {count})."
        return True, ""
    elif jumlah == ">2":
        if count < 3:
            return False, f"Jumlah organisasi lebih dari 2. Pilih minimal 3 jenis organisasi (saat ini {count})."
        return True, ""
    else:
        return False, f"Nilai Jumlah Organisasi tidak dikenal: {jumlah}"


# ---------------------------------------------------------------------------
# Dataset loading & cleaning
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    df = df.rename(columns=COLUMN_MAPPING)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    masa_studi_map = {
        "2,5": "< 4 tahun",
        "3 tahun": "< 4 tahun",
    }
    if "Lama Masa Studi" in df.columns:
        df["Lama Masa Studi"] = df["Lama Masa Studi"].replace(masa_studi_map)

    # Normalize Jumlah Organisasi to canonical values
    if "Jumlah Organisasi" in df.columns:
        df["Jumlah Organisasi"] = (
            df["Jumlah Organisasi"]
            .astype(str)
            .str.strip()
            .replace({"> 2": ">2", "> 1": ">2", ">1": ">2", "> 3": ">2"})
        )

    unk_masa = df.loc[
        df["Lama Masa Studi"].notna()
        & ~df["Lama Masa Studi"].isin(ORDINAL_COLUMNS["Lama Masa Studi"]),
        "Lama Masa Studi",
    ]
    if len(unk_masa) > 0:
        print(f"  [WARN] Unmapped Lama Masa Studi values: {unk_masa.unique().tolist()}")

    return df


def build_target(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = clean_dataset(df)
    available_features = [column for column in FEATURE_COLUMNS if column in df.columns]
    X = df[available_features].copy()
    y = build_target(df[TARGET_COLUMN])
    return X, y


def filter_training_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, int]:
    mask = y.isin(TARGET_CLASSES)
    X_train = X[mask].copy()
    y_train = y[mask].copy()
    excluded = int((~mask).sum())
    return X_train, y_train, excluded


# ---------------------------------------------------------------------------
# Preprocessor construction
# ---------------------------------------------------------------------------

def get_feature_groups(
    X: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], bool]:
    """
    Returns (numeric_features, ordinal_features, nominal_features, has_jenis)

    'Tingkat Keaktifan Organisasi (1-5)' is integer → numeric (scaled).
    'Jenis Organisasi' gets its own multi-hot transformer.
    'Jumlah Organisasi' is ordinal (string categories).
    """
    jenis_col = "Jenis Organisasi"
    has_jenis = jenis_col in X.columns

    ordinal_features = [
        c for c in ORDINAL_COLUMNS if c in X.columns and c != jenis_col
    ]

    nominal_features = [c for c in NOMINAL_COLUMNS if c in X.columns]

    # Numeric: genuine int/float columns NOT in ordinal list
    numeric_features = [
        c for c in X.select_dtypes(include=["number"]).columns
        if c not in ORDINAL_COLUMNS and c != jenis_col
    ]

    # Remove any column that ended up in multiple buckets (ordinal wins over numeric)
    numeric_features = [c for c in numeric_features if c not in ordinal_features]
    nominal_features = [
        c for c in nominal_features
        if c not in ordinal_features and c not in numeric_features
    ]

    return numeric_features, ordinal_features, nominal_features, has_jenis


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features, ordinal_features, nominal_features, has_jenis = get_feature_groups(X)

    transformers = []

    if numeric_features:
        transformers.append((
            "numeric",
            StandardScaler(),
            numeric_features,
        ))

    if ordinal_features:
        ordinal_categories = [ORDINAL_COLUMNS[col] for col in ordinal_features]
        transformers.append((
            "ordinal",
            OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            ordinal_features,
        ))

    if nominal_features:
        transformers.append((
            "nominal",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            nominal_features,
        ))

    if has_jenis:
        transformers.append((
            "jenis_multihot",
            JenisOrganisasiMultiHotEncoder(),
            ["Jenis Organisasi"],
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def create_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = split_features_target(df)
    X, y, _ = filter_training_data(X, y)
    stratify = y if y.nunique() > 1 else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def get_input_options(df: pd.DataFrame) -> dict[str, list]:
    options: dict[str, list] = {}
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            continue
        if column == "Jenis Organisasi":
            continue  # handled separately as multiselect
        values = df[column].dropna().unique().tolist()
        if pd.api.types.is_numeric_dtype(df[column]):
            options[column] = sorted(values)
        else:
            options[column] = sorted(values, key=lambda value: str(value))
    return options


def build_candidate_dataframe(candidate: dict) -> pd.DataFrame:
    return pd.DataFrame([{column: candidate.get(column) for column in FEATURE_COLUMNS}])


def dataset_summary(df: pd.DataFrame) -> dict:
    X, y = split_features_target(df)
    _, _, excluded = filter_training_data(X, y)
    X_filt, y_filt, _ = filter_training_data(X, y)
    return {
        "original_rows": len(df),
        "excluded_unknown": excluded,
        "training_rows": len(X_filt),
        "target_distribution": y_filt.value_counts().to_dict(),
        "features": FEATURE_COLUMNS,
        "excluded_leakage": LEAKAGE_COLUMNS,
    }
