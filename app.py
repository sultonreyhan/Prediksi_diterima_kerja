from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import MODEL_PATH, train_and_save_model, train_models
from src.predict import predict_batch, predict_single
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, get_input_options, load_dataset
from src.visualization import (
    categorical_bar_chart,
    confusion_matrix_chart,
    correlation_heatmap,
    feature_importance_chart,
    probability_bar_chart,
    target_distribution_chart,
    waiting_time_rate_by_column,
    year_histogram,
)


st.set_page_config(
    page_title="Prediksi Kecepatan Diterima Kerja",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = Path("data/Kuesioner Faktor Akademik dan Aktivitas Organisasi terhadap Kecepatan Diterima Kerja Lulusan Mahasiswa  (Responses) - Form responses 1.csv")


CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1220px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #30363d;
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.25rem;
    }

    .app-subtitle {
        color: #8b949e;
        font-size: 1rem;
        line-height: 1.55;
        max-width: 860px;
    }

    .metric-card {
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        background: #161b22;
        min-height: 112px;
    }

    .metric-label {
        color: #8b949e;
        font-size: 0.86rem;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f0f6fc;
    }

    .workflow-step {
        border-left: 3px solid #ff4b55;
        padding: 0.75rem 1rem;
        background: rgba(255, 75, 85, 0.06);
        border-radius: 6px;
        min-height: 92px;
    }

    .small-muted {
        color: #8b949e;
        font-size: 0.9rem;
    }

    .accent-text {
        color: #ff4b55;
        font-size: 0.86rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: #161b22;
    }

    div[data-testid="stMetric"] label {
        color: #8b949e !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f0f6fc !important;
    }

    .pred-class {
        font-size: 2rem;
        font-weight: 800;
        color: #2563eb;
        text-align: center;
        padding: 1rem;
        border: 2px solid #2563eb;
        border-radius: 12px;
        background: rgba(37, 99, 235, 0.08);
    }
</style>
"""


def metric_card(label: str, value: str, helper: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="small-muted">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_data_mtime() -> float:
    return os.path.getmtime(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_dataset_cached(mtime: float) -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_resource(show_spinner="Melatih model multiclass Logistic Regression...")
def get_model_artifact() -> dict:
    if MODEL_PATH.exists():
        try:
            from src.model import load_model

            artifact = load_model(MODEL_PATH)
            if artifact.get("model_version") == 2:
                return artifact
        except Exception:
            pass

    df = load_dataset(DATA_PATH)
    artifact = train_and_save_model(DATA_PATH, MODEL_PATH)
    return artifact


def render_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def render_home(df: pd.DataFrame, artifact: dict) -> None:
    render_header(
        "Prediksi Kecepatan Diterima Kerja",
        "Sistem analitik untuk memprediksi waktu tunggu mendapatkan pekerjaan pertama setelah lulus berdasarkan data akademik, pengalaman, sertifikasi, dan aktivitas organisasi.",
    )

    evaluation = artifact["evaluation"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Accuracy", f"{evaluation['accuracy']:.1%}", artifact.get("primary_model_name", "Multiclass Logistic Regression"))
    with col2:
        metric_card("Macro F1", f"{evaluation['macro_f1']:.3f}", "rata-rata antar kelas")
    with col3:
        metric_card("Balanced Accuracy", f"{evaluation['balanced_accuracy']:.3f}", "rata-rata recall per kelas")
    with col4:
        metric_card("Jumlah Data", f"{len(df):,}", "baris dataset")

    st.write("")
    st.subheader("Machine Learning Workflow")
    steps = [
        ("1. Data Understanding", "Membaca dataset tracer study dan memahami distribusi setiap atribut."),
        ("2. Preprocessing", "Membersihkan data, encoding ordinal/nominal, scaling numerik, split train-test."),
        ("3. Modeling", "Multiclass Logistic Regression dengan 4 kelas waktu tunggu."),
        ("4. Evaluation", "Accuracy, Balanced Accuracy, Macro F1, Confusion Matrix, Cross-Validation."),
    ]
    step_cols = st.columns(4)
    for column, (heading, body) in zip(step_cols, steps):
        with column:
            st.markdown(
                f'<div class="workflow-step"><strong>{heading}</strong><br><span class="small-muted">{body}</span></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns([1.1, 0.9])
    with left:
        st.plotly_chart(year_histogram(df), use_container_width=True)
    with right:
        st.plotly_chart(target_distribution_chart(df), use_container_width=True)


def render_dataset_overview(df: pd.DataFrame) -> None:
    render_header(
        "Dataset Overview",
        "Eksplorasi dataset tracer study: preview data, statistik, missing value, distribusi waktu tunggu, dan relasi antar fitur.",
    )

    tab_preview, tab_quality, tab_visual = st.tabs(["Preview", "Data Quality", "Visualisasi"])
    with tab_preview:
        st.dataframe(df, use_container_width=True, height=360)
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with tab_quality:
        missing = df.isna().sum().reset_index()
        missing.columns = ["Kolom", "Missing Values"]
        col1, col2 = st.columns([0.8, 1.2])
        with col1:
            st.dataframe(missing, use_container_width=True, hide_index=True)
        with col2:
            st.plotly_chart(target_distribution_chart(df), use_container_width=True)

    with tab_visual:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(categorical_bar_chart(df, "Kategori IPK"), use_container_width=True)
            st.plotly_chart(categorical_bar_chart(df, "Rumpun Jurusan"), use_container_width=True)
        with col2:
            st.plotly_chart(waiting_time_rate_by_column(df, "Pernah Magang"), use_container_width=True)
            st.plotly_chart(waiting_time_rate_by_column(df, "Memiliki Sertifikasi"), use_container_width=True)
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)


def render_preprocessing() -> None:
    render_header(
        "Data Preprocessing",
        "Ringkasan pipeline yang digunakan sebelum data masuk ke model multiclass Logistic Regression.",
    )

    preprocessing_steps = [
        (
            "1. Cleaning",
            "Dataset dibaca dari CSV, nama kolom dirapikan. Missing value dicek. Kategori tidak konsisten dinormalisasi (contoh: '2,5' dan '3 tahun' pada Lama Masa Studi diseragamkan ke '< 4 tahun').",
            "df = pd.read_csv('data/kuesioner.csv')\ndf.columns = df.columns.str.strip()\n# Normalisasi kategori\nmapping = {'2,5': '< 4 tahun', '3 tahun': '< 4 tahun'}\ndf['Lama Masa Studi'] = df['Lama Masa Studi'].replace(mapping)",
        ),
        (
            "2. Target Engineering",
            "Target adalah `Waktu Tunggu Kerja` dengan 4 kelas: `< 1 bulan`, `1 - 3 bulan`, `3 - 6 bulan`, `> 6 bulan`. Responden dengan status `Belum bekerja` dieksklusi karena waktu tunggunya belum terobservasi.",
            "y = df['Waktu Tunggu Kerja']\nmask = y != 'Belum bekerja'\nX = X[mask]\ny = y[mask]",
        ),
        (
            "3. Leakage Prevention",
            "Kolom hasil setelah bekerja (`Status Pekerjaan`, `Kesesuaian Pekerjaan`, `Kisaran Gaji`) tidak dipakai sebagai fitur untuk menghindari data leakage.",
            "leakage_columns = ['Status Pekerjaan', 'Kesesuaian Pekerjaan', 'Kisaran Gaji']",
        ),
        (
            "4. Encoding",
            "Fitur ordinal (`IPK`, `Lama Studi`, `Tingkat Keaktifan`, `Jumlah Organisasi`) menggunakan OrdinalEncoder. Fitur nominal (`Rumpun Jurusan`, `Jenis Organisasi`, dll) menggunakan OneHotEncoder.",
            "ColumnTransformer([\n    ('ordinal', OrdinalEncoder(categories=...), ordinal_features),\n    ('nominal', OneHotEncoder(handle_unknown='ignore'), nominal_features),\n    ('numeric', StandardScaler(), numeric_features)\n])",
        ),
        (
            "5. Train Test Split",
            "Data dipisah 80:20 dengan stratifikasi agar komposisi kelas target tetap seimbang.",
            "train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)",
        ),
    ]

    for title, explanation, code in preprocessing_steps:
        with st.expander(title, expanded=title.startswith("1.")):
            st.write(explanation)
            st.code(code, language="python")


def render_prediction(df: pd.DataFrame, artifact: dict) -> None:
    render_header(
        "Prediksi Waktu Tunggu Kerja",
        "Masukkan profil lulusan untuk memprediksi waktu tunggu mendapatkan pekerjaan pertama setelah lulus.",
    )

    options = get_input_options(df)
    candidate: dict[str, object] = {}

    org_columns = {
        "Jumlah Organisasi",
        "Aktif Organisasi",
        "Jenis Organisasi",
        "Jabatan Organisasi",
        "Tingkat Keaktifan Organisasi (1-5)",
    }
    non_org_columns = [c for c in FEATURE_COLUMNS if c not in org_columns]

    col1, col2, col3 = st.columns(3)
    form_cols = [col1, col2, col3]
    idx = 0

    for column in non_org_columns:
        if column not in options:
            continue
        with form_cols[idx % 3]:
            if pd.api.types.is_numeric_dtype(df[column]):
                values = options[column]
                candidate[column] = st.slider(
                    column,
                    min_value=int(min(values)),
                    max_value=int(max(values)),
                    value=int(round(df[column].median())),
                    key=f"slider_{column}",
                )
            else:
                candidate[column] = st.selectbox(
                    column, options[column], key=f"sel_{column}"
                )
        idx += 1

    with form_cols[idx % 3]:
        jumlah = st.selectbox(
            "Jumlah Organisasi",
            options["Jumlah Organisasi"],
            key="jumlah_organisasi",
        )
    idx += 1
    candidate["Jumlah Organisasi"] = jumlah

    if jumlah == "0":
        st.info("Detail organisasi tidak diperlukan karena jumlah organisasi adalah 0.")
        candidate["Jenis Organisasi"] = "Tidak mengikuti organisasi"
        candidate["Jabatan Organisasi"] = "Tidak pernah mengikuti organisasi"
        candidate["Aktif Organisasi"] = "Tidak"
        candidate["Tingkat Keaktifan Organisasi (1-5)"] = 1
    else:
        with form_cols[idx % 3]:
            candidate["Jenis Organisasi"] = st.selectbox(
                "Jenis Organisasi",
                options["Jenis Organisasi"],
                key="jenis_organisasi",
            )
        idx += 1
        with form_cols[idx % 3]:
            candidate["Jabatan Organisasi"] = st.selectbox(
                "Jabatan Organisasi",
                options["Jabatan Organisasi"],
                key="jabatan_organisasi",
            )
        idx += 1
        with form_cols[idx % 3]:
            aktif = st.selectbox(
                "Aktif Organisasi",
                options["Aktif Organisasi"],
                key="aktif_organisasi",
            )
        idx += 1
        candidate["Aktif Organisasi"] = aktif

        if aktif == "Ya":
            with form_cols[idx % 3]:
                candidate["Tingkat Keaktifan Organisasi (1-5)"] = st.slider(
                    "Tingkat Keaktifan Organisasi (1-5)",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key="tingkat_keaktifan",
                )
            idx += 1
        else:
            st.caption("Tingkat keaktifan otomatis diisi 1 karena tidak aktif berorganisasi.")
            candidate["Tingkat Keaktifan Organisasi (1-5)"] = 1

    st.write("")
    predict_clicked = st.button("Prediksi Waktu Tunggu", use_container_width=True)

    if predict_clicked:
        result = predict_single(artifact, candidate)
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        st.write("")
        col_result, col_probs = st.columns([1, 1])

        with col_result:
            st.markdown("<div class='pred-class'>" + predicted_class + "</div>", unsafe_allow_html=True)
            st.caption("Prediksi Waktu Mendapatkan Pekerjaan")
            st.metric("Confidence Prediksi", f"{confidence:.1%}")

            interpretation_map = {
                "< 1 bulan": "Model memperkirakan kandidat cenderung memperoleh pekerjaan pertama dalam waktu kurang dari 1 bulan setelah lulus.",
                "1 - 3 bulan": "Model memperkirakan kandidat cenderung memperoleh pekerjaan pertama dalam rentang 1–3 bulan setelah lulus.",
                "3 - 6 bulan": "Model memperkirakan kandidat cenderung memperoleh pekerjaan pertama dalam rentang 3–6 bulan setelah lulus.",
                "> 6 bulan": "Model memperkirakan waktu mendapatkan pekerjaan pertama cenderung lebih dari 6 bulan setelah lulus.",
            }
            interpretation = interpretation_map.get(predicted_class, "")
            if interpretation:
                st.info(interpretation)

        with col_probs:
            st.plotly_chart(probability_bar_chart(probabilities), use_container_width=True)

    st.divider()
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV kandidat dengan kolom yang sama seperti fitur model", type=["csv"])
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        try:
            prediction_df = predict_batch(artifact, uploaded_df)
            st.dataframe(prediction_df, use_container_width=True)

            output = io.StringIO()
            prediction_df.to_csv(output, index=False)
            st.download_button(
                "Download Prediction Result",
                data=output.getvalue(),
                file_name="waiting_time_prediction_result.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except ValueError as error:
            st.error(str(error))


def render_model_evaluation(artifact: dict) -> None:
    render_header(
        "Model Evaluation",
        "Evaluasi performa multiclass Logistic Regression pada data pengujian.",
    )

    evaluation = artifact["evaluation"]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Accuracy", f"{evaluation['accuracy']:.1%}", artifact.get("primary_model_name", "Multiclass Logistic Regression"))
    with col2:
        metric_card("Balanced Accuracy", f"{evaluation['balanced_accuracy']:.3f}", "")
    with col3:
        metric_card("Macro F1", f"{evaluation['macro_f1']:.3f}", "")
    with col4:
        metric_card("Weighted F1", f"{evaluation['weighted_f1']:.3f}", "")
    with col5:
        cv_f1 = artifact.get("cv_macro_f1_mean")
        metric_card("CV Macro F1", f"{cv_f1:.3f}" if cv_f1 else "N/A", "5-fold stratified CV")

    col_left, col_right = st.columns(2)
    with col_left:
        matrix = evaluation["confusion_matrix"]
        labels = evaluation.get("classes", artifact.get("target_classes", []))
        st.plotly_chart(confusion_matrix_chart(matrix, labels), use_container_width=True)

    with col_right:
        st.subheader("Classification Report")
        report = evaluation.get("classification_report", {})
        report_rows = []
        for cls_name, metrics in report.items():
            if cls_name in ("accuracy", "macro avg", "weighted avg"):
                continue
            if isinstance(metrics, dict):
                report_rows.append({
                    "Kelas": cls_name,
                    "Precision": f"{metrics['precision']:.3f}",
                    "Recall": f"{metrics['recall']:.3f}",
                    "F1": f"{metrics['f1-score']:.3f}",
                    "Support": int(metrics['support']),
                })
        if report_rows:
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    importance_fig = feature_importance_chart(artifact)
    if importance_fig is not None:
        st.plotly_chart(importance_fig, use_container_width=True)

    with st.expander("Cross-Validation Details"):
        cv_f1 = artifact.get("cv_macro_f1_mean")
        cv_f1_std = artifact.get("cv_macro_f1_std")
        cv_acc = artifact.get("cv_accuracy_mean")
        cv_acc_std = artifact.get("cv_accuracy_std")
        if cv_f1:
            st.write(f"5-fold Stratified Cross-Validation:")
            st.write(f"- Macro F1: {cv_f1:.3f} (+/- {cv_f1_std:.3f})")
            st.write(f"- Accuracy: {cv_acc:.3f} (+/- {cv_acc_std:.3f})")


def render_about() -> None:
    render_header(
        "About Project",
        "Portfolio project data science: Multiclass Logistic Regression untuk prediksi waktu tunggu kerja lulusan.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Tujuan")
        st.write(
            "Aplikasi ini memprediksi **waktu tunggu mendapatkan pekerjaan pertama** setelah lulus "
            "berdasarkan faktor akademik dan aktivitas organisasi. Model menggunakan **Multiclass Logistic Regression** "
            "dengan 4 kelas waktu tunggu."
        )
        st.subheader("Teknologi")
        st.write("Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, dan Joblib.")

    with col2:
        st.subheader("Model Machine Learning")
        st.write(
            "Pipeline model menggunakan preprocessing otomatis: OrdinalEncoder untuk fitur ordinal, "
            "OneHotEncoder untuk fitur nominal, dan StandardScaler untuk fitur numerik. "
            "Model utama adalah Multiclass Logistic Regression dengan evaluasi 5-fold Stratified Cross-Validation."
        )
        st.subheader("Streamlit")
        st.write(
            "Streamlit dipakai untuk membuat dashboard cepat, interaktif, dan mudah dideploy ke Streamlit Cloud."
        )

    with st.expander("Catatan Deployment Streamlit Cloud"):
        st.markdown(
            """
            1. Push project ke GitHub.
            2. Pastikan `requirements.txt`, `app.py`, folder `src`, dan folder `data` ikut ter-upload.
            3. Buka Streamlit Cloud, pilih repository, lalu set main file ke `app.py`.
            4. Deploy. Model akan dilatih otomatis saat aplikasi pertama kali dijalankan.
            """
        )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    df = get_dataset_cached(get_data_mtime())
    artifact = get_model_artifact()

    st.sidebar.title("Tracer Study Analytics")
    st.sidebar.caption("Prediksi Waktu Tunggu Kerja")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Dataset Overview",
            "Data Preprocessing",
            "Prediction",
            "Model Evaluation",
            "About",
        ],
    )
    st.sidebar.divider()
    st.sidebar.metric("Model", artifact.get("primary_model_name", "Multiclass Logistic Regression"))
    st.sidebar.metric("Rows", f"{len(df):,}")

    if page == "Home":
        render_home(df, artifact)
    elif page == "Dataset Overview":
        render_dataset_overview(df)
    elif page == "Data Preprocessing":
        render_preprocessing()
    elif page == "Prediction":
        render_prediction(df, artifact)
    elif page == "Model Evaluation":
        render_model_evaluation(artifact)
    else:
        render_about()


if __name__ == "__main__":
    main()
