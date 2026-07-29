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
    employment_rate_by_column,
    feature_importance_chart,
    label_distribution_chart,
    probability_gauge,
    roc_curve_chart,
    year_histogram,
)


st.set_page_config(
    page_title="Employability Prediction",
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


@st.cache_resource(show_spinner="Melatih model machine learning...")
def get_model_artifact() -> dict:
    if MODEL_PATH.exists():
        try:
            from src.model import load_model

            artifact = load_model(MODEL_PATH)
            if "primary_model_name" in artifact:
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
        "Employability Prediction Dashboard",
        "Sistem analitik untuk memahami profil lulusan dan memprediksi peluang diterima kerja berdasarkan data akademik, pengalaman, sertifikasi, dan aktivitas organisasi.",
    )

    evaluation = artifact["evaluation"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Accuracy", f"{evaluation['accuracy']:.1%}", f"Primary Model: {artifact.get('primary_model_name', 'Logistic Regression')}")
    with col2:
        metric_card("Jumlah Data", f"{len(df):,}", "baris dataset")
    with col3:
        metric_card("Jumlah Fitur", f"{len(artifact['feature_columns'])}", "fitur kandidat")
    with col4:
        employed_rate = (df[TARGET_COLUMN].eq("Sudah bekerja").mean()) * 100
        metric_card("Employability Rate", f"{employed_rate:.1f}%", "label sudah bekerja")

    st.write("")
    st.subheader("Machine Learning Workflow")
    steps = [
        ("1. Data Understanding", "Membaca dataset tracer study dan memahami distribusi setiap atribut."),
        ("2. Preprocessing", "Membersihkan data, encoding fitur kategorikal, scaling numerik, lalu split train-test."),
        ("3. Modeling", "Membandingkan Logistic Regression, Random Forest, dan Gradient Boosting."),
        ("4. Evaluation", "Mengukur accuracy, precision, recall, F1-score, confusion matrix, dan ROC curve."),
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
        st.plotly_chart(label_distribution_chart(df), use_container_width=True)


def render_dataset_overview(df: pd.DataFrame) -> None:
    render_header(
        "Dataset Overview",
        "Eksplorasi dataset employability: preview data, statistik, missing value, distribusi label, dan relasi antar fitur.",
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
            st.plotly_chart(label_distribution_chart(df), use_container_width=True)

    with tab_visual:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(categorical_bar_chart(df, "Kategori IPK"), use_container_width=True)
            st.plotly_chart(categorical_bar_chart(df, "Rumpun Jurusan"), use_container_width=True)
        with col2:
            st.plotly_chart(employment_rate_by_column(df, "Pernah Magang"), use_container_width=True)
            st.plotly_chart(employment_rate_by_column(df, "Memiliki Sertifikasi"), use_container_width=True)
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)


def render_preprocessing() -> None:
    render_header(
        "Data Preprocessing",
        "Ringkasan pipeline yang digunakan sebelum data masuk ke model machine learning.",
    )

    preprocessing_steps = [
        (
            "1. Cleaning",
            "Dataset dibaca dari Excel, nama kolom dirapikan, dan missing value dicek melalui halaman Dataset Overview.",
            "df = pd.read_csv('data/kuesioner_faktor_akademik.csv')\ndf.columns = df.columns.str.strip()\ndf.isna().sum()",
        ),
        (
            "2. Target Engineering",
            "`Status Pekerjaan` diubah menjadi target biner: `Sudah bekerja` = 1, selain itu = 0.",
            "y = (df['Status Pekerjaan'] == 'Sudah bekerja').astype(int)",
        ),
        (
            "3. Leakage Prevention",
            "Kolom hasil setelah bekerja seperti waktu tunggu, kesesuaian pekerjaan, dan gaji tidak dipakai sebagai fitur prediksi.",
            "leakage_columns = ['Waktu Tunggu Kerja', 'Kesesuaian Pekerjaan', 'Kisaran Gaji']",
        ),
        (
            "4. Encoding dan Scaling",
            "Fitur kategorikal diproses dengan OneHotEncoder, sementara fitur numerik seperti tahun lulus dan tingkat keaktifan diskalakan.",
            "ColumnTransformer([\n    ('numeric', StandardScaler(), numeric_features),\n    ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)\n])",
        ),
        (
            "5. Train Test Split",
            "Data dipisah 80:20 dengan stratifikasi agar komposisi kelas target tetap seimbang di train dan test set.",
            "train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)",
        ),
    ]

    for title, explanation, code in preprocessing_steps:
        with st.expander(title, expanded=title.startswith("1.")):
            st.write(explanation)
            st.code(code, language="python")


def render_prediction(df: pd.DataFrame, artifact: dict) -> None:
    render_header(
        "Prediction",
        "Masukkan profil kandidat untuk memprediksi kemungkinan diterima kerja. Model menghasilkan skor probabilitas employability.",
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
    predict_clicked = st.button("Prediksi Employability", use_container_width=True)

    if predict_clicked:
        result = predict_single(artifact, candidate)
        probability = result["probability"]
        st.write("")
        col_result, col_gauge = st.columns([0.9, 1.1])
        with col_result:
            st.subheader(result["label"])
            st.progress(probability)
            st.metric("Confidence Score", f"{probability:.1%}")
            if probability >= 0.7:
                st.success("Profil kandidat sangat kompetitif berdasarkan pola data historis.")
            elif probability >= 0.5:
                st.warning("Profil cukup potensial, tetapi masih ada ruang untuk peningkatan.")
            else:
                st.error("Profil perlu diperkuat sebelum masuk proses rekrutmen.")
        with col_gauge:
            st.plotly_chart(probability_gauge(probability), use_container_width=True)

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
                file_name="employability_prediction_result.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except ValueError as error:
            st.error(str(error))


def render_model_evaluation(artifact: dict) -> None:
    render_header(
        "Model Evaluation",
        "Evaluasi performa model Logistic Regression pada data pengujian.",
    )

    evaluation = artifact["evaluation"]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Accuracy", f"{evaluation['accuracy']:.1%}", artifact.get("primary_model_name", "Logistic Regression"))
    with col2:
        metric_card("Precision", f"{evaluation['precision']:.1%}", "")
    with col3:
        metric_card("Recall", f"{evaluation['recall']:.1%}", "")
    with col4:
        metric_card("F1-score", f"{evaluation['f1']:.1%}", "")
    with col5:
        roc_auc = evaluation["roc_auc"]
        metric_card("ROC AUC", "N/A" if roc_auc is None else f"{roc_auc:.1%}", "")

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(confusion_matrix_chart(evaluation["confusion_matrix"]), use_container_width=True)
    with col_right:
        st.plotly_chart(roc_curve_chart(evaluation["roc_curve"]), use_container_width=True)

    importance_fig = feature_importance_chart(artifact)
    if importance_fig is not None:
        st.plotly_chart(importance_fig, use_container_width=True)


def render_about() -> None:
    render_header(
        "About Project",
        "Portfolio project data science yang menggabungkan machine learning, dashboard interaktif, dan storytelling HR analytics.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Tujuan")
        st.write(
            "Aplikasi ini membantu mengeksplorasi faktor yang berkaitan dengan employability lulusan, "
            "menguji model klasifikasi, dan menyediakan simulasi prediksi kandidat secara realtime."
        )
        st.subheader("Teknologi")
        st.write("Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, OpenPyXL, dan Joblib.")

    with col2:
        st.subheader("Model Machine Learning")
        st.write(
            "Pipeline model menggunakan preprocessing otomatis untuk fitur numerik dan kategorikal, "
            "lalu membandingkan beberapa algoritma baseline. Model terbaik disimpan sebagai artifact "
            "di folder `saved_models`."
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
            4. Deploy. Model akan dilatih otomatis saat aplikasi pertama kali dijalankan jika `saved_models/model.pkl` belum tersedia.
            """
        )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    df = get_dataset_cached(get_data_mtime())
    artifact = get_model_artifact()

    st.sidebar.title("HR Analytics")
    st.sidebar.caption("Employability Prediction System")
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
    st.sidebar.metric("Primary Model", artifact.get("primary_model_name", "Logistic Regression"))
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
