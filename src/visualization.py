from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from src.preprocessing import TARGET_COLUMN


COLOR_SEQUENCE = ["#2563eb", "#14b8a6", "#f59e0b", "#ef4444", "#8b5cf6", "#64748b"]


def target_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df[TARGET_COLUMN].value_counts().reset_index()
    counts.columns = [TARGET_COLUMN, "Jumlah"]
    fig = px.pie(
        counts,
        names=TARGET_COLUMN,
        values="Jumlah",
        hole=0.45,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), title="Distribusi Waktu Tunggu Kerja")
    return fig


def categorical_bar_chart(df: pd.DataFrame, column: str) -> go.Figure:
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "Jumlah"]
    fig = px.bar(
        counts,
        x=column,
        y="Jumlah",
        color="Jumlah",
        color_continuous_scale="Blues",
        title=f"Distribusi {column}",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), xaxis_tickangle=-25)
    return fig


def year_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="Tahun Lulus",
        color=TARGET_COLUMN,
        barmode="group",
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Sebaran Tahun Lulus Berdasarkan Waktu Tunggu",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    encoded = pd.DataFrame(index=df.index)
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            encoded[column] = df[column]
        else:
            encoded[column] = pd.factorize(df[column].astype(str))[0]

    corr = encoded.corr(numeric_only=True).round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap (Encoded Features)",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=720)
    return fig


def confusion_matrix_chart(matrix: np.ndarray, labels: list[str]) -> go.Figure:
    fig = ff.create_annotated_heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


_CLASS_ORDER = ["< 1 bulan", "1 - 3 bulan", "3 - 6 bulan", "> 6 bulan"]


def probability_bar_chart(probabilities: dict[str, float]) -> go.Figure:
    # Sort by canonical waiting-time order
    all_keys = list(probabilities.keys())
    ordered_keys = [c for c in _CLASS_ORDER if c in all_keys]
    ordered_keys += [c for c in all_keys if c not in ordered_keys]
    classes = ordered_keys
    probs = [probabilities[c] * 100 for c in classes]
    colors = ["#2563eb", "#14b8a6", "#f59e0b", "#ef4444"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=classes,
        y=probs,
        marker_color=colors[:len(classes)],
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Probabilitas Prediksi per Kelas",
        xaxis_title="Kelas Waktu Tunggu",
        yaxis_title="Probabilitas (%)",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=10, r=10, t=45, b=10),
        height=320,
    )
    return fig


def feature_importance_chart(artifact: dict) -> go.Figure | None:
    pipeline = artifact["pipeline"]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(model, "coef_"):
        return None

    coef = model.coef_
    # Use canonical class order from artifact if available
    class_order = artifact.get("target_classes", model.classes_.tolist())
    pipeline_classes = model.classes_.tolist()

    raw_names = preprocessor.get_feature_names_out()
    # Strip sklearn ColumnTransformer prefix (e.g. "nominal__Pernah Magang_Ya" -> "Pernah Magang_Ya")
    def _clean_name(n: str) -> str:
        parts = n.split("__", 1)
        return parts[1] if len(parts) == 2 else n

    feature_names = [_clean_name(n) for n in raw_names]

    fig = go.Figure()
    for cls in class_order:
        if cls not in pipeline_classes:
            continue
        i = pipeline_classes.index(cls)
        importance = np.abs(coef[i])
        top_indices = np.argsort(importance)[-10:]
        fig.add_trace(go.Bar(
            name=str(cls),
            y=[feature_names[j] for j in top_indices],
            x=coef[i][top_indices],  # signed coefficients, not abs
            orientation="h",
        ))

    fig.update_layout(
        title="Koefisien Logistic Regression per Kelas (top 10 per kelas)",
        xaxis_title="Nilai Koefisien (positif = lebih cepat kerja, negatif = lebih lambat)",
        barmode="group",
        height=480,
        margin=dict(l=10, r=10, t=55, b=10),
        legend_title="Kelas Waktu Tunggu",
    )
    return fig


def waiting_time_rate_by_column(df: pd.DataFrame, column: str) -> go.Figure:
    temp = df.copy()
    grouped = temp.groupby([column, TARGET_COLUMN], as_index=False).size()
    fig = px.bar(
        grouped,
        x=column,
        y="size",
        color=TARGET_COLUMN,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=f"Distribusi Waktu Tunggu by {column}",
        barmode="stack",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), xaxis_tickangle=-25)
    return fig
