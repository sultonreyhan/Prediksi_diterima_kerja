"""Plotly visualization helpers for the Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from src.preprocessing import POSITIVE_LABEL, TARGET_COLUMN, build_target


COLOR_SEQUENCE = ["#2563eb", "#14b8a6", "#f59e0b", "#ef4444", "#8b5cf6", "#64748b"]


def label_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df[TARGET_COLUMN].value_counts().reset_index()
    counts.columns = [TARGET_COLUMN, "Jumlah"]
    fig = px.pie(
        counts,
        names=TARGET_COLUMN,
        values="Jumlah",
        hole=0.45,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), title="Distribusi Status Pekerjaan")
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
        title="Sebaran Tahun Lulus Berdasarkan Status",
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


def confusion_matrix_chart(matrix: np.ndarray) -> go.Figure:
    labels = ["Belum bekerja/studi", POSITIVE_LABEL]
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


def roc_curve_chart(roc_data: dict | None) -> go.Figure:
    fig = go.Figure()
    if roc_data:
        fig.add_trace(
            go.Scatter(
                x=roc_data["fpr"],
                y=roc_data["tpr"],
                mode="lines",
                name="ROC Curve",
                line=dict(color="#2563eb", width=3),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Baseline",
            line=dict(color="#94a3b8", dash="dash"),
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def probability_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 45], "color": "#fee2e2"},
                    {"range": [45, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "#0f172a", "width": 4}, "value": 50},
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=15, r=15, t=20, b=10))
    return fig


def feature_importance_chart(artifact: dict) -> go.Figure | None:
    pipeline = artifact["pipeline"]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return None

    names = preprocessor.get_feature_names_out()
    importance = (
        pd.DataFrame({"Feature": names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .head(12)
    )
    fig = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Top Feature Importance",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=460)
    return fig


def employment_rate_by_column(df: pd.DataFrame, column: str) -> go.Figure:
    temp = df.copy()
    temp["Employability Rate"] = build_target(temp[TARGET_COLUMN])
    grouped = temp.groupby(column, as_index=False)["Employability Rate"].mean()
    grouped["Employability Rate"] = grouped["Employability Rate"] * 100
    fig = px.bar(
        grouped,
        x=column,
        y="Employability Rate",
        color="Employability Rate",
        color_continuous_scale="Teal",
        title=f"Employability Rate by {column}",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), xaxis_tickangle=-25)
    return fig
