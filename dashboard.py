# dashboard.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="SENN Dashboard",
    page_icon="📊",
    layout="wide",
)

PROJECT_ROOT = Path(".")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PARETO_DIR = OUTPUTS_DIR / "pareto_fronts"  # stays global

def list_run_dirs(outputs_dir: Path) -> list[Path]:
    if not outputs_dir.exists():
        return []
    runs = [p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    # newest first (by folder name timestamp)
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs

run_dirs = list_run_dirs(OUTPUTS_DIR)

if not run_dirs:
    st.error(
        "❌ No run folders found.\n\nExpected folders like:\n"
        "- outputs/run_YYYYMMDD_HHMMSS/\n\n"
        "Run your evolution pipeline once to generate outputs."
    )
    st.stop()

# Run selector (top of app)
run_names = [p.name for p in run_dirs]
selected_run_name = st.selectbox("Select run", options=run_names, index=0)
RUN_DIR = OUTPUTS_DIR / selected_run_name

# Prefer metrics.csv at run root; fallback to population/metrics.csv
METRICS_PATH = RUN_DIR / "metrics.csv"
if not METRICS_PATH.exists():
    alt = RUN_DIR / "population" / "metrics.csv"
    if alt.exists():
        METRICS_PATH = alt

ACC_VS_GEN_PATH = RUN_DIR / "acc_vs_gen.png"
BEST_ARCH_PATH = RUN_DIR / "best_architecture.json"


REQUIRED_COLS = ["arch_id", "generation", "val_accuracy", "param_count", "flops", "latency"]


# -----------------------------
# Helpers
# -----------------------------
def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_metrics(csv_path: Path) -> Tuple[Optional[pd.DataFrame], list[str]]:
    warnings: list[str] = []
    if not csv_path.exists():
        return None, warnings

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, [f"Failed to read metrics.csv: {e}"]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, [f"metrics.csv is missing required columns: {missing}"]

    df = df.copy()
    df["arch_id"] = df["arch_id"].astype(str)

    df["generation"] = _safe_numeric(df["generation"])
    df["val_accuracy"] = _safe_numeric(df["val_accuracy"])
    df["param_count"] = _safe_numeric(df["param_count"])
    df["flops"] = _safe_numeric(df["flops"])
    df["latency"] = _safe_numeric(df["latency"])

    if df["latency"].isna().any():
        n = int(df["latency"].isna().sum())
        df["latency"] = df["latency"].fillna(0.0)
        warnings.append(f"Latency had {n} missing/NaN values. Filled with 0.0.")

    essential = ["generation", "val_accuracy", "param_count", "flops"]
    before = len(df)
    df = df.dropna(subset=essential)
    dropped = before - len(df)
    if dropped > 0:
        warnings.append(f"Dropped {dropped} row(s) due to missing essential values: {essential}")

    df["generation"] = df["generation"].astype(int)

    # UI generation (1-based display)
    df["generation_display"] = df["generation"] + 1

    return df, warnings


def image_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def metric_card_row(
    best_acc: float,
    min_params: float,
    min_flops: float,
    min_latency: float,
    total_arch: int,
    total_gen_display: int,
):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Best val_accuracy", f"{best_acc:.4f}")
    c2.metric("Lowest param_count", f"{int(min_params):,}" if np.isfinite(min_params) else "—")
    c3.metric("Lowest FLOPs", f"{int(min_flops):,}" if np.isfinite(min_flops) else "—")
    c4.metric("Lowest Latency", f"{min_latency:.4f}" if np.isfinite(min_latency) else "—")
    c5.metric("Architectures evaluated", f"{total_arch:,}")
    c6.metric("Total generations", f"{total_gen_display:,}")


def topk_filter(df: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
    if k is None:
        return df
    return df.sort_values("val_accuracy", ascending=False).head(k)


def make_line_best_acc(df: pd.DataFrame) -> go.Figure:
    g = (
        df.groupby("generation_display", as_index=False)["val_accuracy"]
        .max()
        .sort_values("generation_display")
    )
    fig = px.line(
        g,
        x="generation_display",
        y="val_accuracy",
        markers=True,
        title="Best val_accuracy per generation",
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_xaxes(title="Generation")
    return fig


def make_line_min_params(df: pd.DataFrame) -> go.Figure:
    g = (
        df.groupby("generation_display", as_index=False)["param_count"]
        .min()
        .sort_values("generation_display")
    )
    fig = px.line(
        g,
        x="generation_display",
        y="param_count",
        markers=True,
        title="Min param_count per generation",
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_xaxes(title="Generation")
    return fig


def make_scatter_focus(df: pd.DataFrame, y_metric: str, title: str) -> go.Figure:
    hover_cols = ["arch_id", "generation_display", "val_accuracy", "param_count", "flops", "latency"]
    fig = px.scatter(
        df,
        x="val_accuracy",
        y=y_metric,
        color="generation_display",
        hover_data={c: True for c in hover_cols if c in df.columns},
        title=title,
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
    fig.update_xaxes(title="val_accuracy")
    fig.update_yaxes(title=y_metric)
    return fig


def pareto_front_2d(df_in: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """
    Pareto front for 2 objectives:
      - maximize val_accuracy
      - minimize x_col
    """
    if len(df_in) == 0:
        return df_in

    d = df_in.copy()
    d["val_accuracy"] = pd.to_numeric(d["val_accuracy"], errors="coerce")
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d = d.dropna(subset=["val_accuracy", x_col])

    d = d.sort_values(["val_accuracy", x_col], ascending=[False, True])

    best_x = np.inf
    keep_mask = []
    for _, row in d.iterrows():
        x = float(row[x_col])
        if x < best_x:
            keep_mask.append(True)
            best_x = x
        else:
            keep_mask.append(False)

    return d.loc[d.index[keep_mask]].copy()


# -----------------------------
# Header
# -----------------------------
st.title("📊 SENN Dashboard")
st.caption("Reads results from `outputs/metrics.csv` and visualizes evolution metrics + Pareto fronts.")

# -----------------------------
# Load Data
# -----------------------------
df, warn_list = load_metrics(METRICS_PATH)

if df is None:
    st.error(
        "❌ Could not load `outputs/metrics.csv`.\n\n"
        "Make sure you are running this from the **project root** and that the file exists at:\n"
        f"- `{METRICS_PATH.as_posix()}`\n\n"
        "Expected columns:\n"
        f"- {', '.join(REQUIRED_COLS)}"
    )
    st.stop()

for w in warn_list:
    st.warning(w)

max_gen_display = int(df["generation_display"].max()) if len(df) else 1
all_gens_display = list(range(1, max_gen_display + 1))

# -----------------------------
# Session defaults (drive whole dashboard)
# -----------------------------
if "focus_gen_range" not in st.session_state:
    st.session_state.focus_gen_range = (1, max_gen_display)
if "focus_topk_choice" not in st.session_state:
    st.session_state.focus_topk_choice = "50"
if "focus_metric" not in st.session_state:
    st.session_state.focus_metric = "param_count"

# Use Focus settings globally
gen_range = st.session_state.focus_gen_range
selected_gens = list(range(gen_range[0], gen_range[1] + 1))

topk_choice = st.session_state.focus_topk_choice
topk = None if topk_choice == "All" else int(topk_choice)

focus_metric = st.session_state.focus_metric

# Filter once (KPIs, Trends, Table use this)
df_f = df[df["generation_display"].isin(selected_gens)].copy()
df_plot = topk_filter(df_f, topk)

# -----------------------------
# KPI Cards
# -----------------------------
st.subheader("KPIs")
if len(df_f) == 0:
    st.info("No data after filtering. Adjust the generation range in Focus View.")
else:
    best_acc = float(df_f["val_accuracy"].max())
    min_params = float(df_f["param_count"].min())
    min_flops = float(df_f["flops"].min())
    min_latency = float(df_f["latency"].min()) if "latency" in df_f.columns else float("nan")
    total_arch = int(df_f["arch_id"].nunique())
    total_gen_display = int(df_f["generation_display"].max())

    metric_card_row(best_acc, min_params, min_flops, min_latency, total_arch, total_gen_display)

st.markdown("---")

# -----------------------------
# Trends
# -----------------------------
st.subheader("Trends")
if len(df_f) == 0:
    st.info("No trend charts to show (empty filtered data).")
else:
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(make_line_best_acc(df_f), use_container_width=True)
    with colB:
        st.plotly_chart(make_line_min_params(df_f), use_container_width=True)

st.markdown("---")

# -----------------------------
# Focus View (controls ON TOP, no Top-K)
# -----------------------------
st.subheader("Focus View")

# Defaults (only if missing)
if "focus_gen_range" not in st.session_state:
    st.session_state["focus_gen_range"] = (1, max_gen_display)
if "focus_metric" not in st.session_state:
    st.session_state["focus_metric"] = "param_count"

fc1, fc2 = st.columns([2, 1])

with fc1:
    gen_range = st.slider(
        "Generation range",
        min_value=1,
        max_value=max_gen_display,
        step=1,
        key="focus_gen_range",
    )

with fc2:
    focus_metric = st.selectbox(
        "Y metric",
        options=["param_count", "flops", "latency"],
        key="focus_metric",
    )

selected_gens = list(range(gen_range[0], gen_range[1] + 1))
df_f_focus = df[df["generation_display"].isin(selected_gens)].copy()

if len(df_f_focus) == 0:
    st.info("No data to show. Try widening the generation range.")
else:
    st.plotly_chart(
        make_scatter_focus(
            df_f_focus,
            y_metric=focus_metric,
            title=f"val_accuracy vs {focus_metric}",
        ),
        use_container_width=True,
    )

st.markdown("---")

# -----------------------------
# Interactive Pareto Front Viewer
# -----------------------------
st.subheader("Pareto Front Viewer")

p1, p2, p3 = st.columns([1, 1, 1])

with p1:
    gen_pick_display = st.selectbox("Generation", options=all_gens_display, index=0)
with p2:
    x_metric = st.selectbox("X metric (minimize)", options=["param_count", "flops", "latency"], index=0)
with p3:
    show_png = st.checkbox("Also show saved PNG", value=False)

df_g = df[df["generation_display"] == gen_pick_display].copy()

if len(df_g) == 0:
    st.warning("No data for this generation in metrics.csv.")
else:
    # Compute Pareto front (maximize val_accuracy, minimize x_metric)
    front = pareto_front_2d(df_g, x_metric)

    # Mark front points
    df_g["_is_front"] = False
    df_g.loc[front.index, "_is_front"] = True

    hover_cols = ["arch_id", "generation_display", "val_accuracy", "param_count", "flops", "latency"]
    fig = px.scatter(
        df_g,
        x=x_metric,
        y="val_accuracy",
        color="_is_front",
        hover_data={c: True for c in hover_cols if c in df_g.columns},
        title=f"Interactive Pareto View — Gen {gen_pick_display} (maximize val_accuracy, minimize {x_metric})",
    )

    fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
    fig.update_xaxes(title=x_metric)
    fig.update_yaxes(title="val_accuracy")

    # Optional: connect Pareto front points
    if len(front) >= 2:
        front_sorted = front.sort_values(x_metric, ascending=True)
        fig.add_trace(
            go.Scatter(
                x=front_sorted[x_metric],
                y=front_sorted["val_accuracy"],
                mode="lines+markers",
                name="Pareto front curve",
            )
        )

    st.plotly_chart(fig, use_container_width=True)

    # Summary
    best_acc_g = float(df_g["val_accuracy"].max())
    min_x = float(df_g[x_metric].min())
    st.write(
        f"**Gen {gen_pick_display} summary:** best accuracy = **{best_acc_g:.4f}**, "
        f"best (min) {x_metric} = **{min_x:,.4f}**"
    )

    # Optional: show PNG if you still want (PNG file uses internal 0-based index)
    if show_png:
        internal_gen = gen_pick_display - 1
        pareto_path = PARETO_DIR / f"gen_{internal_gen}.png"
        if image_exists(pareto_path):
            st.image(str(pareto_path), caption=f"Saved Pareto Image — gen_{internal_gen}.png", use_container_width=True)
        else:
            st.warning(f"Saved PNG missing: `{pareto_path.as_posix()}`")

# -----------------------------
# Best Architecture Panel
# -----------------------------
import json

st.subheader("🏆 Best Architecture DNA")

best_arch_path = BEST_ARCH_PATH  # already run-aware
dna = None

if best_arch_path.exists():
    try:
        dna = json.loads(best_arch_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Could not read best_architecture.json: {e}")
else:
    st.info("No file found: best_architecture.json")

# Get best arch_id from metrics.csv
best_row = None
if len(df) > 0:
    best_row = df.sort_values("val_accuracy", ascending=False).iloc[0]

# Display Best arch_id + metrics
if best_row is not None:
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

    c1.metric("Best arch_id", str(best_row["arch_id"]))
    c2.metric("Best val_accuracy", f"{float(best_row['val_accuracy']):.4f}")
    c3.metric("Params", f"{int(best_row['param_count']):,}")
    c4.metric("Generation", str(int(best_row["generation_display"])))

# Display DNA JSON
if dna is not None:
    with st.expander("Show Architecture DNA"):
        st.json(dna)

st.markdown("---")

# -----------------------------
# Data Table
# -----------------------------
with st.expander("Show filtered metrics table"):
    st.dataframe(
        df_f.sort_values(["generation_display", "val_accuracy"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )

st.caption("✅ Run evolution/training to refresh `outputs/metrics.csv` and the images. Then reload this page.")
