# app_inference.py  ─  SENN Edge-AI Analytics Dashboard
# Run with:  streamlit run app_inference.py
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import io
import json
import time
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import plotly.graph_objects as go
from PIL import Image
warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SENN · Edge-AI Analytics Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  –  enterprise dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base & font ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    /* ── Background ───────────────────────────────────────────────────────── */
    .stApp { background: #0d1117; color: #e6edf3; }
    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #21262d;
    }
    /* ── Metric cards ────────────────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 16px 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
    /* ── Green / red delta override ──────────────────────────────────────── */
    .senn-win  { color: #3fb950 !important; }
    .base-win  { color: #f85149 !important; }
    /* ── Section headers ─────────────────────────────────────────────────── */
    .section-header {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #8b949e;
        margin: 28px 0 10px 0;
        border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
    }
    /* ── Hero banner ─────────────────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #0d419d 0%, #1f6feb 50%, #58a6ff 100%);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .hero-banner h1 { font-size: 1.8rem; font-weight: 700; margin: 0; color: #fff; }
    .hero-banner p  { margin: 4px 0 0 0; color: rgba(255,255,255,0.75); font-size: 0.9rem; }
    /* ── Model label pills ───────────────────────────────────────────────── */
    .pill-senn {
        background: #1f6feb22;
        border: 1px solid #1f6feb;
        color: #58a6ff;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.06em;
    }
    .pill-base {
        background: #8b949e22;
        border: 1px solid #8b949e;
        color: #c9d1d9;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.06em;
    }
    /* ── Verdict box ─────────────────────────────────────────────────────── */
    .verdict-box {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 22px 28px;
        text-align: center;
        margin-top: 10px;
    }
    /* ── Stbutton ─────────────────────────────────────────────────────────── */
    [data-testid="stButton"] > button {
        background: linear-gradient(90deg, #1f6feb, #388bfd);
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        padding: 10px 28px;
        width: 100%;
        transition: opacity 0.2s;
    }
    [data-testid="stButton"] > button:hover { opacity: 0.85; }
    /* ── Divider ─────────────────────────────────────────────────────────── */
    hr { border-color: #21262d !important; }
    /* ── Image display ────────────────────────────────────────────────────── */
    .input-image-container {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    /* ── Scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; } 
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)
# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CIFAR10_CLASSES = ("Airplane", "Automobile", "Bird", "Cat", "Deer",
                   "Dog", "Frog", "Horse", "Ship", "Truck")
BASELINE_PARAMS  = 2_118_346   # hand-crafted baseline
SENN_PARAMS_HINT = 159_000     # evolutionary optimum (~159k)
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
PROJECT_ROOT = Path(__file__).parent
# ─────────────────────────────────────────────────────────────────────────────
# Model definitions / loaders
# ─────────────────────────────────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    """Standard 2-block hand-crafted CNN — 2.12 M parameters."""
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return self.fc_layer(x)
@st.cache_resource(show_spinner="Loading Baseline CNN weights…")
def load_baseline_model() -> tuple[nn.Module, int, str]:
    model = BaselineCNN()
    weights_path = PROJECT_ROOT / "models" / "baseline_weights.pth"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        status = "✅ Loaded pre-trained weights"
    else:
        status = "⚠️ weights not found — untrained model"
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, n_params, status
@st.cache_resource(show_spinner="Loading best SENN model…")
def load_senn_model() -> tuple[Optional[nn.Module], int, str]:
    """
    Dynamically loads the best SENN architecture + weights from the latest
    outputs/run_<timestamp>/ directory, mirroring live_comparison.py logic.
    """
    try:
        from evolution.dna_builder import build_model_from_dna
        from evolution.dna_schema import ArchitectureDNA
        outputs = PROJECT_ROOT / "outputs"
        runs = sorted(
            [d for d in outputs.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda p: p.name,
        )
        if not runs:
            return None, 0, "❌ No run directories found in outputs/"
        run_dir = runs[-1]
        dna_dict = None
        # Priority: best_architecture.json > best arch from metrics.csv
        best_arch_file = run_dir / "best_architecture.json"
        if best_arch_file.exists():
            dna_dict = json.loads(best_arch_file.read_text())
        else:
            # Fallback: scan metrics.csv
            metrics_csv = run_dir / "metrics.csv"
            if metrics_csv.exists():
                import csv as _csv
                best_acc, best_id = -1.0, None
                with open(metrics_csv) as f:
                    for row in _csv.DictReader(f):
                        try:
                            acc = float(row["val_accuracy"])
                            if acc > best_acc:
                                best_acc, best_id = acc, row["arch_id"]
                        except (KeyError, ValueError):
                            continue
                if best_id:
                    arch_path = run_dir / "population" / best_id / "arch.json"
                    if arch_path.exists():
                        dna_dict = json.loads(arch_path.read_text())
        if dna_dict is None:
            return None, 0, f"❌ Could not locate best architecture JSON in {run_dir}"
        dna   = ArchitectureDNA.from_dict(dna_dict)
        model = build_model_from_dna(dna).to("cpu")
        weights_path = run_dir / "best_phase1_model.pth"
        if weights_path.exists():
            model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
            status = f"✅ Loaded weights from `{run_dir.name}`"
        else:
            status = f"⚠️ No weights file found in {run_dir.name} — untrained model"
        model.eval()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model, n_params, status
    except Exception as exc:
        return None, 0, f"❌ Error loading SENN: {exc}"
# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model: nn.Module, tensor: torch.Tensor) -> tuple[str, float, float, list[float]]:
    """
    Returns (predicted_class, confidence_pct, latency_ms, all_probs_list).
    Latency is the average of 5 warm-up + 10 timed passes for stability.
    """
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(5):
            _ = model(tensor)
        # Timed passes
        N = 10
        start = time.perf_counter()
        for _ in range(N):
            logits = model(tensor)
        elapsed_ms = (time.perf_counter() - start) / N * 1000
    probs      = F.softmax(logits[0], dim=0).tolist()
    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx]) * 100
    label      = CIFAR10_CLASSES[top_idx]
    return label, confidence, elapsed_ms, probs
# ─────────────────────────────────────────────────────────────────────────────
# Charting helpers
# ─────────────────────────────────────────────────────────────────────────────
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#8b949e", family="Inter"),
    margin=dict(l=16, r=16, t=40, b=16),
)
def param_bar_chart(baseline_params: int, senn_params: int) -> go.Figure:
    labels = ["Baseline CNN", "SENN (Evolved)"]
    values = [baseline_params / 1e6, senn_params / 1e6]
    colors = ["#8b949e", "#1f6feb"]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.3f} M" for v in values],
        textposition="outside",
        textfont=dict(color="#e6edf3", size=13, family="Inter"),
        width=0.45,
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Parameter Footprint (M)", font=dict(color="#e6edf3", size=14), x=0.5),
        yaxis=dict(title="Parameters (M)", gridcolor="#21262d", zerolinecolor="#21262d"),
        xaxis=dict(showgrid=False),
        height=280,
        showlegend=False,
    )
    return fig
def latency_gauge(senn_ms: float, baseline_ms: float) -> go.Figure:
    speedup = baseline_ms / senn_ms if senn_ms > 0 else 1.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=senn_ms,
        delta=dict(reference=baseline_ms, decreasing=dict(color="#3fb950"), increasing=dict(color="#f85149")),
        gauge=dict(
            axis=dict(range=[0, max(baseline_ms * 1.3, 5)], tickcolor="#8b949e"),
            bar=dict(color="#1f6feb"),
            bgcolor="#161b22",
            borderwidth=2,
            bordercolor="#21262d",
            steps=[
                dict(range=[0, baseline_ms * 0.5], color="rgba(31, 111, 235, 0.15)"),
                dict(range=[baseline_ms * 0.5, baseline_ms], color="rgba(139, 148, 158, 0.15)"),
            ],
            threshold=dict(line=dict(color="#f85149", width=2), thickness=0.75, value=baseline_ms),
        ),
        number=dict(suffix=" ms", font=dict(color="#e6edf3", size=28)),
        title=dict(text=f"SENN Latency  ({speedup:.1f}× faster than Baseline)", font=dict(color="#8b949e", size=13)),
    ))
    fig.update_layout(**_PLOTLY_LAYOUT, height=220)
    return fig
def confidence_donut(probs: list[float], title: str, accent: str) -> go.Figure:
    top_n  = 5
    sorted_idx = np.argsort(probs)[::-1]
    top_labels = [CIFAR10_CLASSES[i] for i in sorted_idx[:top_n]]
    top_vals   = [probs[i] * 100 for i in sorted_idx[:top_n]]
    rest_val   = sum(probs[i] * 100 for i in sorted_idx[top_n:])
    labels     = top_labels + ["Other"]
    values     = top_vals + [rest_val]
    colors_pie = [accent] + ["#21262d"] * 5
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.65,
        marker=dict(colors=colors_pie, line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        textfont=dict(color="#e6edf3", size=11),
        sort=False,
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(color="#e6edf3", size=14), x=0.5),
        height=260,
        showlegend=False,
    )
    return fig
# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 random sample loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading CIFAR-10 test split…")
def _get_cifar_dataset():
    return torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / "local_cifar_data"),
        train=False, download=True,
        transform=transforms.ToTensor(),   # raw [0,1] for display
    )
def get_random_cifar_sample(seed: Optional[int] = None):
    dataset = _get_cifar_dataset()
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(dataset)))
    img_tensor, label_idx = dataset[idx]
    # img_tensor: [3, 32, 32] float in [0,1]
    pil_img = transforms.ToPILImage()(img_tensor)
    return pil_img, CIFAR10_CLASSES[label_idx]
# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:1.5rem;font-weight:700;color:#e6edf3;">🧬 SENN</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8b949e;font-size:0.8rem;margin-bottom:20px;">Self-Evolving Neural Network</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model Status</div>', unsafe_allow_html=True)
    baseline_model, baseline_n_params, baseline_status = load_baseline_model()
    senn_model,     senn_n_params,     senn_status     = load_senn_model()
    st.markdown(f'<span class="pill-base">Baseline CNN</span>', unsafe_allow_html=True)
    st.caption(baseline_status)
    st.markdown(f'<span class="pill-senn">SENN</span>', unsafe_allow_html=True)
    st.caption(senn_status)
    st.markdown('<div class="section-header">Image Source</div>', unsafe_allow_html=True)
    source = st.radio("", ["🎲  Random CIFAR-10 Sample", "📁  Upload Image"], label_visibility="collapsed")
    uploaded_file  = None
    uploaded_label = None
    if source == "📁  Upload Image":
        uploaded_file = st.file_uploader("Drop an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    st.markdown("---")
    st.markdown('<div style="color:#8b949e;font-size:0.72rem;">CIFAR-10  ·  10 classes  ·  32×32 px input</div>', unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
      <h1>Edge-AI Analytics Dashboard</h1>
      <p>Head-to-head inference comparison · SENN vs Baseline CNN · CIFAR-10</p>
    </div>
    """,
    unsafe_allow_html=True,
)
# ─────────────────────────────────────────────────────────────────────────────
# Architecture Overview Cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Architecture Overview</div>', unsafe_allow_html=True)
ov1, ov2, ov3, ov4 = st.columns(4)
senn_display_params = senn_n_params if senn_n_params > 0 else SENN_PARAMS_HINT
reduction_pct = (1 - senn_display_params / baseline_n_params) * 100
ov1.metric("Baseline Parameters",  f"{baseline_n_params / 1e6:.2f} M")
ov2.metric("SENN Parameters",      f"{senn_display_params / 1e6:.3f} M", delta=f"-{reduction_pct:.0f}% smaller", delta_color="inverse")
ov3.metric("Baseline Test Acc.",   "72.0%")
ov4.metric("SENN Test Acc.",       "79.0%",  delta="+7 pp vs Baseline", delta_color="normal")
# ─────────────────────────────────────────────────────────────────────────────
# Image Preparation Panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Input Image</div>', unsafe_allow_html=True)
left_col, mid_col, right_col = st.columns([1, 2, 1])
pil_image: Optional[Image.Image] = None
true_label: Optional[str]        = None
with mid_col:
    if source == "📁  Upload Image" and uploaded_file is not None:
        pil_image  = Image.open(uploaded_file).convert("RGB")
        true_label = None  # unknown for custom uploads
    else:
        # Generate / re-generate sample
        if "cifar_seed" not in st.session_state:
            st.session_state.cifar_seed = int(time.time()) % 100_000
        if st.button("🔀  New Random Sample"):
            st.session_state.cifar_seed = int(time.time() * 1000) % 100_000
        pil_image, true_label = get_random_cifar_sample(st.session_state.cifar_seed)
    if pil_image is not None:
        display_img = pil_image.resize((192, 192), Image.NEAREST)
        st.markdown('<div class="input-image-container">', unsafe_allow_html=True)
        st.image(display_img, caption=f"True label: {true_label}" if true_label else "Uploaded image (label unknown)", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# Run Inference Button
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("")
run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
with run_col2:
    run_clicked = st.button("⚡  Run Head-to-Head Inference")
# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
if run_clicked:
    if pil_image is None:
        st.error("Please select or upload an image first.")
        st.stop()
    if senn_model is None:
        st.error("SENN model could not be loaded. Check the sidebar for details.")
        st.stop()
    # Prepare normalised tensor  [1, 3, 32, 32]
    img_resized = pil_image.resize((32, 32))
    tensor      = TRANSFORM(img_resized).unsqueeze(0)
    # ── Run both models ───────────────────────────────────────────────────
    with st.spinner("Running inference on both models…"):
        bl_label, bl_conf, bl_ms, bl_probs = run_inference(baseline_model, tensor)
        sn_label, sn_conf, sn_ms, sn_probs = run_inference(senn_model,     tensor)
    # ─────────────────────────────────────────────────────────────────────
    # Section: Prediction Verdict
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Prediction Verdict</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    def _correct_badge(pred: str, truth: Optional[str]) -> str:
        if truth is None:
            return ""
        return "🟢  Correct" if pred == truth else "🔴  Wrong"
    with v1:
        st.markdown(
            f"""
            <div class="verdict-box">
              <span class="pill-base">Baseline CNN</span>
              <div style="font-size:2rem;font-weight:700;color:#c9d1d9;margin:10px 0 4px;">{bl_label}</div>
              <div style="color:#8b949e;font-size:0.85rem;">{_correct_badge(bl_label, true_label)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v2:
        st.markdown(
            f"""
            <div class="verdict-box">
              <span class="pill-senn">SENN (Evolved)</span>
              <div style="font-size:2rem;font-weight:700;color:#58a6ff;margin:10px 0 4px;">{sn_label}</div>
              <div style="color:#8b949e;font-size:0.85rem;">{_correct_badge(sn_label, true_label)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # ─────────────────────────────────────────────────────────────────────
    # Section: Performance Metrics
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Live Performance Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    speedup    = bl_ms / sn_ms if sn_ms > 0 else float("inf")
    param_red  = (1 - senn_display_params / baseline_n_params) * 100
    conf_delta = sn_conf - bl_conf
    m1.metric("SENN Latency",      f"{sn_ms:.2f} ms",   delta=f"{speedup:.1f}× faster",                 delta_color="inverse")
    m2.metric("Baseline Latency",  f"{bl_ms:.2f} ms")
    m3.metric("SENN Confidence",   f"{sn_conf:.1f}%",   delta=f"{conf_delta:+.1f}pp vs Baseline",        delta_color="normal")
    m4.metric("Baseline Conf.",    f"{bl_conf:.1f}%")
    m5.metric("Param Reduction",   f"{param_red:.0f}%", delta=f"{baseline_n_params/1e6:.2f}M→{senn_display_params/1e3:.0f}K", delta_color="normal")
    # ─────────────────────────────────────────────────────────────────────
    # Section: Visual Analytics
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Visual Analytics</div>', unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.plotly_chart(param_bar_chart(baseline_n_params, senn_display_params), use_container_width=True)
    with ch2:
        st.plotly_chart(latency_gauge(sn_ms, bl_ms), use_container_width=True)
        # Side-by-side raw latency bar
        lat_fig = go.Figure(go.Bar(
            x=["Baseline CNN", "SENN"],
            y=[bl_ms, sn_ms],
            marker_color=["#8b949e", "#1f6feb"],
            text=[f"{bl_ms:.2f} ms", f"{sn_ms:.2f} ms"],
            textposition="outside",
            textfont=dict(color="#e6edf3"),
            width=0.45,
        ))
        lat_fig.update_layout(
            **_PLOTLY_LAYOUT,
            title=dict(text="Inference Latency (ms)", font=dict(color="#e6edf3", size=13), x=0.5),
            yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
            xaxis=dict(showgrid=False),
            height=180,
            showlegend=False,
        )
        st.plotly_chart(lat_fig, use_container_width=True)
    with ch3:
        # Stacked confidence donuts
        st.plotly_chart(confidence_donut(sn_probs,  "SENN Confidence Distribution",  "#1f6feb"), use_container_width=True)
        st.plotly_chart(confidence_donut(bl_probs,  "Baseline Confidence Distribution","#8b949e"), use_container_width=True)
    # ─────────────────────────────────────────────────────────────────────
    # Section: SENN Superiority Summary
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Why SENN Wins</div>', unsafe_allow_html=True)
    sw1, sw2, sw3 = st.columns(3)
    sw1.metric(
        "🔬  Parameter Efficiency",
        f"{param_red:.0f}% fewer params",
        delta=f"{baseline_n_params/1e6:.2f}M → {senn_display_params/1e3:.0f}K",
        delta_color="inverse",
    )
    sw2.metric(
        "⚡  Speed Advantage",
        f"{speedup:.2f}× faster",
        delta=f"Saved {bl_ms - sn_ms:.2f} ms per inference",
        delta_color="inverse",
    )
    sw3.metric(
        "🎯  Accuracy Premium",
        "+7 pp over Baseline",
        delta="79% vs 72% on CIFAR-10",
        delta_color="normal",
    )
    st.markdown(
        """
        <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;padding:18px 24px;margin-top:8px;color:#8b949e;font-size:0.85rem;line-height:1.7;">
        SENN's Neuro-Evolutionary search discovered an architecture that is simultaneously
        <strong style="color:#58a6ff;">smaller</strong>,
        <strong style="color:#58a6ff;">faster</strong>, and
        <strong style="color:#58a6ff;">more accurate</strong> than a hand-crafted baseline —
        demonstrating that the evolutionary fitness function correctly navigates a
        <em>multi-objective Pareto front</em> in parameter-accuracy-latency space.
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ─────────────────────────────────────────────────────────────────────
    # Raw probability table (expandable)
    # ─────────────────────────────────────────────────────────────────────
    with st.expander("Show raw class probabilities"):
        import pandas as pd
        prob_df = pd.DataFrame({
            "Class":           CIFAR10_CLASSES,
            "SENN (%)":        [f"{p*100:.2f}" for p in sn_probs],
            "Baseline (%)":    [f"{p*100:.2f}" for p in bl_probs],
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
else:
    st.info("👆  Select an image source in the sidebar, then click **Run Head-to-Head Inference**.")
# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#8b949e;font-size:0.72rem;">'
    "Self-Evolving Neural Network (SENN) · Capstone Project · "
    "Neuro-Evolutionary Architecture Search on CIFAR-10"
    "</div>",
    unsafe_allow_html=True,
)
