"""
app.py — Diabetic Retinopathy Analysis  (Streamlit)
=====================================================
Run with:
    streamlit run app.py

Expects:
    weights/convnext_ordinal_checkpoint.pth   ← ConvNeXt-Base ordinal checkpoint
    weights/best_weights_only.pth             ← FR-UNet segmentation weights
    models.py                                 ← in the same directory
"""
import os
from huggingface_hub import hf_hub_download

os.makedirs("weights", exist_ok=True)

clf_path = "weights/convnext_ordinal_checkpoint.pth"
seg_path = "weights/best_weights_only.pth"

if not os.path.exists(clf_path):
    hf_hub_download(repo_id="AnshisUWU/dr-weights",
                    filename="convnext_ordinal_checkpoint.pth",
                    local_dir="weights")

if not os.path.exists(seg_path):
    hf_hub_download(repo_id="AnshisUWU/dr-weights",
                    filename="best_weights_only.pth",
                    local_dir="weights")

import io
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch

from models import load_models, predict_single, compute_saliency_maps, GRADE_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DR Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded weight paths (not shown to user)
# ─────────────────────────────────────────────────────────────────────────────

CLF_PATH = "weights/convnext_ordinal_checkpoint.pth"
SEG_PATH = "weights/best_weights_only.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Display options")
    show_prob_map = st.checkbox("Show raw vessel probability map", value=False)
    show_saliency = st.checkbox("Show saliency maps (GradCAM++ & EigenCAM)", value=False)

    st.markdown("---")
    st.caption("Models: ConvNeXt-Base (classification) + FR-UNet (segmentation)")

# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached — weights load only once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model weights…")
def get_models(clf_weight_path: str, seg_weight_path: str):
    try:
        clf, seg, device = load_models(clf_weight_path, seg_weight_path)
        return clf, seg, device, None
    except FileNotFoundError as e:
        return None, None, None, str(e)
    except Exception as e:
        return None, None, None, f"Unexpected error: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# Grade colour and description helpers
# ─────────────────────────────────────────────────────────────────────────────

GRADE_COLORS = {
    0: "#2ECC71",
    1: "#F1C40F",
    2: "#E67E22",
    3: "#E74C3C",
    4: "#8E44AD",
}

GRADE_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Mild NPDR: microaneurysms only.",
    2: "Moderate NPDR: more than microaneurysms but less than severe.",
    3: "Severe NPDR: any of 4-2-1 rule features without PDR signs.",
    4: "Proliferative DR: neovascularisation or vitreous/pre-retinal haemorrhage.",
}

def grade_badge(grade: int) -> str:
    color = GRADE_COLORS[grade]
    name  = GRADE_NAMES[grade]
    return (
        f'<span style="background-color:{color};color:white;padding:6px 16px;'
        f'border-radius:20px;font-size:1.1rem;font-weight:600;">{name}</span>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_probability_bar_chart(class_probs: np.ndarray) -> plt.Figure:
    colors = [GRADE_COLORS[i] for i in range(len(GRADE_NAMES))]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    bars = ax.barh(GRADE_NAMES, class_probs * 100, color=colors,
                   edgecolor="white", height=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=9, color="gray")
    ax.tick_params(colors="gray", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, val in zip(bars, class_probs):
        if val > 0.02:
            ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=8, color="gray")
    fig.tight_layout(pad=0.5)
    return fig


def overlay_contours(base_img: np.ndarray, mask_binary: np.ndarray) -> np.ndarray:
    out  = base_img.copy()
    mask = (mask_binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.drawContours(out_bgr, contours, -1, (0, 220, 60), 1)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def make_cam_figure(
    base_f32: np.ndarray,
    cam_raw_gradcam: np.ndarray,
    overlay_gradcam: np.ndarray,
    cam_raw_eigen: np.ndarray,
    overlay_eigen: np.ndarray,
    grade_name: str,
) -> plt.Figure:
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Saliency Maps  |  Predicted grade: {grade_name}  |  "
        "Target layer: ConvNeXt last stage",
        fontsize=12,
    )
    panels = [
        (base_f32,        None,   "Preprocessed input"),
        (cam_raw_gradcam, "jet",  "GradCAM++ (raw activation)"),
        (overlay_gradcam, None,   "GradCAM++ overlay"),
        (overlay_eigen,   None,   "EigenCAM overlay"),
    ]
    for ax, (img, cmap, title) in zip(axs, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap integrity check
# ─────────────────────────────────────────────────────────────────────────────

def analyse_cam_activation(cam_map: np.ndarray) -> dict:
    """
    Extract quantitative metrics from a GradCAM/EigenCAM heatmap (values 0-1).

    Returns:
        activation_coverage_pct  — % of pixels above the 0.5 threshold
        peak_intensity           — maximum activation value in the map
        mean_hot_intensity       — mean activation within the hot zone only
        spatial_concentration    — std-dev of the full map (high = focused spot)
        suspicion_score          — composite score combining the three key metrics
    """
    cam = np.clip(cam_map.astype(np.float32), 0.0, 1.0)

    hot_mask              = cam > 0.5
    activation_coverage   = hot_mask.mean() * 100.0
    peak_intensity        = float(cam.max())
    hot_zone              = cam[hot_mask]
    mean_hot_intensity    = float(hot_zone.mean()) if hot_zone.size > 0 else 0.0
    spatial_concentration = float(cam.std())

    # Suspicion score: composite 0-100 range (roughly)
    suspicion_score = float(activation_coverage * peak_intensity * mean_hot_intensity)

    return {
        "activation_coverage_pct": round(activation_coverage,   2),
        "peak_intensity":          round(peak_intensity,         3),
        "mean_hot_intensity":      round(mean_hot_intensity,     3),
        "spatial_concentration":   round(spatial_concentration,  3),
        "suspicion_score":         round(suspicion_score,        3),
    }


def heatmap_grade_agreement_check(
    grade: int,
    class_probs: np.ndarray,
    cam_gradcam: np.ndarray,
    cam_eigen: np.ndarray,
    coverage_threshold: float  = 3.0,
    suspicion_threshold: float = 15.0,
) -> dict:
    """
    Compare the model's grade prediction against BOTH GradCAM++ and EigenCAM.

    A disagreement is flagged when:
      - The model predicted Grade 0 (No DR), AND
      - At least one heatmap has activation_coverage > coverage_threshold AND
        suspicion_score > suspicion_threshold.

    A low-confidence flag is raised when:
      - The model predicted Grade 0 or 1 with < 60% confidence, AND
      - At least one heatmap is suspicious.
    """
    stats_gc = analyse_cam_activation(cam_gradcam)
    stats_ec = analyse_cam_activation(cam_eigen)

    def _implied_grade(suspicion: float) -> int:
        if suspicion < 10:  return 0
        if suspicion < 25:  return 1
        if suspicion < 50:  return 2
        if suspicion < 80:  return 3
        return 4

    implied_gc    = _implied_grade(stats_gc["suspicion_score"])
    implied_ec    = _implied_grade(stats_ec["suspicion_score"])
    implied_grade = max(implied_gc, implied_ec)   # take the higher concern

    gc_flags = (
        stats_gc["activation_coverage_pct"] > coverage_threshold and
        stats_gc["suspicion_score"]          > suspicion_threshold
    )
    ec_flags = (
        stats_ec["activation_coverage_pct"] > coverage_threshold and
        stats_ec["suspicion_score"]          > suspicion_threshold
    )

    heatmap_suggests_pathology = gc_flags or ec_flags
    model_said_clean           = (grade == 0)
    disagreement               = model_said_clean and heatmap_suggests_pathology

    low_conf_flag = (
        class_probs[grade] < 0.6 and
        heatmap_suggests_pathology and
        grade <= 1
    )

    combined_suspicion = round(
        (stats_gc["suspicion_score"] + stats_ec["suspicion_score"]) / 2, 3
    )

    return {
        "disagreement":               disagreement,
        "low_confidence_flag":        low_conf_flag,
        "heatmap_suggests_pathology": heatmap_suggests_pathology,
        "gradcam_flags":              gc_flags,
        "eigencam_flags":             ec_flags,
        "implied_grade":              implied_grade,
        "implied_grade_name":         GRADE_NAMES[implied_grade],
        "combined_suspicion":         combined_suspicion,
        "stats_gradcam":              stats_gc,
        "stats_eigencam":             stats_ec,
    }


def render_heatmap_integrity_section(
    agreement: dict,
    grade: int,
    grade_name: str,
    class_probs: np.ndarray,
) -> None:
    """Render the Heatmap Integrity Check section in Streamlit."""

    st.markdown("---")
    st.markdown("### 🔍 Heatmap Integrity Check")
    st.caption(
        "Independently analyses the GradCAM++ and EigenCAM activation maps to "
        "verify whether the saliency evidence is consistent with the model's "
        "grade prediction. Acts as an automatic second-opinion system."
    )

    sg = agreement["stats_gradcam"]
    se = agreement["stats_eigencam"]

    # ── Main verdict banner ──────────────────────────────────────────────────
    if agreement["disagreement"]:
        st.error(
            f"**⚠️ Model–Heatmap Disagreement Detected**\n\n"
            f"The classifier predicted **Grade {grade} — {grade_name}** "
            f"but the saliency maps show significant activation over suspicious "
            f"regions that were not weighted heavily enough by the classification "
            f"head — likely due to label noise in the training data.\n\n"
            f"**Heatmap-implied grade: {agreement['implied_grade']} "
            f"— {agreement['implied_grade_name']}**\n\n"
            f"*This image should be reviewed by a qualified ophthalmologist.*"
        )
    elif agreement["low_confidence_flag"]:
        st.warning(
            f"**⚠️ Low-Confidence Prediction with Suspicious Heatmap**\n\n"
            f"The model predicted **Grade {grade} — {grade_name}** with only "
            f"**{class_probs[grade]*100:.1f}%** confidence, and the heatmaps "
            f"show notable activation. Treat this as a borderline case.\n\n"
            f"**Heatmap-implied grade: {agreement['implied_grade']} "
            f"— {agreement['implied_grade_name']}**"
        )
    else:
        st.success(
            f"**✅ Model and Heatmap are Consistent**\n\n"
            f"The saliency maps support the model's prediction of "
            f"**Grade {grade} — {grade_name}**. "
            f"No significant disagreement detected."
        )

    # ── Per-CAM detail columns ───────────────────────────────────────────────
    st.markdown("#### Activation Statistics")
    col_gc, col_ec = st.columns(2)

    def _cam_metrics_card(col, title: str, stats: dict, flagged: bool):
        flag_icon = "🔴" if flagged else "🟢"
        with col:
            st.markdown(f"**{flag_icon} {title}**")
            st.markdown(
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| Activation coverage | **{stats['activation_coverage_pct']}%** of image |\n"
                f"| Peak intensity | **{stats['peak_intensity']}** |\n"
                f"| Mean hot-zone intensity | **{stats['mean_hot_intensity']}** |\n"
                f"| Spatial concentration (σ) | **{stats['spatial_concentration']}** |\n"
                f"| Suspicion score | **{stats['suspicion_score']}** |"
            )

    _cam_metrics_card(col_gc, "GradCAM++", sg, agreement["gradcam_flags"])
    _cam_metrics_card(col_ec, "EigenCAM",  se, agreement["eigencam_flags"])

    # ── Combined suspicion gauge ─────────────────────────────────────────────
    st.markdown("#### Combined Suspicion Score")
    cs = agreement["combined_suspicion"]
    progress_val = min(cs / 100.0, 1.0)

    if cs < 10:
        level, level_color = "Very Low",  "#2ECC71"
    elif cs < 25:
        level, level_color = "Low",       "#F1C40F"
    elif cs < 50:
        level, level_color = "Moderate",  "#E67E22"
    elif cs < 80:
        level, level_color = "High",      "#E74C3C"
    else:
        level, level_color = "Very High", "#8E44AD"

    st.markdown(
        f'<div style="margin-bottom:4px">Score: '
        f'<span style="color:{level_color};font-weight:700;">'
        f'{cs:.1f} / 100 — {level}</span></div>',
        unsafe_allow_html=True,
    )
    st.progress(progress_val)

    # ── Interpretation guide ─────────────────────────────────────────────────
    with st.expander("ℹ️ How to read these results"):
        st.markdown(
            """
**Activation coverage** — What percentage of the retinal image contains
"hot" regions (activation > 0.5). A healthy Grade 0 image typically has
very low coverage; lesions drive this number up.

**Peak intensity** — The maximum activation value anywhere in the map.
Values close to 1.0 indicate a very strong, localised response — the network
is highly confident it found something at that spot.

**Mean hot-zone intensity** — Average activation within the hot zone only.
High values mean the detected regions are strongly activated, not just
barely above the threshold.

**Spatial concentration (σ)** — Standard deviation of the full activation
map. A high σ means heat is focused on specific spots (consistent with
discrete lesions). A low σ means diffuse, uniform activation (less specific).

**Suspicion score** — Composite metric combining coverage, peak, and
hot-zone intensity. Roughly:
- < 10 → Consistent with no pathology
- 10–25 → Mild concern (possible early lesions)
- 25–50 → Moderate concern
- 50–80 → High concern
- > 80 → Very high concern — expert review strongly recommended

**Disagreement flag** — Raised when the model says Grade 0 but the
heatmap suspicion score exceeds the detection threshold. This does NOT
mean the model is definitely wrong — it means the visual evidence
warrants a second look by a clinician.
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# PDF report builder
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_png_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def pil_to_png_bytes(img) -> bytes:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def build_pdf_report(
    original_image: Image.Image,
    grade: int,
    grade_name: str,
    class_probs: np.ndarray,
    overlay_rgb: np.ndarray,
    mask_binary: np.ndarray,
    mask_prob: np.ndarray,
    saliency: dict | None,
    agreement: dict | None,
) -> bytes:
    """
    Build a comprehensive PDF report using ReportLab Platypus.
    Includes: classification result, probability chart, all image panels,
    raw probability map, optional saliency figures, and heatmap integrity report.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, HRFlowable, PageBreak,
    )

    buf      = io.BytesIO()
    doc      = SimpleDocTemplate(buf, pagesize=A4,
                                 leftMargin=2*cm, rightMargin=2*cm,
                                 topMargin=2*cm,  bottomMargin=2*cm)
    styles   = getSampleStyleSheet()
    grade_hex = GRADE_COLORS[grade]
    page_w   = A4[0] - 4*cm
    half_w   = (page_w - 0.4*cm) / 2
    third_w  = (page_w - 0.8*cm) / 3

    title_style       = ParagraphStyle("RT",  parent=styles["Title"],    fontSize=20, spaceAfter=6,  alignment=TA_CENTER)
    subtitle_style    = ParagraphStyle("ST",  parent=styles["Normal"],   fontSize=10, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=12)
    section_style     = ParagraphStyle("SEC", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#2C3E50"))
    body_style        = ParagraphStyle("BOD", parent=styles["Normal"],   fontSize=10, spaceAfter=4)
    caption_style     = ParagraphStyle("CAP", parent=styles["Normal"],   fontSize=8,  textColor=colors.grey, alignment=TA_CENTER, spaceAfter=8)
    grade_label_style = ParagraphStyle("GL",  parent=styles["Normal"],   fontSize=14, fontName="Helvetica-Bold", textColor=colors.HexColor(grade_hex), alignment=TA_LEFT)

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Diabetic Retinopathy Analysis Report", title_style))
    story.append(Paragraph("ConvNeXt-Base (classification) + FR-UNet (segmentation)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BDC3C7")))
    story.append(Spacer(1, 10))

    # ── Classification result ─────────────────────────────────────────────────
    story.append(Paragraph("Classification Result", section_style))
    story.append(Paragraph(f"<b>DR Grade {grade} / 4  —  {grade_name}</b>", grade_label_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(GRADE_DESCRIPTIONS[grade], body_style))
    story.append(Spacer(1, 6))

    conf_data = [["Grade", "Name", "Probability"]]
    for i, (name, prob) in enumerate(zip(GRADE_NAMES, class_probs)):
        conf_data.append([str(i), name, f"{prob*100:.1f}%"])
    conf_table = Table(conf_data, colWidths=[2*cm, 8*cm, 4*cm])
    conf_table.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),         colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",      (0,0),(-1,0),         colors.white),
        ("FONTNAME",       (0,0),(-1,0),         "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1),        9),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),        [colors.HexColor("#F9F9F9"), colors.white]),
        ("GRID",           (0,0),(-1,-1), 0.5,  colors.HexColor("#BDC3C7")),
        ("ALIGN",          (2,0),(2,-1),         "CENTER"),
        ("FONTNAME",       (0,grade+1),(-1,grade+1), "Helvetica-Bold"),
        ("TEXTCOLOR",      (0,grade+1),(-1,grade+1), colors.HexColor(grade_hex)),
    ]))
    story.append(conf_table)
    story.append(Spacer(1, 10))

    if class_probs[grade] < 0.6:
        story.append(Paragraph("<b>Low confidence prediction</b> — consider expert review.",
                                ParagraphStyle("warn", parent=body_style, textColor=colors.HexColor("#E67E22"))))
    else:
        story.append(Paragraph("<b>High confidence prediction.</b>",
                                ParagraphStyle("ok", parent=body_style, textColor=colors.HexColor("#27AE60"))))
    story.append(Spacer(1, 6))

    # ── Probability chart ─────────────────────────────────────────────────────
    story.append(Paragraph("Grade Probability Distribution", section_style))
    fig_chart = make_probability_bar_chart(class_probs)
    fig_chart.patch.set_facecolor("white")
    chart_png = fig_to_png_bytes(fig_chart, dpi=150)
    plt.close(fig_chart)
    story.append(RLImage(io.BytesIO(chart_png), width=page_w*0.65, height=page_w*0.35))
    story.append(Spacer(1, 10))

    # ── Image analysis ────────────────────────────────────────────────────────
    story.append(Paragraph("Image Analysis", section_style))
    orig_png    = pil_to_png_bytes(original_image)
    overlay_png = pil_to_png_bytes(overlay_rgb)
    base_512    = np.array(original_image.resize((512, 512)))
    contour_img = overlay_contours(base_512, mask_binary)
    contour_png = pil_to_png_bytes(contour_img)
    vessel_pct  = mask_binary.mean() * 100

    img_row = Table([[
        RLImage(io.BytesIO(orig_png),    width=third_w, height=third_w),
        RLImage(io.BytesIO(overlay_png), width=third_w, height=third_w),
        RLImage(io.BytesIO(contour_png), width=third_w, height=third_w),
    ]], colWidths=[third_w+0.13*cm]*3)
    img_row.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(img_row)
    story.append(Table([[
        Paragraph("Original Image", caption_style),
        Paragraph(f"Vessel Segmentation Overlay<br/>(vessels: {vessel_pct:.1f}% of pixels)", caption_style),
        Paragraph("Vessel Contour Map", caption_style),
    ]], colWidths=[third_w+0.13*cm]*3))
    story.append(Spacer(1, 10))

    # ── Raw vessel probability map ────────────────────────────────────────────
    story.append(Paragraph("Raw Vessel Probability Map", section_style))
    story.append(Paragraph(
        "Pixel-level sigmoid probability output from the FR-UNet segmentation model. "
        "Warmer colours indicate higher vessel likelihood.", body_style))
    fig_prob, ax_prob = plt.subplots(figsize=(5, 5))
    im = ax_prob.imshow(mask_prob, cmap="hot", vmin=0, vmax=1)
    ax_prob.axis("off")
    ax_prob.set_title("Sigmoid probability per pixel", fontsize=10)
    fig_prob.colorbar(im, ax=ax_prob, fraction=0.046, pad=0.04)
    fig_prob.patch.set_facecolor("white")
    fig_prob.tight_layout()
    prob_png = fig_to_png_bytes(fig_prob, dpi=150)
    plt.close(fig_prob)
    story.append(RLImage(io.BytesIO(prob_png), width=page_w*0.5, height=page_w*0.5))
    story.append(Spacer(1, 10))

    # ── Saliency maps ─────────────────────────────────────────────────────────
    if saliency is not None:
        story.append(PageBreak())
        story.append(Paragraph("Saliency Maps — GradCAM++ & EigenCAM", section_style))
        story.append(Paragraph(
            "Highlights the retinal regions that most influenced the grade prediction. "
            "Warmer colours (red/yellow) = higher importance. "
            "Target layer: last ConvNeXt stage.", body_style))
        story.append(Spacer(1, 6))

        cam_fig = make_cam_figure(
            base_f32        = saliency["base_img_f32"],
            cam_raw_gradcam = saliency["cam_gradcam"],
            overlay_gradcam = saliency["overlay_gradcam"],
            cam_raw_eigen   = saliency["cam_eigen"],
            overlay_eigen   = saliency["overlay_eigen"],
            grade_name      = grade_name,
        )
        cam_fig.patch.set_facecolor("white")
        cam_png = fig_to_png_bytes(cam_fig, dpi=120)
        plt.close(cam_fig)
        story.append(RLImage(io.BytesIO(cam_png), width=page_w, height=page_w*0.26))
        story.append(Paragraph(
            "Left to right: Preprocessed input | GradCAM++ raw activation | "
            "GradCAM++ overlay | EigenCAM overlay", caption_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Close-up Comparison", section_style))
        gc_img = saliency["overlay_gradcam"]
        ec_img = saliency["overlay_eigen"]
        gc_png = pil_to_png_bytes((gc_img*255).astype(np.uint8) if gc_img.max() <= 1.0 else gc_img)
        ec_png = pil_to_png_bytes((ec_img*255).astype(np.uint8) if ec_img.max() <= 1.0 else ec_img)
        sal_row = Table([[
            RLImage(io.BytesIO(gc_png), width=half_w, height=half_w),
            RLImage(io.BytesIO(ec_png), width=half_w, height=half_w),
        ]], colWidths=[half_w+0.2*cm, half_w+0.2*cm])
        sal_row.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        story.append(sal_row)
        story.append(Table([[
            Paragraph("GradCAM++ overlay<br/><i>Uses gradient information flowing back "
                      "from the target class to weight the feature map activations.</i>", caption_style),
            Paragraph("EigenCAM overlay<br/><i>Uses the first principal component of the "
                      "feature maps — no gradients needed, faster and more stable.</i>", caption_style),
        ]], colWidths=[half_w+0.2*cm, half_w+0.2*cm]))
        story.append(Spacer(1, 10))

    # ── Heatmap integrity section ─────────────────────────────────────────────
    if agreement is not None:
        story.append(PageBreak())
        story.append(Paragraph("Heatmap Integrity Check", section_style))
        story.append(Paragraph(
            "Independent quantitative analysis of saliency map activations to verify "
            "consistency with the model's grade prediction. Acts as an automatic "
            "second-opinion system.", body_style))
        story.append(Spacer(1, 6))

        if agreement["disagreement"]:
            verdict_text  = (f"<b>DISAGREEMENT DETECTED:</b> Model predicted Grade {grade} ({grade_name}) "
                             f"but heatmap analysis implies Grade {agreement['implied_grade']} "
                             f"({agreement['implied_grade_name']}). Expert review is recommended.")
            verdict_color = colors.HexColor("#E74C3C")
        elif agreement["low_confidence_flag"]:
            verdict_text  = (f"<b>LOW-CONFIDENCE WARNING:</b> Model predicted Grade {grade} ({grade_name}) "
                             f"with low confidence and suspicious heatmap activation. "
                             f"Borderline case — review advised.")
            verdict_color = colors.HexColor("#E67E22")
        else:
            verdict_text  = (f"<b>CONSISTENT:</b> Heatmap activation supports the model's "
                             f"prediction of Grade {grade} ({grade_name}).")
            verdict_color = colors.HexColor("#27AE60")

        story.append(Paragraph(verdict_text,
                                ParagraphStyle("verdict", parent=body_style,
                                               textColor=verdict_color, fontSize=11)))
        story.append(Spacer(1, 8))

        sg = agreement["stats_gradcam"]
        se = agreement["stats_eigencam"]
        stats_data = [
            ["Metric", "GradCAM++", "EigenCAM"],
            ["Activation coverage (%)", str(sg["activation_coverage_pct"]), str(se["activation_coverage_pct"])],
            ["Peak intensity",          str(sg["peak_intensity"]),          str(se["peak_intensity"])],
            ["Mean hot-zone intensity", str(sg["mean_hot_intensity"]),      str(se["mean_hot_intensity"])],
            ["Spatial concentration (σ)",str(sg["spatial_concentration"]), str(se["spatial_concentration"])],
            ["Suspicion score",          str(sg["suspicion_score"]),        str(se["suspicion_score"])],
            ["Combined suspicion score", str(agreement["combined_suspicion"]), "—"],
            ["Heatmap-implied grade",
             f"{agreement['implied_grade']} — {agreement['implied_grade_name']}", "—"],
        ]
        stats_table = Table(stats_data, colWidths=[7*cm, 4*cm, 4*cm])
        stats_table.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0),         colors.HexColor("#2C3E50")),
            ("TEXTCOLOR",      (0,0),(-1,0),         colors.white),
            ("FONTNAME",       (0,0),(-1,0),         "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1),        9),
            ("ROWBACKGROUNDS", (0,1),(-1,-1),        [colors.HexColor("#F9F9F9"), colors.white]),
            ("GRID",           (0,0),(-1,-1), 0.5,  colors.HexColor("#BDC3C7")),
            ("ALIGN",          (1,0),(-1,-1),        "CENTER"),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 10))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#BDC3C7")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Made by Ansh (22BEC0057) and Chinmaya Prakash (22BEC0624). "
        "Under the guide Dr. M. Jasmine Pemeena Priyadarsini. "
        "School of Electronics Engineering, Vellore Institute of Technology, Vellore, India.",
        ParagraphStyle("footer", parent=body_style, fontSize=8,
                       textColor=colors.grey, alignment=TA_CENTER)))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("👁️ Diabetic Retinopathy Analysis")
st.markdown(
    "Upload a fundus photograph to receive a **DR severity grade** "
    "and a **retinal vessel segmentation map**."
)

clf, seg, device, load_error = get_models(CLF_PATH, SEG_PATH)

if load_error:
    st.error(f"**Could not load model weights.**\n\n{load_error}")
    st.info(
        "Expected folder layout:\n```\ndr_app/\n├── app.py\n├── models.py\n"
        "└── weights/\n    ├── convnext_ordinal_checkpoint.pth\n"
        "    └── best_weights_only.pth\n```"
    )
    st.stop()

device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
st.sidebar.success(f"Models loaded — running on **{device_label}**")

uploaded_file = st.file_uploader(
    "Upload a fundus image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="Accepts standard fundus photo formats. Processed locally.",
)

if uploaded_file is None:
    st.info("Upload a fundus image above to begin analysis.")
    with st.expander("ℹ️ About this app"):
        st.markdown(
            """
            **Classification**: ConvNeXt-Base with an ordinal regression head
            trained on EyePACS / APTOS / Messidor. Predicts DR severity (grade 0–4).

            **Segmentation**: FR-UNet trained on FIVES, CHASE, STARE and DRIVE.
            Produces pixel-level retinal vessel maps at 512×512.

            **Saliency maps** (optional): GradCAM++ and EigenCAM computed on the
            last ConvNeXt stage, showing which regions drove the grade prediction.

            **Heatmap integrity check** (automatic when saliency is enabled):
            Independently analyses activation maps to detect cases where the
            heatmap evidence contradicts the model's classification — acting as
            an automatic second-opinion system.

            **Preprocessing applied before each model**:
            fundus circle crop → CLAHE (LAB space) → resize → ImageNet normalisation.
            """
        )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

image = Image.open(uploaded_file).convert("RGB")

with st.spinner("Running classification and segmentation…"):
    results = predict_single(image, clf, seg, device)

grade       = results["grade"]
grade_name  = results["grade_name"]
class_probs = results["class_probs"]
mask_prob   = results["mask_prob"]
mask_binary = results["mask_binary"]
overlay_rgb = results["overlay_rgb"]

# ─────────────────────────────────────────────────────────────────────────────
# Grade result banner
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Classification Result")
col_badge, col_desc = st.columns([1, 3])
with col_badge:
    st.markdown(grade_badge(grade), unsafe_allow_html=True)
    st.markdown(f"**Grade {grade} / 4**")
with col_desc:
    st.markdown(GRADE_DESCRIPTIONS[grade])

# ─────────────────────────────────────────────────────────────────────────────
# Image panels
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Image Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Original image**")
    st.image(image, use_container_width=True)

with col2:
    st.markdown("**Vessel segmentation overlay**")
    st.image(overlay_rgb, use_container_width=True)
    vessel_pct = mask_binary.mean() * 100
    st.caption(f"Vessels detected: **{vessel_pct:.1f}%** of pixels")

with col3:
    st.markdown("**Vessel contour map**")
    base_512    = np.array(image.resize((512, 512)))
    contour_img = overlay_contours(base_512, mask_binary)
    st.image(contour_img, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Probability chart
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Grade Probability Distribution")

chart_col, info_col = st.columns([2, 1])

with chart_col:
    fig = make_probability_bar_chart(class_probs)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with info_col:
    st.markdown("**Confidence scores**")
    for i, (name, prob) in enumerate(zip(GRADE_NAMES, class_probs)):
        color = GRADE_COLORS[i]
        st.markdown(
            f'<span style="color:{color};font-weight:600;">{name}</span>: '
            f'{prob*100:.1f}%',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    if class_probs[grade] < 0.6:
        st.warning("Low confidence prediction — consider expert review.")
    else:
        st.success("High confidence prediction.")

# ─────────────────────────────────────────────────────────────────────────────
# Optional: raw vessel probability map
# ─────────────────────────────────────────────────────────────────────────────

if show_prob_map:
    st.markdown("---")
    st.markdown("### Raw Vessel Probability Map")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    im = ax2.imshow(mask_prob, cmap="hot", vmin=0, vmax=1)
    ax2.axis("off")
    ax2.set_title("Sigmoid probability per pixel", fontsize=10)
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# Optional: GradCAM++ / EigenCAM + Heatmap Integrity Check
# ─────────────────────────────────────────────────────────────────────────────

saliency_data  = None
agreement_data = None

if show_saliency:
    st.markdown("---")
    st.markdown("### Saliency Maps — GradCAM++ & EigenCAM")
    st.caption(
        "Highlights the retinal regions that most influenced the grade prediction. "
        "Warmer colours (red/yellow) = higher importance. "
        "Target layer: last ConvNeXt stage."
    )

    try:
        import pytorch_grad_cam  # noqa: F401
        grad_cam_available = True
    except ImportError:
        grad_cam_available = False

    if not grad_cam_available:
        st.warning(
            "The `grad-cam` package is not installed. "
            "Run `pip install grad-cam` in your terminal then restart the app."
        )
    else:
        with st.spinner("Computing GradCAM++ and EigenCAM…"):
            try:
                saliency_data = compute_saliency_maps(image, clf, device, grade)

                cam_fig = make_cam_figure(
                    base_f32        = saliency_data["base_img_f32"],
                    cam_raw_gradcam = saliency_data["cam_gradcam"],
                    overlay_gradcam = saliency_data["overlay_gradcam"],
                    cam_raw_eigen   = saliency_data["cam_eigen"],
                    overlay_eigen   = saliency_data["overlay_eigen"],
                    grade_name      = grade_name,
                )
                st.pyplot(cam_fig, use_container_width=True)
                plt.close(cam_fig)

                st.markdown("#### Close-up comparison")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**GradCAM++ overlay**")
                    st.image(saliency_data["overlay_gradcam"], use_container_width=True)
                    st.caption(
                        "Uses gradient information flowing back from the target "
                        "class to weight the feature map activations."
                    )
                with c2:
                    st.markdown("**EigenCAM overlay**")
                    st.image(saliency_data["overlay_eigen"], use_container_width=True)
                    st.caption(
                        "Uses the first principal component of the feature maps — "
                        "no gradients needed, faster and more stable."
                    )

                # ── Run heatmap integrity check automatically ─────────────────
                agreement_data = heatmap_grade_agreement_check(
                    grade       = grade,
                    class_probs = class_probs,
                    cam_gradcam = saliency_data["cam_gradcam"],
                    cam_eigen   = saliency_data["cam_eigen"],
                )
                render_heatmap_integrity_section(
                    agreement_data, grade, grade_name, class_probs
                )

            except Exception as e:
                st.error(f"Saliency computation failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Download section
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Export")

dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    overlay_pil = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download vessel overlay (PNG)",
        data=buf.getvalue(),
        file_name="vessel_overlay.png",
        mime="image/png",
    )

with dl_col2:
    with st.spinner("Generating PDF report…"):
        try:
            pdf_bytes = build_pdf_report(
                original_image = image,
                grade          = grade,
                grade_name     = grade_name,
                class_probs    = class_probs,
                overlay_rgb    = overlay_rgb,
                mask_binary    = mask_binary,
                mask_prob      = mask_prob,
                saliency       = saliency_data,
                agreement      = agreement_data,
            )
            st.download_button(
                label="⬇️ Download full PDF report",
                data=pdf_bytes,
                file_name="dr_analysis_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(
                f"PDF generation failed: {e}\n\n"
                "Make sure `reportlab` is installed: `pip install reportlab`"
            )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Made by Ansh(22BEC0057) and Chinmaya Prakash(22BEC0624). "
    "Under the guide Dr.M. Jasmine Pemeena Priyadarsini. "
    "School of Electronics Engineering, Vellore Institute of Technology, Vellore, India"
)
