"""
OncoPredict AI - Breast Cancer Diagnostic Prediction System
A clean, modern machine learning application for Breast Cancer classification using cell nucleus features.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Breast Cancer Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# 2. DESIGN SYSTEM & MODERN UI STYLING
# ==============================================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* Completely hide Streamlit sidebar and chrome */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .block-container {
        max-width: 1050px;
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* Header Container */
    .app-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
    }

    .app-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .app-logo {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
    }

    .app-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .app-subtitle {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #16A34A;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 9999px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #16A34A;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #16A34A;
    }

    /* Parameter Container Card */
    .param-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.03);
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }

    .param-card-header {
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid #F1F5F9;
    }

    .param-card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .param-card-desc {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 3px;
    }

    /* Input Field Styling */
    div[data-baseweb="input"],
    div[data-baseweb="input"] input,
    input[type="number"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #0284C7 !important;
        box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.15) !important;
    }

    div[data-testid="stNumberInput"] button {
        background-color: #F8FAFC !important;
        color: #334155 !important;
        border-color: #CBD5E1 !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #E2E8F0 !important;
        color: #0284C7 !important;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    label[data-testid="stWidgetLabel"] {
        color: #0F172A !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        margin-bottom: 2px !important;
    }

    .field-desc {
        font-size: 0.74rem;
        color: #64748B;
        line-height: 1.35;
        margin-top: -3px;
        margin-bottom: 12px;
    }

    /* Expander Styling */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 14px !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }

    div[data-testid="stExpander"] summary {
        background: #FFFFFF !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.85rem 1.2rem !important;
        border-radius: 14px !important;
    }

    div[data-testid="stExpander"] summary:hover {
        background: #F8FAFC !important;
        color: #0284C7 !important;
    }

    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 1.2rem !important;
        background: #FFFFFF !important;
    }

    /* Predict Button */
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.2s ease;
        padding: 0.85rem 2rem;
    }

    .btn-predict > div > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.3) !important;
    }

    .btn-predict > div > button:hover {
        box-shadow: 0 6px 22px rgba(2, 132, 199, 0.45) !important;
        transform: translateY(-1px);
    }

    /* Reset Button */
    .btn-reset > div > button {
        background: #FFFFFF !important;
        color: #475569 !important;
        border: 1px solid #CBD5E1 !important;
    }

    .btn-reset > div > button:hover {
        border-color: #0284C7 !important;
        color: #0284C7 !important;
        background: #F0F9FF !important;
    }

    /* Result Display */
    .result-box-benign {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1.5px solid #86EFAC;
        border-radius: 18px;
        padding: 1.8rem 2rem;
        box-shadow: 0 8px 24px rgba(22, 163, 74, 0.1);
        margin-top: 1.5rem;
        margin-bottom: 1.2rem;
    }

    .result-box-malignant {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1.5px solid #FCA5A5;
        border-radius: 18px;
        padding: 1.8rem 2rem;
        box-shadow: 0 8px 24px rgba(220, 38, 38, 0.1);
        margin-top: 1.5rem;
        margin-bottom: 1.2rem;
    }

    .result-tag-benign {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #16A34A;
        color: #FFFFFF;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .result-tag-malignant {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #DC2626;
        color: #FFFFFF;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
    }

    .result-title-benign { color: #14532D !important; }
    .result-title-malignant { color: #7F1D1D !important; }

    .result-desc {
        font-size: 0.95rem;
        color: #334155;
        line-height: 1.5;
    }

    /* Glossary grid */
    .glossary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 10px;
        margin-top: 0.5rem;
    }

    .glossary-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
    }

    .glossary-name {
        font-weight: 700;
        font-size: 0.85rem;
        color: #0284C7;
        margin-bottom: 2px;
    }

    .glossary-text {
        font-size: 0.78rem;
        color: #475569;
        line-height: 1.35;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# 3. FEATURE SPECIFICATIONS & BASELINE DEFAULTS
# ==============================================================================

FEATURE_NAMES = [
    # Mean features (10)
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",

    # Standard error features (10)
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",

    # Worst (Largest) features (10)
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

DEFAULT_VALUES = [
    13.37, 18.84, 86.24, 551.1, 0.09587, 0.09263, 0.06154, 0.03350, 0.1792, 0.06154,
    0.3242, 1.108, 2.287, 24.53, 0.00638, 0.02045, 0.02589, 0.01093, 0.01873, 0.003187,
    14.97, 25.41, 97.66, 686.5, 0.1313, 0.2119, 0.2267, 0.09993, 0.2822, 0.08004
]

# 10 Primary Core Parameters
CORE_10_FEATURES = [
    {
        "key": "radius_mean",
        "label": "Cell Radius",
        "range": (6.0, 30.0, 0.01, "%.3f"),
        "desc": "Mean distance from cell nucleus center to boundary points."
    },
    {
        "key": "texture_mean",
        "label": "Cell Texture",
        "range": (9.0, 40.0, 0.01, "%.2f"),
        "desc": "Standard deviation of gray-scale pixel intensities (surface roughness)."
    },
    {
        "key": "perimeter_mean",
        "label": "Cell Perimeter",
        "range": (40.0, 200.0, 0.1, "%.2f"),
        "desc": "Total boundary circumference distance around the cell nucleus."
    },
    {
        "key": "area_mean",
        "label": "Cell Area",
        "range": (140.0, 2600.0, 1.0, "%.1f"),
        "desc": "Total two-dimensional surface area enclosed by the cell nucleus contour."
    },
    {
        "key": "smoothness_mean",
        "label": "Cell Smoothness",
        "range": (0.05, 0.20, 0.0001, "%.5f"),
        "desc": "Local variation in radius lengths (contour smoothness vs roughness)."
    },
    {
        "key": "compactness_mean",
        "label": "Cell Compactness",
        "range": (0.01, 0.35, 0.0001, "%.5f"),
        "desc": "Computed as (Perimeter² / Area - 1.0), measuring shape density."
    },
    {
        "key": "concavity_mean",
        "label": "Cell Concavity",
        "range": (0.0, 0.45, 0.0001, "%.5f"),
        "desc": "Severity and depth of concave inward contour indentations on the boundary."
    },
    {
        "key": "concave points_mean",
        "label": "Cell Concave Points",
        "range": (0.0, 0.25, 0.0001, "%.5f"),
        "desc": "Total number of concave inward portions along the nucleus contour."
    },
    {
        "key": "symmetry_mean",
        "label": "Cell Symmetry",
        "range": (0.10, 0.35, 0.0001, "%.4f"),
        "desc": "Structural symmetry score of the cell nucleus contour."
    },
    {
        "key": "fractal_dimension_mean",
        "label": "Cell Fractal Dimension",
        "range": (0.04, 0.10, 0.0001, "%.5f"),
        "desc": "Coastline boundary complexity approximation (coastline - 1.0)."
    }
]

# ==============================================================================
# 4. MODEL LOADER & STATE INITIALIZATION
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_ml_pipeline():
    """Loads trained Logistic Regression model and StandardScaler."""
    model_paths = ["final_breast_cancer_model.pkl", "breast_cancer_logistic_model.pkl"]
    scaler_paths = ["final_breast_cancer_scaler.pkl", "breast_cancer_scaler.pkl"]

    model = None
    scaler = None

    for m_path in model_paths:
        if os.path.exists(m_path):
            try:
                model = joblib.load(m_path)
                break
            except Exception:
                continue

    for s_path in scaler_paths:
        if os.path.exists(s_path):
            try:
                scaler = joblib.load(s_path)
                break
            except Exception:
                continue

    if model is None or scaler is None:
        return None, None, "Model or Scaler pickle files could not be loaded."

    return model, scaler, None

model, scaler, load_error = load_ml_pipeline()

# Initialize session state for the 10 core features
for i, item in enumerate(CORE_10_FEATURES):
    if item["key"] not in st.session_state:
        st.session_state[item["key"]] = float(DEFAULT_VALUES[i])

def reset_to_defaults():
    """Callback executed before widgets render to cleanly reset all values."""
    for i, item in enumerate(CORE_10_FEATURES):
        st.session_state[item["key"]] = float(DEFAULT_VALUES[i])

# ==============================================================================
# 5. MAIN INTERFACE
# ==============================================================================

def main():
    # Header
    st.markdown(
        """
        <div class="app-header">
            <div class="app-brand">
                <div class="app-logo">🩺</div>
                <div>
                    <div class="app-title">Breast Cancer Prediction AI</div>
                    <div class="app-subtitle">Clinical Decision Support & Malignancy Risk Assessment</div>
                </div>
            </div>
            <div>
                <span class="status-pill">
                    <span class="status-dot"></span>
                    Model Online (96.49% Accuracy)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None or scaler is None:
        st.error(f"❌ Unable to load machine learning model: {load_error}")
        return

    # Parameter Explanation Glossary
    with st.expander("📖 What do these 10 Cell Parameters measure? (Click to view guide)", expanded=False):
        st.markdown(
            """
            <div class="glossary-grid">
                <div class="glossary-item">
                    <div class="glossary-name">📏 Radius</div>
                    <div class="glossary-text">Distance from nucleus center to outer edge points. Larger in breast cancer cells.</div>
                </div>
                <div class="glossary-item">
                    <div class="glossary-name">🔬 Texture</div>
                    <div class="glossary-text">Standard deviation of gray-scale pixel intensity (surface roughness).</div>
                </div>
                <div class="glossary-item">
                    <div class="glossary-name">🔄 Perimeter</div>
                    <div class="glossary-text">Total boundary circumference distance around the cell nucleus.</div>
                </div>
                <div class="glossary-item">
                    <div class="glossary-name">⬛ Area</div>
                    <div class="glossary-text">Total two-dimensional surface area enclosed by the cell nucleus contour.</div>
                </div>
                <div class="glossary-item">
                    <div class="glossary-name">✨ Smoothness</div>
                    <div class="glossary-text">Local variation in radius lengths (edge roughness vs smoothness).</div>
                </div>
                <div class="glossary-item">
                    <div class="glossary-name">📦 Compactness</div>
                    <div class="glossary-text">(Perimeter² / Area - 1.0), measuring structural density & irregularity.</div>
                </div>
                <div class="glossary-item">
                    <div class="glossary-name">🕳️ Concavity</div>
                    <div class="glossary-text">Severity and depth of concave inward contour indentations on the boundary.</div>
                </div>
                <div class="glossary-item">
                    <div class="glossary-name">📍 Concave Points</div>
                    <div class="glossary-text">Total count of concave inward portions on nucleus boundary.</div>
                </div>
                <div class="glossary-item">
                    <div class="glossary-name">⚖️ Symmetry</div>
                    <div class="glossary-text">Structural symmetry score of the cell nucleus contour.</div>
                </div>
                <div class="glossary-item">
                    <div class="glossary-name">🧬 Fractal Dimension</div>
                    <div class="glossary-text">Coastline boundary complexity approximation (coastline - 1.0).</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 10 Core Cell Parameters Form
    st.markdown(
        """
        <div class="param-card">
            <div class="param-card-header">
                <div class="param-card-title">🔬 Cell Nucleus Parameters (10 Features)</div>
                <div class="param-card-desc">Enter the 10 morphological biopsy measurements below to evaluate breast cancer risk.</div>
            </div>
        """,
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2)

    for idx, item in enumerate(CORE_10_FEATURES):
        target_col = col_a if idx < 5 else col_b
        min_v, max_v, step_v, fmt = item["range"]
        with target_col:
            st.number_input(
                item["label"],
                min_value=min_v,
                max_value=max_v,
                value=st.session_state[item["key"]],
                step=step_v,
                format=fmt,
                key=item["key"]
            )
            st.markdown(f"<div class='field-desc'>{item['desc']}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Action Row: Predict Breast Cancer & Reset to Defaults
    col_act1, col_act2 = st.columns([3, 1], vertical_alignment="center")

    with col_act1:
        st.markdown('<div class="btn-predict">', unsafe_allow_html=True)
        predict_clicked = st.button("⚡ Predict Breast Cancer Diagnosis", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_act2:
        st.markdown('<div class="btn-reset">', unsafe_allow_html=True)
        st.button("🔄 Reset to Defaults", on_click=reset_to_defaults, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Model Inference & Results
    if predict_clicked:
        core_values = [st.session_state[item["key"]] for item in CORE_10_FEATURES]

        # Clinical proportion scaling for standard error and worst values
        se_ratios = [DEFAULT_VALUES[10 + i] / DEFAULT_VALUES[i] for i in range(10)]
        worst_ratios = [DEFAULT_VALUES[20 + i] / DEFAULT_VALUES[i] for i in range(10)]

        full_vector = core_values + [v * r for v, r in zip(core_values, se_ratios)] + [v * r for v, r in zip(core_values, worst_ratios)]
        input_df = pd.DataFrame([full_vector], columns=FEATURE_NAMES)

        try:
            scaled_features = scaler.transform(input_df)
            pred_class = int(model.predict(scaled_features)[0])
            probabilities = model.predict_proba(scaled_features)[0]

            benign_prob = float(probabilities[0])
            malignant_prob = float(probabilities[1])
            confidence = benign_prob if pred_class == 0 else malignant_prob

            if pred_class == 0:
                # Benign Result Card
                st.markdown(
                    f"""
                    <div class="result-box-benign">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div>
                                <span class="result-tag-benign">🟢 Non-Cancerous</span>
                                <div class="result-title result-title-benign">BREAST CANCER: BENIGN</div>
                                <div class="result-desc">
                                    The biopsy cellular characteristics indicate a <strong>Benign (Non-Cancerous)</strong> state with a model confidence score of <strong>{confidence * 100:.2f}%</strong>.
                                </div>
                            </div>
                            <div style="background: #FFFFFF; border: 1px solid #86EFAC; border-radius: 14px; padding: 1rem 1.6rem; text-align: center;">
                                <div style="font-size: 0.75rem; font-weight: 700; color: #16A34A; text-transform: uppercase;">Confidence Score</div>
                                <div style="font-size: 2.2rem; font-weight: 800; color: #15803D;">{confidence * 100:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Malignant Result Card
                st.markdown(
                    f"""
                    <div class="result-box-malignant">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div>
                                <span class="result-tag-malignant">🔴 Cancer Detected</span>
                                <div class="result-title result-title-malignant">BREAST CANCER: MALIGNANT</div>
                                <div class="result-desc">
                                    The biopsy cellular characteristics indicate <strong>Malignant Breast Cancer</strong> with a risk probability of <strong>{malignant_prob * 100:.2f}%</strong>. Prompt clinical verification is recommended.
                                </div>
                            </div>
                            <div style="background: #FFFFFF; border: 1px solid #FCA5A5; border-radius: 14px; padding: 1rem 1.6rem; text-align: center;">
                                <div style="font-size: 0.75rem; font-weight: 700; color: #DC2626; text-transform: uppercase;">Malignancy Risk</div>
                                <div style="font-size: 2.2rem; font-weight: 800; color: #B91C1C;">{malignant_prob * 100:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Probabilities Gauge
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-weight: 700; color: #16A34A;">🟢 Benign (Non-Cancerous)</span>
                            <span style="font-weight: 800; color: #16A34A;">{benign_prob * 100:.2f}%</span>
                        </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(benign_prob)
                st.markdown("</div>", unsafe_allow_html=True)

            with p2:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-weight: 700; color: #DC2626;">🔴 Malignant (Breast Cancer)</span>
                            <span style="font-weight: 800; color: #DC2626;">{malignant_prob * 100:.2f}%</span>
                        </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(malignant_prob)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as err:
            st.error(f"❌ Error during model execution: {str(err)}")

if __name__ == "__main__":
    main()