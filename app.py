import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer ML Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #f6f8fc;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, p, label {
        color: #1f2937 !important;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        color: #111827 !important;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #6b7280 !important;
        margin-bottom: 20px;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        text-align: center;
        min-height: 120px;
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280 !important;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #111827 !important;
    }


    /* ========================================================
       SECTION CARDS
       ======================================================== */

    .section-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-top: 20px;
        margin-bottom: 15px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.03);
    }

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #111827 !important;
        margin-bottom: 5px;
    }

    .section-description {
        font-size: 14px;
        color: #6b7280 !important;
        margin-bottom: 15px;
    }


    /* ========================================================
       RESULT
       ======================================================== */

    .result-title {
        font-size: 30px;
        font-weight: 800;
        color: #111827 !important;
    }

    .result-description {
        font-size: 16px;
        color: #4b5563 !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] * {
        color: #1f2937 !important;
    }


    /* ========================================================
       INPUTS
       ======================================================== */

    [data-testid="stWidgetLabel"] p {
        color: #374151 !important;
        font-weight: 600;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 10px;
        min-height: 48px;
        font-weight: 700;
        font-size: 16px;
    }


    /* ========================================================
       DISCLAIMER
       ======================================================== */

    .disclaimer {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 18px;
        border-radius: 10px;
        color: #7c2d12 !important;
        margin-top: 25px;
    }

    .disclaimer p {
        color: #7c2d12 !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #9ca3af !important;
        font-size: 13px;
        margin-top: 30px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

st.write("Loading model...")

@st.cache_resource
def load_models():
    try:
        model_path = "final_breast_cancer_model.pkl"
        scaler_path = "final_breast_cancer_scaler.pkl"
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.stop()
        
        if not os.path.exists(scaler_path):
            st.error(f"❌ Scaler file not found: {scaler_path}")
            st.stop()
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


try:
    model, scaler = load_models()
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# ============================================================
# FEATURE NAMES
# ============================================================

features = [

    # Mean features
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    # Standard error features
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",

    # Worst features
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]


# ============================================================
# DEMO SAMPLE
# ============================================================

demo_values = [
    11.41,
    10.82,
    73.34,
    403.3,
    0.09373,
    0.06685,
    0.03512,
    0.02623,
    0.1667,
    0.06113,

    0.1408,
    0.4607,
    1.103,
    10.5,
    0.00604,
    0.01529,
    0.01514,
    0.00646,
    0.01344,
    0.002206,

    12.82,
    15.97,
    83.74,
    510.5,
    0.1548,
    0.239,
    0.2102,
    0.08958,
    0.3016,
    0.08523
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🩺 Breast Cancer ML")

    st.caption("Final Year Machine Learning Project")

    st.divider()

    st.markdown("### 📌 Project Overview")

    st.write(
        """
        This application demonstrates a machine-learning
        classification system for breast tumor measurements.

        The trained Logistic Regression model classifies
        a sample as **Benign** or **Malignant**.
        """
    )

    st.divider()

    st.markdown("### 🤖 Final Model")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Features:** 30")
    st.write("**Training Samples:** 455")
    st.write("**Testing Samples:** 114")

    st.divider()

    st.markdown("### 📊 Performance")

    st.write("Accuracy: **96.49%**")
    st.write("Precision: **97.50%**")
    st.write("Recall: **92.86%**")
    st.write("F1 Score: **95.12%**")
    st.write("ROC-AUC: **99.60%**")

    st.divider()

    st.markdown("### 🧠 Technologies")

    st.write(
        """
        • Python  
        • Pandas  
        • NumPy  
        • Scikit-learn  
        • Joblib  
        • Streamlit
        """
    )

    st.divider()

    st.caption(
        "Educational project — not intended for clinical diagnosis."
    )


# ============================================================
# HERO SECTION
# ============================================================

hero_path = "hero.png"

if os.path.exists(hero_path):

    st.image(
        hero_path,
        use_container_width=True
    )

else:

    st.markdown(
        '<div class="hero-title">🩺 Breast Cancer Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Machine Learning based breast tumor classification'
        '</div>',
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="hero-title">
        Breast Cancer Prediction
    </div>

    <div class="hero-subtitle">
        An end-to-end machine learning application using
        Logistic Regression and 30 tumor characteristics.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

st.markdown("### 📊 Model Performance")

c1, c2, c3, c4, c5 = st.columns(5)


metrics = [
    ("Accuracy", "96.49%"),
    ("Precision", "97.50%"),
    ("Recall", "92.86%"),
    ("F1 Score", "95.12%"),
    ("ROC-AUC", "99.60%")
]

columns = [c1, c2, c3, c4, c5]

for col, (title, value) in zip(columns, metrics):

    with col:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# WORKFLOW IMAGE
# ============================================================

workflow_path = "workflow.png"

if os.path.exists(workflow_path):

    with st.expander("🔬 View Machine Learning Workflow"):

        st.image(
            workflow_path,
            use_container_width=True
        )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="section-title">
        🔬 Tumor Measurement Input
    </div>

    <div class="section-description">
        Enter the 30 measurements used by the trained model.
        The inputs are divided into three groups for easier understanding.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DEMO / RESET CONTROLS
# ============================================================

button_col1, button_col2, button_col3 = st.columns([1, 1, 4])

with button_col1:

    if st.button(
        "🧪 Demo Sample",
        use_container_width=True
    ):

        for feature, value in zip(features, demo_values):
            st.session_state[feature] = value

        st.rerun()


with button_col2:

    if st.button(
        "🔄 Reset",
        use_container_width=True
    ):

        for feature in features:

            if feature in st.session_state:
                del st.session_state[feature]

        st.rerun()


st.info(
    "💡 A demo sample is provided so you can test the application "
    "without manually entering all 30 measurements."
)


# ============================================================
# FORM
# ============================================================

with st.form("prediction_form"):

    # --------------------------------------------------------
    # MEAN FEATURES
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-card">

        <div class="section-title">
            📊 Mean Features
        </div>

        <div class="section-description">
            Average measurements describing the size, shape and
            characteristics of the tumor.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    mean_features = features[:10]

    mean_values = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(mean_features):

        default_value = demo_values[i]

        if feature not in st.session_state:
            st.session_state[feature] = default_value

        if i % 2 == 0:

            with col1:

                value = st.number_input(
                    feature,
                    key=feature,
                    format="%.5f",
                    help=f"Mean measurement: {feature}"
                )

        else:

            with col2:

                value = st.number_input(
                    feature,
                    key=feature,
                    format="%.5f",
                    help=f"Mean measurement: {feature}"
                )

        mean_values.append(value)


    # --------------------------------------------------------
    # SE FEATURES
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-card">

        <div class="section-title">
            📐 Standard Error Features
        </div>

        <div class="section-description">
            Standard error measurements describe the variability
            associated with the corresponding tumor measurements.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    se_features = features[10:20]

    se_values = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(se_features):

        feature_index = i + 10
        default_value = demo_values[feature_index]

        if feature not in st.session_state:
            st.session_state[feature] = default_value

        if i % 2 == 0:

            with col1:

                value = st.number_input(
                    feature,
                    key=feature,
                    format="%.5f",
                    help=f"Standard error measurement: {feature}"
                )

        else:

            with col2:

                value = st.number_input(
                    feature,
                    key=feature,
                    format="%.5f",
                    help=f"Standard error measurement: {feature}"
                )

        se_values.append(value)


    # --------------------------------------------------------
    # WORST FEATURES
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-card">

        <div class="section-title">
            🔬 Worst Features
        </div>

        <div class="section-description">
            Extreme or largest observed measurements associated
            with each tumor characteristic.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    worst_features = features[20:]

    worst_values = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(worst_features):

        feature_index = i + 20
        default_value = demo_values[feature_index]

        if feature not in st.session_state:
            st.session_state[feature] = default_value

        if i % 2 == 0:

            with col1:

                value = st.number_input(
                    feature,
                    key=feature,
                    format="%.5f",
                    help=f"Worst measurement: {feature}"
                )

        else:

            with col2:

                value = st.number_input(
                    feature,
                    key=feature,
                    format="%.5f",
                    help=f"Worst measurement: {feature}"
                )

        worst_values.append(value)


    st.markdown("")

    submitted = st.form_submit_button(
        "🔍 Predict Diagnosis",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    all_values = (
        mean_values +
        se_values +
        worst_values
    )

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [all_values],
        columns=features
    )


    # --------------------------------------------------------
    # SCALE INPUT
    # --------------------------------------------------------

    input_scaled = scaler.transform(input_data)


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    benign_probability = probabilities[0]

    malignant_probability = probabilities[1]


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="result-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.write("")


    result_col1, result_col2 = st.columns([1, 1.5])


    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    with result_col1:

        if prediction == 0:

            st.success(
                "## 🟢 BENIGN"
            )

            st.write(
                "The model classified this sample as **Benign**."
            )

        else:

            st.error(
                "## 🔴 MALIGNANT"
            )

            st.write(
                "The model classified this sample as **Malignant**."
            )


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    with result_col2:

        st.markdown("### Prediction Probabilities")

        st.write(
            f"**🟢 Benign — {benign_probability * 100:.2f}%**"
        )

        st.progress(
            float(benign_probability)
        )

        st.write(
            f"**🔴 Malignant — {malignant_probability * 100:.2f}%**"
        )

        st.progress(
            float(malignant_probability)
        )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.markdown("### 🧠 Model Interpretation")

    if prediction == 0:

        st.info(
            f"""
            The Logistic Regression model predicts **Benign**
            with a probability of **{benign_probability * 100:.2f}%**.

            This prediction is based on the 30 tumor measurements
            supplied to the model.
            """
        )

    else:

        st.warning(
            f"""
            The Logistic Regression model predicts **Malignant**
            with a probability of **{malignant_probability * 100:.2f}%**.

            This prediction is based on the 30 tumor measurements
            supplied to the model.
            """
        )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander("🔎 View Technical Details"):

        st.markdown("#### Input Shape")

        st.code(
            f"Input shape: {input_data.shape}"
        )

        st.markdown("#### Model")

        st.code(
            "Logistic Regression"
        )

        st.markdown("#### Prediction Class")

        st.code(
            str(prediction)
        )

        st.markdown("#### Probabilities")

        probability_df = pd.DataFrame({
            "Diagnosis": [
                "Benign",
                "Malignant"
            ],
            "Probability": [
                benign_probability,
                malignant_probability
            ]
        })

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL PERFORMANCE IMAGE
# ============================================================

performance_path = "performance.png"

if os.path.exists(performance_path):

    st.markdown("---")

    with st.expander("📈 View Model Evaluation Summary"):

        st.image(
            performance_path,
            use_container_width=True
        )


# ============================================================
# ABOUT MODEL
# ============================================================

st.markdown("---")

st.markdown("## 🤖 About the Machine Learning Model")

about_col1, about_col2 = st.columns(2)

with about_col1:

    st.markdown("### Why Logistic Regression?")

    st.write(
        """
        Logistic Regression was selected as the final model because
        it performed extremely well on the test dataset.

        It is also a useful classification algorithm because it
        produces class probabilities, making it suitable for this
        demonstration.
        """
    )


with about_col2:

    st.markdown("### Why StandardScaler?")

    st.write(
        """
        The dataset contains features with very different numerical
        ranges. StandardScaler transforms the features to a comparable
        scale.

        The same scaler fitted during training is used to transform
        new input before prediction.
        """
    )


# ============================================================
# FEATURE GROUP EXPLANATION
# ============================================================

st.markdown("## 📚 Understanding the Features")

feature_info = pd.DataFrame({
    "Feature Group": [
        "Mean Features",
        "Standard Error Features",
        "Worst Features"
    ],

    "Purpose": [
        "Average tumor measurements",
        "Measurement variability",
        "Largest / most extreme measurements"
    ],

    "Examples": [
        "radius_mean, area_mean, texture_mean",
        "radius_se, area_se, texture_se",
        "radius_worst, area_worst, texture_worst"
    ]
})

st.dataframe(
    feature_info,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

    <b>⚠️ Important Disclaimer</b>

    <p>
    This application is an educational machine-learning project
    developed to demonstrate data preprocessing, model training,
    evaluation and deployment using Streamlit.
    </p>

    <p>
    It is <b>not a medical diagnostic system</b> and should not
    be used to make clinical or medical decisions.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Breast Cancer Classification • Machine Learning Final Year Project
        <br>
        Python • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)