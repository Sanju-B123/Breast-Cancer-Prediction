# OncoPredict AI — Breast Cancer Prediction System 🩺

An AI-powered clinical decision support and tumor classification platform built with **Python**, **Scikit-Learn**, and **Streamlit**. The system predicts whether a breast tumor is **Benign or Malignant** using 30 cell morphology characteristics derived from Fine Needle Aspirate (FNA) biopsies.

---

## 🌟 Key Highlights

- **🎯 High Accuracy**: **96.49%** Test Accuracy & **99.60%** ROC-AUC with calibrated Logistic Regression.
- **🔬 30 Clinical Biomarkers**: Categorized into **Mean**, **Standard Error (SE)**, and **Worst (Extreme)** morphological features.
- **✨ Modern Healthcare UI**: Custom medical design system with Glassmorphic cards, soft gradients, responsive metrics, and interactive Plotly analytics.
- **⚡ Quick Case Presets**: One-click demo buttons for verified Benign and Malignant cases for instant portfolio demonstrations.
- **📊 In-Depth Model Diagnostics**: Confusion matrix heatmaps, feature importance coefficients, and mathematical breakdowns.

---

## 🧭 Application Architecture

1. **📊 Executive Dashboard & Validation**
   - Core KPI scorecards (96.49% Accuracy, 569 Cases, 30 Metrics, 99.60% ROC-AUC).
   - End-to-End Machine Learning Pipeline Architecture diagram.
   - Morphological grouping breakdown table.
   - Integrated Model Validation metrics & Confusion Matrix heatmap.

2. **🔬 Diagnostic Prediction Console**
   - 3-tier organized input interface (Mean, SE, Worst) with intuitive tabs (Size/Geometry, Texture/Contour, Symmetry/Fractal).
   - Quick one-click clinical benchmark presets (Benign Low Risk, Malignant High Risk, Median Baseline).
   - Real-time diagnostic risk assessment cards with confidence scoring and probability distributions.
   - Detailed technical diagnostics with scaled z-score vectors.

---

## 🛠️ Technology Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Logistic Regression, StandardScaler)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Interactive Visualizations**: [Plotly](https://plotly.com/python/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)

---

## 🚀 Quick Start & Local Development

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Sanju-B123/Breast-Cancer-Prediction.git
cd Breast-Cancer-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Deployment

### Deploy to Render
The repository is pre-configured with `render.yaml`, `Procfile`, and `.streamlit/config.toml` for seamless deployment:
- **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
- **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false`

### Deploy to Streamlit Community Cloud
1. Fork / push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Select your repository and set `app.py` as the main entry point.

---

## 📂 Project Structure

```
├── app.py                              # Main Streamlit web application & UI
├── final_breast_cancer_model.pkl       # Trained Logistic Regression model
├── final_breast_cancer_scaler.pkl      # Trained StandardScaler artifact
├── requirements.txt                    # Python dependencies
├── render.yaml                         # Render deployment configuration
├── Procfile                            # Web server start command
├── .streamlit/
│   └── config.toml                     # Streamlit server & theme configuration
├── hero.png                            # Hero branding image
├── workflow.png                        # Machine learning workflow diagram
├── performance.png                     # Model performance summary chart
└── README.md                           # Project documentation
```

---

## 📈 Model Performance Summary

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **96.49%** | Overall correct classifications across 114 test samples |
| **Precision** | **97.50%** | Minimizes false positive malignancy alarms |
| **Recall (Sensitivity)**| **92.86%** | High sensitivity for detecting malignant cases |
| **F1-Score** | **95.12%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **99.60%** | Exceptional discriminative power across threshold spectrum |

---

## ⚠️ Medical Disclaimer

*This application is developed strictly for **academic, research, and portfolio demonstration purposes**. It does not constitute medical advice or a clinical diagnosis. Always consult a qualified medical professional for health evaluations.*

---

