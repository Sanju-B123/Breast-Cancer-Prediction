# Breast Cancer Prediction 🩺

A machine learning web application for breast tumor classification as **Benign** or **Malignant** using Logistic Regression.

## Features

- 🤖 **ML Model**: Logistic Regression with 96.49% accuracy
- 📊 **30 Features**: Comprehensive tumor measurements
- 📈 **High Performance**: 97.50% precision, 99.60% ROC-AUC
- 🎨 **Interactive UI**: Built with Streamlit
- ⚡ **Fast Inference**: Real-time predictions

## Deployment on Render

### Prerequisites
- GitHub account with the repository pushed
- Render account (free tier available at https://render.com)

### Steps to Deploy

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Connect to Render**
   - Go to [https://dashboard.render.com](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub account and select this repository

3. **Configure the Service**
   - **Name**: `breast-cancer-predictor` (or your preferred name)
   - **Environment**: `Python`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Build Command**: Leave as default or ensure it runs `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py`
   - **Instance Type**: Free tier (adequate for this app)

4. **Environment Variables** (Auto-configured in render.yaml)
   - `STREAMLIT_SERVER_PORT`: 10000
   - `STREAMLIT_SERVER_ADDRESS`: 0.0.0.0
   - `STREAMLIT_SERVER_HEADLESS`: true

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete (2-5 minutes)
   - Access your app at the provided Render URL

### Alternative: Using render.yaml (One-Click Deploy)
If you have the `render.yaml` file committed:
1. Go to [https://dashboard.render.com](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Select "Deploy with render.yaml" option (if available)
4. Render will auto-configure based on the YAML file

## Local Development

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Project Structure

```
├── app.py                              # Main Streamlit application
├── final_breast_cancer_model.pkl       # Trained Logistic Regression model
├── final_breast_cancer_scaler.pkl      # Feature scaler
├── requirements.txt                    # Python dependencies
├── render.yaml                         # Render deployment config
├── .streamlit/config.toml              # Streamlit configuration
├── hero.png                            # Hero image
├── workflow.png                        # ML workflow diagram
└── README.md                           # This file
```

## Model Details

- **Algorithm**: Logistic Regression
- **Features**: 30 tumor characteristics
- **Training Samples**: 455
- **Testing Samples**: 114
- **Accuracy**: 96.49%
- **Precision**: 97.50%
- **Recall**: 92.86%
- **F1 Score**: 95.12%
- **ROC-AUC**: 99.60%

## Technology Stack

- **Framework**: Streamlit
- **ML Library**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Model Serialization**: Joblib
- **Deployment**: Render

## Important Disclaimer

⚠️ **This is an educational project and NOT intended for clinical diagnosis or medical decision-making. Always consult with qualified healthcare professionals for medical advice.**

## License

This project is open source and available under the MIT License.
