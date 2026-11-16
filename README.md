# PhishGuard – Explainable and Robust Machine Learning Framework for Phishing URL Detection

## Project Overview
PhishGuard is a machine learning-based framework designed to identify phishing and legitimate URLs. It uses URL-based feature extraction and an XGBoost classification model to detect potentially malicious URLs.

The framework also provides explainability through SHAP (SHapley Additive exPlanations), allowing users to understand which URL characteristics influence a prediction.

## Key Features
- Real-time URL scanning and classification
- Phishing prediction with confidence scores
- Extraction and display of URL characteristics
- SHAP-based model explainability
- Interactive URL-to-feature-to-prediction graph
- Dataset exploration and visual analytics
- Model comparison and threshold analysis

## Dataset
The project uses the PhiUSIIL Phishing URL Dataset.

- Total URLs: 235,370
- Unique domains: 175,509
- Phishing URLs: 100,520
- URL features: 17

## Machine Learning
- Logistic Regression
- Random Forest
- XGBoost (selected model)
- SHAP for explainability

## Technology Stack
- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Plotly

## Project Structure
```text
PhishGuard_AIML/
├── app.py
├── requirements.txt
├── README.md
├── models/
├── src/
└── artifacts/
```

## Run Locally

1. Clone the repository:
```bash
git clone https://github.com/akankshabhuyan-tech/PhishGuard_AIML.git
```

2. Navigate to the project directory:
```bash
cd PhishGuard_AIML
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch the application:
```bash
streamlit run app.py
```

## Limitations
PhishGuard uses URL-based features for classification. It does not currently inspect webpage HTML, DNS records, TLS certificates, or live domain reputation.

## Project Purpose
This project demonstrates an explainable machine learning approach to phishing URL detection, combining predictive performance with interpretable model outputs and an interactive user interface.
