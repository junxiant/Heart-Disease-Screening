# Heart Disease Screening

An artificial intelligence and Large Language Model-driven clinical screening application for cardiovascular risk stratification and patient record assessment.

---

## Overview

Cardiovascular diseases are the leading cause of mortality globally. This application integrates classical supervised machine learning for structured physiological risk indicators (blood pressure, cholesterol, ECG features) with LLMs to interpret unstructured patient clinical notes and summarize cardiovascular risk profiles for healthcare providers.

## Key Capabilities

- Tabular Risk Modeling: Ensembles (XGBoost, LightGBM, Random Forests) trained on clinical biomarker data.
- Clinical Text Interpretation: LLM prompt pipelines parsing unstructured discharge summaries and symptoms.
- Risk Score Generation: Actionable stratification categorization with interpretable feature attributions (SHAP values).

## Tech Stack

- ML Frameworks: Scikit-Learn, XGBoost, LightGBM, SHAP
- LLM / NLP: LangChain, OpenAI API, Hugging Face Transformers
- Backend / UI: Python, Flask, Streamlit

## Setup & Running

```bash
git clone https://github.com/junxiant/Heart-Disease-Screening.git
cd Heart-Disease-Screening

pip install -r requirements.txt

# Launch application
streamlit run app.py
```

Author Jun Tan

Linkedin https://www.linkedin.com/in/junxiant/
