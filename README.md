# Heart Disease Screening

A machine learning and deep learning benchmarking pipeline for cardiovascular disease risk prediction and screening across large-scale public health cohorts (CDC BRFSS 2020, CDC BRFSS 2022, and UCI Heart Disease datasets).

---

## Overview

Cardiovascular diseases are among the leading global causes of mortality. Early risk stratification through standard health indicators enables timely clinical intervention. This project systematically trains, tunes, and benchmarks classical machine learning algorithms alongside PyTorch deep neural networks across multiple dataset cohorts, evaluating the impact of class imbalance handling (class weighting) on clinical diagnostic metrics (Recall, Precision, ROC-AUC).

## Datasets Benchmarked

1. **CDC BRFSS 2020** (~319k clinical survey records)
2. **CDC BRFSS 2022** (Updated behavioral risk factor surveillance data)
3. **UCI Heart Disease Dataset** (Clinical biomarker and physiological diagnostic data)

## Models & Algorithms Evaluated

- **Ensemble Methods**: Random Forest, Gradient Boosting, XGBoost
- **Linear & Tree Baselines**: Logistic Regression, Decision Trees, K-Nearest Neighbors (KNN)
- **Deep Learning**: Multi-Layer Perceptron (MLP) Neural Networks implemented in PyTorch
- **Optimization**: Hyperparameter tuning via GridSearchCV, automated data scaling, and class-weighted loss functions.

## Tech Stack

- **Deep Learning**: PyTorch (`torch.nn`, `torch.optim`, `DataLoader`)
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib
- **Data Analysis & Preprocessing**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn


## Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/junxiant/Heart-Disease-Screening.git
cd Heart-Disease-Screening

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```
## Usage
```
# Run exploratory data analysis
python 0-Data_Analysis_UCI.py
# Train models with class weighting
python 3-Model_Train_UCI_Data_ClassWeights.py
# Run batch training across all cohorts
bash train.sh
# Run inference on sample clinical features
python 4-Inference_NoTune.py
```

Author Jun Tan

Linkedin https://www.linkedin.com/in/junxiant/
