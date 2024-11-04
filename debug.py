import pandas as pd
import numpy as np
import logging
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.impute import SimpleImputer

import torch

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.empty_cache()
print("GPU?", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("./logs/UCI_Data_Train_ClassWeights.log"),
                        logging.StreamHandler()
                    ])

# Load and preprocess data
data_path = '../data'
df_2022 = pd.read_csv(f"{data_path}/heart_disease_uci.csv")

# Setup the data dir
saved_dir = './saved_models_uci_classweights'
saved_figs = './figs_uci_classweights/'
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)
if not os.path.exists(saved_figs):
    os.makedirs(saved_figs)

##############################################################################
# Handling missing data
print("Before dropping NAN", df_2022.shape)
# Find missing data parts
print("Missing data", df_2022.isnull().sum()[df_2022.isnull().sum()>0].index.tolist())
print("Missing data vals", df_2022.isnull().sum()[df_2022.isnull().sum()>0].sort_values(ascending=False))

# # Columns to drop
# columns_to_drop = df_2022.isnull().sum()[df_2022.isnull().sum()>0].index.tolist()
# # Drop
# df_2022.drop(columns=columns_to_drop, inplace=True)
# print("Dropped columns:", columns_to_drop)

# df_2022 = df_2022.dropna()
# print("After dropping NAN", df_2022.shape)

# Convert target labels to binary classification
df_2022['num'] = df_2022['num'].replace({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

imputer = SimpleImputer(strategy='mean')  # Replace NaNs with the mean of the column

# Convert categorical to numeric
le = LabelEncoder()
for col in df_2022.columns:
    if df_2022[col].dtype == 'object' or df_2022[col].dtype == 'category':
        df_2022[col] = le.fit_transform(df_2022[col])

X = df_2022.drop('num', axis=1)
y = df_2022['num']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

# Impute missing values using mean strategy
imputer = SimpleImputer(strategy='mean')
# Fit the imputer on the training data and transform both training and test sets
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print("X Train", X_train.shape)
print("X Test", X_test.shape)
print("y train", y_train.shape)
print("y test", y_test.shape)

# Calculate Class Weights
def calculate_class_weights(y):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    return dict(zip(np.unique(y), class_weights))

class_weights = calculate_class_weights(y_train)
logging.info(f"Calculated class weights: {class_weights}")

    
# Model Performance Evaluation
def get_model_performance(model, param_grid, X_train, X_test, y_train, y_test):
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'auc': 'roc_auc'
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='f1', n_jobs=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    cv_scores = {metric: grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_] 
                 for metric in scoring.keys()}

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    test_scores = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }

    return cv_scores, test_scores, best_model


def save_model(model, name, save_dir=f'{saved_dir}'):
    os.makedirs(save_dir, exist_ok=True)
    # if isinstance(model, PyTorchClassifier):
    #     torch.save(model.model.state_dict(), os.path.join(save_dir, f"{name}_pytorch.pth"))
    #     # Save the PyTorchClassifier wrapper
    #     joblib.dump(model, os.path.join(save_dir, f"{name}_wrapper.joblib"))
    # else:
    joblib.dump(model, os.path.join(save_dir, f"{name}.joblib"))
    logging.info(f"Saved model: {name}")
    

def model_evaluation(classifier, x_test, y_test):
    cm = confusion_matrix(y_test, classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=labels, cmap='Greens', fmt='')
    plt.title(f'Confusion Matrix')
    plt.savefig(f'{saved_figs}/confusion_matrix_{str(classifier.__class__.__name__)}.png', dpi=300, bbox_inches='tight')

    report = classification_report(y_test, classifier.predict(x_test))
    logging.info(f"Classification Report:\n{report}")

# Define classifiers
classifiers = {
    'Logistic Regression': (LogisticRegression(max_iter=5000, class_weight=class_weights), 
                            {'C': [0.1, 1, 10], 'penalty': ['l2']}),
    
    'Decision Tree': (DecisionTreeClassifier(class_weight=class_weights), 
                      {'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]}),
    
    'Random Forest': (RandomForestClassifier(class_weight=class_weights), 
                      {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5, 10]}),
    
    'Gradient Boosting': (GradientBoostingClassifier(), 
                          {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}),
    
    'K-Nearest Neighbors': (KNeighborsClassifier(), 
                            {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
    
    'XGBoost': (XGBClassifier(eval_metric='logloss'), 
                {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5],
                 'scale_pos_weight': [class_weights[1] / class_weights[0]]}),
    
    # 'LightGBM': (LGBMClassifier(class_weight=class_weights), 
    #              {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'num_leaves': [31, 63, 127]}),
}

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform model training and evaluation
results = []
for name, (model, param_grid) in classifiers.items():
    print(f"\nTraining {name}...")
    cv_scores, test_scores, best_model = get_model_performance(model, param_grid, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save the best model
    save_model(best_model, name)
    
    results.append({
        'Model': name,
        'CV F1': cv_scores['f1'],
        'Test AUC': test_scores['auc'],
        'Test Accuracy': test_scores['accuracy'],
        'Test Precision': test_scores['precision'],
        'Test Recall': test_scores['recall'],
        'Test F1': test_scores['f1'],
        'Best Parameters': best_model.get_params()
    })
    
    logging.info(f"Best parameters for {name}: {best_model.get_params()}")
    logging.info(f"Cross-validation scores for {name}: {cv_scores}")
    logging.info(f"Test scores for {name}: {test_scores}")
    model_evaluation(best_model, X_test_scaled, y_test)

# Display results
results_df = pd.DataFrame(results)
logging.info("\nModel Performance Comparison:")
logging.info("\n" + results_df.to_string(index=False))