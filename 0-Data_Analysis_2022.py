import pandas as pd
import numpy as np
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import math

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("./figs/2022_Data_Analysis.log"),
                        logging.StreamHandler()
                    ])

seed = 42

data_path = '../data'

df_2022 = pd.read_csv(f"{data_path}/2022/heart_2022_no_nans.csv")
print("Columns", df_2022.columns)
# Check data types of each column
print("Columns and their data types:")
print(df_2022.dtypes)

# Separate categorical and integer columns
categorical_cols = df_2022.select_dtypes(include=['object', 'category']).columns.tolist()

# For categorical-like columns with a limited number of unique values
for col in df_2022.select_dtypes(include=['int64', 'float64']).columns:
    if df_2022[col].nunique() < 20:  # Threshold for treating a numeric column as categorical
        categorical_cols.append(col)

integer_cols = df_2022.select_dtypes(include=['int64']).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Integer columns:", integer_cols)


# 1. Data distribution for the 2022 dataset
print(df_2022['HadHeartAttack'].value_counts())

plt.figure(figsize=(8, 6))
df_2022['HadHeartAttack'].value_counts().plot(kind='bar')
plt.title('Train Dataset Distribution')
plt.xlabel('HadHeartAttack')
plt.ylabel('Count')
plt.ticklabel_format(style='plain', axis='y')
# plt.show()
plt.savefig('./figs/2022_Heart_Disease_Data_Distribution_Before_Split.png')

#Categorical columns: ['State', 'Sex', 'GeneralHealth', 'LastCheckupTime', 'PhysicalActivities', 'RemovedTeeth', 
# 'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 
# 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes',
# 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 
# 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus', 
# 'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory', 'AlcoholDrinkers', 
# 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos']


# Analyze categorial column
categorical_vars = ['State', 'Sex', 'GeneralHealth', 'LastCheckupTime', 'PhysicalActivities', 'RemovedTeeth', 
'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 
'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes',
'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 
'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus', 
'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory', 'AlcoholDrinkers', 
'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos']

# Calculate the number of rows and columns based on the number of variables
n_vars = len(categorical_vars)
n_cols = 2  # You want 2 columns
n_rows = math.ceil(n_vars / n_cols)  # Calculate the number of rows needed

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
fig.suptitle('Distribution of Key Categorical Variables')

# Flatten axes if there is more than one row or column
axes = axes.flatten()

for i, var in enumerate(categorical_vars):
    ax = axes[i]
    df_2022[var].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(var)
    ax.set_ylabel('Count')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('./figs/2022_Categorical_Analysis.png')

# Analyze Age column
plt.figure(figsize=(12, 6))
df_2022['AgeCategory'].value_counts().sort_index().plot(kind='bar')
plt.title('Age Distribution')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
# plt.show()
plt.savefig('./figs/2022_Age_Analysis.png')

# Analyze BMI
plt.figure(figsize=(10, 6))
sns.histplot(df_2022['BMI'], kde=True)
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Count')
# plt.show()
plt.savefig('./figs/2022_BMI_Analysis.png')

# 2. Feature importance analysis

# Convert categorical to numeric
le = LabelEncoder()
for column in df_2022.select_dtypes(include=['object']).columns:
    df_2022[column] = le.fit_transform(df_2022[column])

X = df_2022.drop('HadHeartAttack', axis=1)
y = df_2022['HadHeartAttack']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

logging.info(f"X Train: {X_train.shape}")
logging.info(f"X Test: {X_test.shape}")
logging.info(f"y Train: {y_train.shape}")
logging.info(f"y Test: {y_test.shape}")

# After splitting
# Distribution after split
train_distribution = Counter(y_train)
test_distribution = Counter(y_test)

logging.info(f"Train set distribution:")
for label, count in train_distribution.items():
    percentage = count / len(y_train) * 100
    logging.info(f"Class {label}: {count} ({percentage:.2f}%)")

logging.info("\nTest set distribution:")
for label, count in test_distribution.items():
    percentage = count / len(y_test) * 100
    logging.info(f"Class {label}: {count} ({percentage:.2f}%)")

logging.info(f"\nTotal samples in train set: {len(y_train)}")
logging.info(f"Total samples in test set: {len(y_test)}")

# Correlation matrix
# Combine first
train_data = X_train.copy()
train_data['HeartDisease'] = y_train

# Do corr with label
correlation_with_target = train_data.drop('HeartDisease', axis=1).apply(lambda x: x.corr(train_data['HeartDisease'])).abs().sort_values(ascending=False)

# Visualize correlations with target
plt.figure(figsize=(12, 8))
correlation_with_target.plot(kind='bar')
plt.title('Feature Correlation with Heart Disease')
plt.xlabel('Features')
plt.ylabel('Absolute Correlation')
plt.tight_layout()
# plt.show()
plt.savefig('./figs/2022_Correlation_Bar.png')

logging.info("Ranked features by correlation with target with Corr:")
logging.info(correlation_with_target)

# Correlation heatmap including the target variable
plt.figure(figsize=(16, 14))
cor = df_2022.corr()
sns.heatmap(cor, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Features (including Heart Disease)")
plt.tight_layout()
# plt.show()
plt.savefig('./figs/2022_Correlation_Matrix_Heatmap.png')

# Using RF classifier first
rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
rf_model.fit(X_train, y_train)

# Extract features ranking
importances = rf_model.feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
feature_importances.plot(kind='bar')
plt.title('Feature Importances for Heart Disease Prediction with RF')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
# plt.show()
plt.savefig('./figs/2022_Feature_Importance_RF.png')

logging.info("Ranked important features:")
logging.info(feature_importances)
