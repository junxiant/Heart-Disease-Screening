import pandas as pd
import numpy as np
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


seed = 42
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("CSML_2020_Data_Train.log"),
                        logging.StreamHandler()
                    ])

data_path = '../data'

df_2022 = pd.read_csv(f"{data_path}/2022/heart_2022_with_nans.csv")
df_2020 = pd.read_csv(f"{data_path}/2020/heart_2020_cleaned.csv")

# 1. Data distribution for the 2020 dataset
print(df_2020['HeartDisease'].value_counts())

plt.figure(figsize=(8, 6))
df_2020['HeartDisease'].value_counts().plot(kind='bar')
plt.title('Train Dataset Distribution')
plt.xlabel('HeartDisease')
plt.ylabel('Count')
plt.ticklabel_format(style='plain', axis='y')
# plt.show()
plt.savefig('./figs/Heart_Disease_Data_Distribution_Before_Split.png')

# Analyze categorial column
categorical_vars = ['Smoking', 'AlcoholDrinking', 'Stroke', 'Diabetic']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Distribution of Key Categorical Variables')

for i, var in enumerate(categorical_vars):
    ax = axes[i // 2, i % 2]
    df_2020[var].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(var)
    ax.set_ylabel('Count')
    
plt.tight_layout()
# plt.show()
plt.savefig('./figs/Categorical_Analysis.png')

# Analyze Age column
plt.figure(figsize=(12, 6))
df_2020['AgeCategory'].value_counts().sort_index().plot(kind='bar')
plt.title('Age Distribution')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
# plt.show()
plt.savefig('./figs/Age_Analysis.png')

# Analyze BMI
plt.figure(figsize=(10, 6))
sns.histplot(df_2020['BMI'], kde=True)
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Count')
# plt.show()
plt.savefig('./figs/BMI_Analysis.png')

# 2. Feature importance analysis

# Convert categorical to numeric
le = LabelEncoder()
for column in df_2020.select_dtypes(include=['object']).columns:
    df_2020[column] = le.fit_transform(df_2020[column])

X = df_2020.drop('HeartDisease', axis=1)
y = df_2020['HeartDisease']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

print("X Train", X_train.shape)
print("X Test", X_test.shape)
print("y rain", y_train.shape)
print("y test", y_test.shape)

# After splitting
# Distribution after split
train_distribution = Counter(y_train)
test_distribution = Counter(y_test)

print("Train set distribution:")
for label, count in train_distribution.items():
    percentage = count / len(y_train) * 100
    print(f"Class {label}: {count} ({percentage:.2f}%)")

print("\nTest set distribution:")
for label, count in test_distribution.items():
    percentage = count / len(y_test) * 100
    print(f"Class {label}: {count} ({percentage:.2f}%)")

print(f"\nTotal samples in train set: {len(y_train)}")
print(f"Total samples in test set: {len(y_test)}")

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
plt.savefig('./figs/Correlation_Bar.png')

print("Ranked features by correlation with target:")
print(correlation_with_target)

# Correlation heatmap including the target variable
plt.figure(figsize=(16, 14))
cor = df_2020.corr()
sns.heatmap(cor, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Features (including Heart Disease)")
plt.tight_layout()
# plt.show()
plt.savefig('./figs/Correlation_Matrix_Heatmap.png')


# Using RF classifier first
rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
rf_model.fit(X_train, y_train)

# Extract features ranking
importances = rf_model.feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
feature_importances.plot(kind='bar')
plt.title('Feature Importances for Heart Disease Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
# plt.show()
plt.savefig('./figs/Feature_Importance_RF.png')

print("Ranked important features:")
print(feature_importances)