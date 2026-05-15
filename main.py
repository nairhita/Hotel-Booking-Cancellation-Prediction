import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import shap
import from sklearn.linear_model import LogisticRegression

# --- 1. Data Loading ---
# Dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
df = pd.read_csv('hotel_bookings.csv')

# --- 2. Data Cleaning & Preprocessing ---
# Handle missing values
df['children'] = df['children'].fillna(0)
df['country'] = df['country'].fillna('Unknown')
df = df.drop(['company', 'agent'], axis=1) # Too many missing values/uninformative

# Feature Engineering: Combining stay nights
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

# Filter out zero-guest entries (data errors)
df = df[df['adults'] + df['children'] + df['babies'] > 0]

# --- 3. Encoding ---
# High cardinality handling: Keep top 10 countries, group others as 'Other'
top_countries = df['country'].value_counts().nlargest(10).index
df['country'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')

# Label Encoding for categorical features
cat_cols = df.select_dtypes(include=['O']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# --- 4. Modeling ---
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# --- 5. Evaluation ---
preds = model.predict(X_test)
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
print(classification_report(y_test, preds))

# --- 6. SHAP Interpretability ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# Summary plot for the README
shap.summary_plot(shap_values, X_test, plot_type="bar")
