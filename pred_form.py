import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from collections import Counter
from imblearn.over_sampling import SMOTE
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ===============================
# 1. Load and preprocess dataset
# ===============================
df = pd.read_csv(r"C:\Users\keith\Documents\3RDYEAR3RDSEM\accitrack_project - 4th yr\data\combinedd.csv")

df['Date Reported'] = pd.to_datetime(df['Date Reported'], errors='coerce')
df = df.dropna(subset=['Date Reported'])

# Daily grouping
df['Day'] = df['Date Reported'].dt.date.astype(str)
df['Month'] = df['Date Reported'].dt.to_period('M').astype(str).str.lower()

# Accident count per day per barangay
daily_counts = df.groupby(['Barangay_Location', 'Day']).size().reset_index(name='Accident_Count')

# Define risk levels (more sensitive threshold)
def risk_level(count):
    if count == 0:
        return "Low"
    elif count == 1:
        return "Medium"
    else:  # 2+ accidents
        return "High"

daily_counts['Risk_Level'] = daily_counts['Accident_Count'].apply(risk_level)

# ===============================
# 2. Aggregate daily features
# ===============================
features = df.groupby(['Barangay_Location', 'Day']).agg({
    'Weather Conditions': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
    'Road Conditions': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
    'Traffic_Volume': 'sum',
    'Month': lambda x: x.iloc[0]
}).reset_index()

dataset = daily_counts.merge(features, on=['Barangay_Location', 'Day'], how='left')

# ===============================
# 3. Encode categorical features
# ===============================
categorical_cols = ['Barangay_Location', 'Weather Conditions', 'Road Conditions', 'Month']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = dataset[col].astype(str).str.strip().str.lower()
    le.fit(dataset[col])
    dataset[col] = le.transform(dataset[col])
    label_encoders[col] = le
    joblib.dump(le, f"{col}_encoder.pkl")

# Encode target
le_risk = LabelEncoder()
dataset['Risk_Level'] = le_risk.fit_transform(dataset['Risk_Level'])
joblib.dump(le_risk, "risk_level_encoder.pkl")

# ===============================
# 4. Train-test split
# ===============================
X = dataset.drop(columns=['Accident_Count', 'Risk_Level', 'Day'])
y = dataset['Risk_Level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

feature_order = list(X.columns)
joblib.dump(feature_order, "feature_order.pkl")

# ===============================
# 5. Handle class imbalance (SMOTE)
# ===============================
print("\n📊 Original Class Distribution:", Counter(y_train))
sm = SMOTE(random_state=42, k_neighbors=3)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("📊 After SMOTE:", Counter(y_train))

# ===============================
# 6. Hybrid scorer
# ===============================
def hybrid_score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return 0.5 * acc + 0.5 * f1   # give more weight to F1

scorer = make_scorer(hybrid_score, greater_is_better=True)

# ===============================
# 7. Hyperparameter tuning
# ===============================
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.5, 0.5),
    'scale_pos_weight': [1, 3, 5, 10]  # boost High class
}

xgb = XGBClassifier(eval_metric='mlogloss', random_state=42, verbosity=0)

search = RandomizedSearchCV(
    xgb, param_distributions=param_dist,
    n_iter=30, scoring=scorer, cv=3,
    verbose=1, n_jobs=-1, random_state=42
)

search.fit(X_train, y_train)

print("\n✅ Best Parameters:", search.best_params_)

# ===============================
# 8. Evaluate model
# ===============================
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\n🎯 Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_risk.classes_))

# ===============================
# 9. Save model
# ===============================
joblib.dump(best_model, "accident_risk_model.pkl")
print("\n💾 Model, encoders, and feature order saved successfully (per-day with SMOTE).")
