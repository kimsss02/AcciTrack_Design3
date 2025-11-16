import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\keith\Documents\3RDYEAR3RDSEM\accitrack_project - 4th yr\data\Combined1.csv")
print(df.head())
df

plt.figure(figsize=(20, 8))
plt.title("Top 10 Barangay with the Highest Number of Accidents")

# Get the counts for the top 10 barangays
state_counts = df["Barangay"].value_counts()

# Create the bar plot
sns.barplot(x=state_counts[:10].values, y=state_counts[:10].index, orient="h")

# Set labels
plt.xlabel("Number of Accidents")
plt.ylabel("Barangay")

# Show the plot
plt.show()

# Count occurrences of each weather condition
data = df["Weather_Conditions"].value_counts().reset_index()
data.columns = ['Weather_Conditions', 'Count']

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Weather_Conditions', data=data, orient='h', palette="Set2")
plt.title("Most Frequent Weather Conditions")
plt.xlabel("Count")
plt.ylabel("Weather Condition")
plt.tight_layout()
plt.show()


# Road conditions
counts = df["Road_Conditions"].value_counts()[:3]

plt.figure(figsize=(20, 8))
plt.title("Histogram distribution of the top 3 road conditions")  # Also fixed the title
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel("Road Condition")
plt.ylabel("Value")
plt.show()

# Step 1: Convert to datetime (handle errors)
df['Date_Reported'] = pd.to_datetime(df['Date_Reported'], errors='coerce')

# Optional: Check how many couldn't be converted
print("Unparsable dates:", df['Date_Reported'].isna().sum())

# Optional: Drop rows where date parsing failed
df = df.dropna(subset=['Date_Reported'])

# Step 2: Create a simplified month-year column like 'Jan 2016'
df['Month_Reported'] = df['Date_Reported'].dt.strftime('%b %Y')

# Step 3: Count occurrences by month
counts = df['Month_Reported'].value_counts().sort_index()

# Step 4: Plot
plt.figure(figsize=(20, 8))
plt.title("Number of Accidents by Month Reported")
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel("Month")
plt.ylabel("Number of Reports")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

features = ["Type_of_Place", "Case_Status"]
df = df.drop(features, axis=1)
df


# Drop duplicates
print("Number of rows:", len(df.index))
df.drop_duplicates(inplace=True)
print("Number of rows after drop of duplicates:", len(df.index))


# # Correlation
# # Label Encoding
df.columns
l=LabelEncoder()

x = df.copy()  
x_encoded = x.copy()

for col in x_encoded.columns:
    if x_encoded[col].dtype == 'object':
        x_encoded[col] = LabelEncoder().fit_transform(x_encoded[col])

v=l.fit_transform(df["Traffic_Volume"])
b=l.fit_transform(df["Barangay"])
r=l.fit_transform(df["Road_Conditions"])
w=l.fit_transform(df["Weather_Conditions"])
d=l.fit_transform(df["Date_Reported"])

df["Traffic_Volume"]=v
df["Barangay"]=b
df["Road_Conditions"]=r
df["Weather_Conditions"]=w
df["Date_Reported"]=d

df

x = df.iloc[:500000,1:]
y = df.iloc[:500000,:1]

x
y


# Train, Test data splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,train_size=0.7,random_state=0)

# Scaling
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Make a copy to avoid changing the original
x_encoded = x.copy()

# Encode categorical columns
for col in x_encoded.columns:
    if x_encoded[col].dtype == 'object':
        x_encoded[col] = LabelEncoder().fit_transform(x_encoded[col])

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.3, train_size=0.7, random_state=0)

# Scale the data
scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)


# Model Building
accuracy = dict()


# XGBClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your dataset
data = pd.read_csv(r"C:\Users\keith\Documents\3RDYEAR3RDSEM\accitrack_project - 4th yr\data\Combined1.csv")

# Define target and features
target_column = 'Details_of_Main_Cause'
y = data[target_column]
x = data.drop(columns=[target_column, 'Date_Reported', 'Time_Committed', 'Barangay', 'Case_Status'])

# Encode categorical features
label_encoders = {}
for column in x.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    x[column] = le.fit_transform(x[column].astype(str))  # Cast to string to avoid errors
    label_encoders[column] = le

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.3, random_state=42)

# Define and train XGBoost model
model = XGBClassifier(objective='multi:softmax', learning_rate=0.1, max_depth=5, n_estimators=100, n_jobs=-1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)

accuracy['XBG Classifier']=accuracy_score(y_test, y_pred)

y_pred1 = model.predict(x_train)
accuracy_score(y_train, y_pred1)
print(classification_report(y_train,y_pred1))

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Select categorical columns
categorical_cols = x_train.select_dtypes(include=['object']).columns.tolist()

# Create ColumnTransformer with OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep non-categorical columns
)

# Transform the training and testing sets
x_train_encoded = preprocessor.fit_transform(x_train)
x_test_encoded = preprocessor.transform(x_test)

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(x_train_encoded, y_train)

y_pred = model.predict(x_test_encoded)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Decision Tree
DT = DecisionTreeClassifier(random_state=100)
params = [{"criterion": ["gini", "entropy"], "max_depth": [5, 10, 15, 30]}]
grid1 = GridSearchCV(DT, params, n_jobs=-1)
grid1.fit(x_train, y_train)

grid1.best_params_

DT1=DecisionTreeClassifier(criterion='entropy', max_depth=30)
DT1.fit(x_train, y_train)

y_pred4 = DT1.predict(x_train)

accuracy_score(y_train, y_pred4)

print(classification_report(y_train, y_pred4))

y_pred5 = DT1.predict(x_test)
accuracy_score(y_test, y_pred5)

accuracy['Decision Tree']=accuracy_score(y_test, y_pred5)

print(classification_report(y_test, y_pred5))


# Random Forest
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',max_depth=30,n_jobs=-1)
classifier.fit(x_train,y_train)

y_pred6 = classifier.predict(x_train)
accuracy_score(y_train, y_pred6)
print(classification_report(y_train, y_pred6))

y_pred7 = classifier.predict(x_test)
accuracy_score(y_test, y_pred7)
print(classification_report(y_test, y_pred7))

accuracy['Random Forest']=accuracy_score(y_test, y_pred7)
accuracy

plt.figure(figsize=(20, 8))
plt.title("Accuracy on Validation Set for Each Model")

sns.barplot(x=list(accuracy.keys()), y=list(accuracy.values()))
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
plt.show()
