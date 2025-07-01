import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
import os
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# Drop unnecessary columns
data = data.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

# Encode the type column
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

# Prepare features and target
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train XGBoost model with class weights
model = xgb.XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),  # Handle class imbalance
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

# Fit the model
model.fit(
    X_train_balanced, 
    y_train_balanced
)

# Save the model
model_path = os.path.join(os.path.dirname(__file__), 'xgboost_model.json')
model.save_model(model_path)

print("Model trained and saved successfully!")
print("\nFeature importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}") 