# ============================================================
# Customer Churn Prediction Project (Command Line Version)
# ============================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run_pipeline():
    print("1. Loading dataset...")
    data = pd.read_csv("Churn_Modelling.csv")
    
    print("2. Preprocessing data...")
    le = LabelEncoder()
    data["Geography"] = le.fit_transform(data["Geography"])
    data["Gender"] = le.fit_transform(data["Gender"])
    data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    
    # Split into features (X) and target (y)
    X = data.drop("Exited", axis=1)
    y = data["Exited"]
    
    # Train-Test Split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    print("3. Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    print("4. Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("=" * 30)
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print("=" * 30)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    run_pipeline()
