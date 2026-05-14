# ============================================================
# Customer Churn Prediction Project
# Dataset: Maven Analytics Customer Churn Dataset
# Goal: Predict if a customer will churn (leave) or stay
# ============================================================

# --- Step 2: Load Dataset ---

# Import the pandas library to work with data
import pandas as pd

# Load the CSV dataset into a DataFrame (like a table)
data = pd.read_csv("Churn_Modelling.csv")

# Show the first 5 rows of the dataset
print("=== First 5 Rows ===")
print(data.head())

# Show how many rows and columns the dataset has
print("\n=== Dataset Shape (rows, columns) ===")
print(data.shape)

# Show all column names
print("\n=== Column Names ===")
print(data.columns.tolist())

# Show data types and count of non-null values per column
print("\n=== Dataset Info ===")
print(data.info())

# --- Step 3: Data Cleaning ---
print("\n" + "="*30)
print("Step 3: Data Cleaning")
print("="*30)

# 1. Check for missing (null) values in each column
print("\n=== Missing Values Before Cleaning ===")
print(data.isnull().sum())

# 2. Remove any rows that have missing values
# (dropna() removes rows with any missing data)
data = data.dropna()

# 3. Remove duplicate rows if there are any
# (drop_duplicates() removes exact duplicate rows)
data = data.drop_duplicates()

print("\n=== Dataset Shape After Cleaning ===")
print(data.shape)

# --- Step 4: Data Visualization ---
print("\n" + "="*30)
print("Step 4: Data Visualization")
print("="*30)

# Import matplotlib for creating graphs
import matplotlib.pyplot as plt

# ---- Graph 1: Churn Count Plot ----
# Count how many customers stayed (0) and churned (1)
churn_counts = data["Exited"].value_counts()

plt.figure(figsize=(6, 4))                          # Set figure size
plt.bar(["Stayed (0)", "Churned (1)"],              # X-axis labels
        churn_counts.values,                         # Bar heights
        color=["steelblue", "tomato"])               # Bar colors
plt.title("Customer Churn Count")                   # Chart title
plt.xlabel("Churn Status")                          # X-axis label
plt.ylabel("Number of Customers")                   # Y-axis label
plt.tight_layout()
plt.savefig("churn_count_plot.png")                 # Save chart as image
plt.close()                                         # Close chart (no popup)
print("Chart 1 saved: churn_count_plot.png")

# ---- Graph 2: Balance Histogram ----
# Shows how customer account balances are spread out
plt.figure(figsize=(6, 4))
plt.hist(data["Balance"],                           # Column to plot
         bins=30,                                    # Number of bars
         color="mediumpurple",                       # Bar color
         edgecolor="white")                          # Bar border color
plt.title("Distribution of Customer Balance")       # Chart title
plt.xlabel("Balance")                               # X-axis label
plt.ylabel("Number of Customers")                   # Y-axis label
plt.tight_layout()
plt.savefig("balance_histogram.png")                # Save chart as image
plt.close("all")                                    # Close all chart windows (no popup)
print("Chart 2 saved: balance_histogram.png")

# --- Step 5: Data Preprocessing ---
print("\n" + "="*30)
print("Step 5: Data Preprocessing")
print("="*30)

# Import LabelEncoder to convert text columns into numbers
# (Machine Learning models only understand numbers, not text)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()  # Create a LabelEncoder object

# Convert "Geography" column: e.g. France=0, Germany=1, Spain=2
data["Geography"] = le.fit_transform(data["Geography"])

# Convert "Gender" column: e.g. Female=0, Male=1
data["Gender"] = le.fit_transform(data["Gender"])

# Drop columns that are NOT useful for prediction
# RowNumber, CustomerId, Surname are just IDs — not real patterns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Separate features (X) and target (y)
# X = all columns the model learns from
# y = the column we want to predict (Exited: 0=stayed, 1=churned)
X = data.drop("Exited", axis=1)   # All columns except Exited
y = data["Exited"]                 # Only the Exited column

print("\nFeatures (X) — columns used for prediction:")
print(X.columns.tolist())

print("\nTarget (y) — column we are predicting:")
print(y.name)

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

# --- Step 6: Train-Test Split ---
print("\n" + "="*30)
print("Step 6: Train-Test Split")
print("="*30)

# Import train_test_split
from sklearn.model_selection import train_test_split

# Split data into 80% for training and 20% for testing
# random_state=42 ensures we get the same split every time we run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("\n--- Training Data ---")
print("X_train shape (80% features):", X_train.shape)
print("y_train shape (80% target):", y_train.shape)

print("\n--- Testing Data ---")
print("X_test shape (20% features):", X_test.shape)
print("y_test shape (20% target):", y_test.shape)

# --- Step 7: Model Training ---
print("\n" + "="*30)
print("Step 7: Model Training")
print("="*30)

# Import the Logistic Regression model
from sklearn.linear_model import LogisticRegression

# 1. Create the model
# max_iter=1000 gives the model enough time to find the best pattern
model = LogisticRegression(max_iter=1000, random_state=42)

# 2. Train the model using the training data!
print("Training the Logistic Regression model... (this takes a second)")
model.fit(X_train, y_train)

print("Model training complete!")

# --- Step 8: Prediction ---
print("\n" + "="*30)
print("Step 8: Prediction")
print("="*30)

# Ask the model to predict 'Exited' (churn) for the 1000 testing rows
print("Making predictions on the testing data...")
y_pred = model.predict(X_test)

# Display a few sample predictions compared to the actual answers
print("\n--- Sample Predictions ---")

# We convert them to lists just so they print nicely side-by-side
actual_values = y_test.head(10).tolist()
predicted_values = y_pred[:10].tolist()

print("Customer | Actual (0=Stay, 1=Churn) | Predicted (0=Stay, 1=Churn)")
print("-" * 65)
for i in range(10):
    print(f"   {i+1:<5} | {actual_values[i]:<25} | {predicted_values[i]}")
