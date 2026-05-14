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
plt.show()
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
plt.show()
print("Chart 2 saved: balance_histogram.png")
