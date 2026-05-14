import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Page Configuration ---
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# --- 1. Main Title ---
st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a customer is likely to leave (churn) or stay based on their profile.")

# --- 2. Sidebar ---
st.sidebar.header("📌 Project Details")
st.sidebar.info(
    "This is a beginner-friendly Machine Learning project "
    "using the Maven Analytics Customer Churn Dataset."
)
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Navigation")
st.sidebar.markdown("- Dataset Overview\n- Data Visualization\n- Customer Input Form\n- Prediction Results")

# --- Load Data & Train Model ---
@st.cache_data
def load_data():
    # Load the CSV dataset into a DataFrame
    data = pd.read_csv("Churn_Modelling.csv")
    return data

data = load_data()

# Preprocessing
le_geo = LabelEncoder()
le_gender = LabelEncoder()

# Create a clean copy for ML so we don't mess up the visual table
ml_data = data.copy()
ml_data["Geography"] = le_geo.fit_transform(ml_data["Geography"])
ml_data["Gender"] = le_gender.fit_transform(ml_data["Gender"])
ml_data = ml_data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

X = ml_data.drop("Exited", axis=1)
y = ml_data["Exited"]

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the Logistic Regression Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- 3. Dataset Section ---
st.header("📂 Dataset Overview")
st.write("First 5 rows of the dataset:")
st.dataframe(data.head())
st.write(f"**Dataset Shape:** {data.shape[0]} rows and {data.shape[1]} columns.")

# --- 4. Data Visualization ---
st.header("📈 Data Visualization")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Churn Count")
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    churn_counts = data["Exited"].value_counts()
    ax1.bar(["Stayed (0)", "Churned (1)"], churn_counts.values, color=["steelblue", "tomato"])
    ax1.set_ylabel("Customers")
    st.pyplot(fig1)

with col2:
    st.subheader("Age Distribution")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.hist(data["Age"], bins=20, color="skyblue", edgecolor="white")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Customers")
    st.pyplot(fig2)

with col3:
    st.subheader("Balance Distribution")
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.hist(data["Balance"], bins=20, color="mediumpurple", edgecolor="white")
    ax3.set_xlabel("Balance")
    ax3.set_ylabel("Customers")
    st.pyplot(fig3)

# --- 7 & 8. Model Information ---
st.header("🤖 Model Performance")
st.write(f"**Logistic Regression Accuracy:** {accuracy * 100:.2f}%")

col_metric1, col_metric2 = st.columns(2)
with col_metric1:
    st.subheader("Confusion Matrix")
    st.text(confusion_matrix(y_test, y_pred))
    st.caption("(Top-Left: Correctly guessed Stay | Bottom-Right: Correctly guessed Churn)")

with col_metric2:
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, zero_division=0))

# --- 5. Customer Input Form ---
st.header("🧑‍💻 Predict Customer Churn")
st.markdown("Enter customer details below to predict if they will churn or stay.")

# Create input layout with two columns
col_in1, col_in2 = st.columns(2)

with col_in1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    age = st.slider("Age", min_value=18, max_value=100, value=35)
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

with col_in2:
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])

# Convert text inputs to numbers for the model
geo_encoded = le_geo.transform([geography])[0]
gender_encoded = le_gender.transform([gender])[0]
card_encoded = 1 if has_cr_card == "Yes" else 0
active_encoded = 1 if is_active_member == "Yes" else 0

# Create feature array matching training data columns
input_data = pd.DataFrame([[
    credit_score, geo_encoded, gender_encoded, age, tenure, balance, 
    num_products, card_encoded, active_encoded, estimated_salary
]], columns=X.columns)

# --- 6. Prediction System ---
if st.button("Predict Churn"):
    # Predict outcome and probabilities
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    
    # Show results clearly
    if prediction == 1:
        st.error(f"🚨 **Prediction: Customer Will CHURN (Leave)**")
    else:
        st.success(f"✅ **Prediction: Customer Will STAY**")
        
    st.write(f"**Probability of Churning:** {probability[1] * 100:.2f}%")
    st.write(f"**Probability of Staying:** {probability[0] * 100:.2f}%")
