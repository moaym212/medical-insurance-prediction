import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import kagglehub
import os

# -------------------------------
# Load dataset from Kaggle
# -------------------------------
path = kagglehub.dataset_download("mosapabdelghany/medical-insurance-cost-dataset")
df = pd.read_csv(os.path.join(path, "insurance.csv"))

# -------------------------------
# Data preprocessing
# -------------------------------
categorical_cols = ["sex", "smoker", "region"]
df_label = df.copy()
le = LabelEncoder()
for col in categorical_cols:
    df_label[col] = le.fit_transform(df_label[col])

X = df_label.drop("charges", axis=1)
y = df_label["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train the model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Evaluate the model
# -------------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💊 Medical Insurance Cost Prediction")
st.write("This app predicts medical insurance cost based on user input data.")

st.sidebar.header("Enter your information:")

# User input fields
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=1)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Prepare input data for prediction
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# Encode categorical columns (using the same encoder)
for col in categorical_cols:
    input_data[col] = le.fit(df[col]).transform(input_data[col])

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("Predict Now"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Insurance Cost: ${prediction:,.2f}")

# -------------------------------
# Model information
# -------------------------------
with st.expander("📊 Model Information"):
    st.write("**RMSE:**", round(rmse, 2))
    st.write("**R² Score:**", round(r2, 3))
    st.dataframe(df.head())


