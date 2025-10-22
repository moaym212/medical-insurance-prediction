import streamlit as st
import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. App Header
# -------------------------------
st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="💰", layout="wide")
st.title("🌲 Medical Insurance Cost Prediction (Random Forest)")
st.write("This app predicts medical insurance costs using a **Random Forest Regressor** trained on real data from Kaggle.")

# -------------------------------
# 2. Cached Data Loading and Model Training
# -------------------------------
@st.cache_data
def load_and_train():
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("mosapabdelghany/medical-insurance-cost-dataset")
    df = pd.read_csv(os.path.join(path, "insurance.csv"))

    # Label encoding for categorical columns
    categorical_cols = ["sex", "smoker", "region"]
    df_label = df.copy()
    le = LabelEncoder()
    for col in categorical_cols:
        df_label[col] = le.fit_transform(df_label[col])

    # Split data
    X = df_label.drop("charges", axis=1)
    y = df_label["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=None
    )
    rf_model.fit(X_train, y_train)

    # Evaluate model
    y_pred = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return rf_model, le, rmse, r2, df


model, le, rmse, r2, df = load_and_train()

# -------------------------------
# 3. Sidebar Inputs
# -------------------------------
st.sidebar.header("🧍 Enter Your Information")

age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=1)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# -------------------------------
# 4. Prepare Input Data
# -------------------------------
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# Encode categorical columns
categorical_cols = ["sex", "smoker", "region"]
for col in categorical_cols:
    input_data[col] = le.fit(df[col]).transform(input_data[col])

X_columns = ["age", "sex", "bmi", "children", "smoker", "region"]
input_data = input_data[X_columns]

# -------------------------------
# 5. Prediction
# -------------------------------
st.write("---")
if st.button("🔍 Predict Insurance Cost"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"💵 Estimated Insurance Cost: **${prediction:,.2f}**")

# -------------------------------
# 6. Model Info
# -------------------------------
st.write("---")
col1, col2 = st.columns(2)
col1.metric("Model RMSE", f"{rmse:,.2f}")
col2.metric("R² Score", f"{r2:.3f}")

with st.expander("📊 Dataset Preview"):
    st.dataframe(df.head())

st.caption("Data Source: Kaggle – Medical Insurance Cost Dataset by mosapabdelghany")
