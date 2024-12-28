import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st

# Load Dataset
@st.cache
def load_data():
    data = pd.read_csv("house_data.csv")
    return data

# Train the Model
def train_model(data):
    # Selecting features and target variable
    X = data.drop(columns=["Id", "Price"])
    y = data["Price"]

    # Categorical and numerical columns
    categorical_columns = ["Location", "Condition", "Garage"]
    numerical_columns = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_columns),
            ("cat", OneHotEncoder(), categorical_columns),
        ]
    )

    # Linear Regression pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    predictions = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return pipeline, mse

# Streamlit App
def main():
    st.title("House Price Prediction App")
    st.write("Enter house details to predict its price.")

    # Load data
    data = load_data()
    st.write("Dataset Preview:")
    st.write(data.head())

    # Train model
    model, mse = train_model(data)
    st.write(f"Model Mean Squared Error: {mse:.2f}")

    # Input features
    st.sidebar.header("Input Features")
    area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=10000, step=100, value=1500)
    bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1, value=3)
    bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1, value=2)
    floors = st.sidebar.number_input("Number of Floors", min_value=1, max_value=5, step=1, value=1)
    year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2024, step=1, value=2000)
    location = st.sidebar.selectbox("Location", options=data["Location"].unique())
    condition = st.sidebar.selectbox("Condition", options=data["Condition"].unique())
    garage = st.sidebar.selectbox("Garage", options=data["Garage"].unique())

    # Predict house price
    if st.sidebar.button("Predict"):
        input_data = pd.DataFrame(
            [[area, bedrooms, bathrooms, floors, year_built, location, condition, garage]],
            columns=["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt", "Location", "Condition", "Garage"]
        )
        prediction = model.predict(input_data)
        st.write(f"Predicted House Price: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
