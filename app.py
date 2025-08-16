import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os # Import the os module for path checking
from sklearn.base import BaseEstimator, TransformerMixin # Needed for the custom transformer

# --- Custom Transformer (MUST be defined for joblib to load the pipeline) ---
# This class needs to be exactly as it was defined in your original notebook
# when you created and saved the 'full_pipeline'.
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # These indices refer to the column positions BEFORE one-hot encoding
    # Ensure they match your original data's column order.
    # From your notebook: rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
    # Assuming the order is consistent with the original housing dataframe:
    # 'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'
    # So, total_rooms is index 3, total_bedrooms is index 4, population is index 5, households is index 6.
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        # Define indices as instance variables, or ensure they are globally accessible if hardcoded
        # In a real app, it's safer to pass these indices or column names dynamically if columns could shift
        self.rooms_ix = 3
        self.bedrooms_ix = 4
        self.population_ix = 5
        self.households_ix = 6

    def fit(self, X, y=None):
        return self # Nothing to learn here

    def transform(self, X):
        # Ensure X is a NumPy array for correct indexing
        X_array = X.values if isinstance(X, pd.DataFrame) else X

        rooms_per_household = X_array[:, self.rooms_ix] / X_array[:, self.households_ix]
        population_per_household = X_array[:, self.population_ix] / X_array[:, self.households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X_array[:, self.bedrooms_ix] / X_array[:, self.rooms_ix]
            # Use np.c_ to concatenate columns
            return np.c_[X_array, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X_array, rooms_per_household, population_per_household]


# --- 1. Load the Saved Model and Pipeline ---
# Ensure the .pkl files are in the same directory as this script.
# You might need to adjust paths if they are in different locations.
pipeline_path = 'full_pipeline.pkl'
model_path = 'house_price_model.pkl'

# Check if files exist before loading
if not os.path.exists(pipeline_path):
    st.error(f"Error: Pipeline file not found at {pipeline_path}. Please make sure 'full_pipeline.pkl' is in the same directory.")
    st.stop() # Stop the app if file is missing

if not os.path.exists(model_path):
    st.error(f"Error: Model file not found at {model_path}. Please make sure 'house_price_model.pkl' is in the same directory.")
    st.stop() # Stop the app if file is missing

try:
    full_pipeline = joblib.load(pipeline_path)
    final_model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model or pipeline: {e}")
    st.stop() # Stop the app if loading fails


# --- 2. Define the Streamlit App Layout and Widgets ---
st.set_page_config(page_title="California House Price Predictor", layout="centered")

st.title("🏡 California House Price Predictor")
st.markdown("Enter the property details below to get an estimated house price.")

# Input fields for numerical features
st.header("Property Details")

# Using columns for better layout
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-122.25, format="%.2f", help="A measure of how far west a house is; a higher value is farther west")
    latitude = st.number_input("Latitude", value=37.88, format="%.2f", help="A measure of how far north a house is; a higher value is farther north")
    housing_median_age = st.number_input("Housing Median Age", value=29.0, format="%.1f", min_value=1.0, max_value=st.session_state.get('max_age_data', 52.0), help="Median age of a house within a block; lower number is a newer building")
    total_rooms = st.number_input("Total Rooms", value=2635.0, format="%.1f", min_value=1.0, help="Total number of rooms within a block")
    total_bedrooms = st.number_input("Total Bedrooms", value=538.0, format="%.1f", min_value=1.0, help="Total number of bedrooms within a block")

with col2:
    population = st.number_input("Population", value=1132.0, format="%.1f", min_value=1.0, help="Total number of people residing within a block")
    households = st.number_input("Households", value=498.0, format="%.1f", min_value=1.0, help="Total number of households, a group of people residing within a home unit")
    median_income = st.number_input("Median Income (in tens of thousands USD)", value=3.87, format="%.2f", min_value=0.0, help="Median income for households within a block of houses (e.g., 3.87 is $38,700)")
    # Ocean Proximity is a categorical feature
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ('NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'),
        help="Location of the house with respect to the ocean"
    )

# --- 3. Make Prediction on Button Click ---
if st.button("Predict House Price"):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([{
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }])

    try:
        # Preprocess the input data using the loaded pipeline
        # The pipeline expects data in the original column order/format
        prepared_input = full_pipeline.transform(input_data)

        # Make prediction using the loaded model
        prediction = final_model.predict(prepared_input)

        # Display the prediction
        st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
        st.balloons() # A little celebration!

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please check your input values and try again.")

st.markdown("---")
st.caption("This model predicts median house values in California districts based on 1990 census data.")
st.caption("Created using Scikit-learn and Streamlit.")
