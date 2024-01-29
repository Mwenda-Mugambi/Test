

import streamlit as st
import pickle
import joblib
from prophet import Prophet
import pandas as pd

# Load the saved models
# Correct way to load the model using pickle
with open('prophet_model_super.pkl', 'rb') as file:
    model_super = pickle.load(file)

with open('prophet_model_diesel.pkl', 'rb') as file:
    model_diesel = pickle.load(file)

with open('prophet_model_kerosene.pkl', 'rb') as file:
    model_kerosene = pickle.load(file)

def main():

    # Streamlit app layout
    st.title('EPRA Fuel Predictor')

    # Date selection widget
    selected_date = st.date_input("Select a date for fuel price prediction:")

    # Function to make predictions
    def make_prediction(model, future_date):
        future_df = pd.DataFrame({'ds': [future_date]})
        forecast = model.predict(future_df)
        return forecast['yhat'].iloc[0]

# Display predictions
    if st.button('Predict Prices'):
        price_super = make_prediction(model_super, selected_date)
        price_diesel = make_prediction(model_diesel, selected_date)
        price_kerosene = make_prediction(model_kerosene, selected_date)

        st.write(f"Predicted Super Price on {selected_date}: {price_super:.2f}")
        st.write(f"Predicted Diesel Price on {selected_date}: {price_diesel:.2f}")
        st.write(f"Predicted Kerosene Price on {selected_date}: {price_kerosene:.2f}")

if __name__ == "__main__":
    main()