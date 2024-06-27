import streamlit as st
import pandas as pd
import statsmodels.api as sm
import pickle

# Function to preprocess data for Streamlit app
def preprocess_data_for_streamlit(df):
    X = df.drop('target_deathrate', axis=1)
    return X

# Function to predict using the trained linear regression model
def predict(lr_model, input_data):
    input_data_with_const = sm.add_constant(input_data)
    prediction = lr_model.predict(input_data_with_const)
    return prediction[0]  # Assuming you want to return a scalar prediction

def main():
    st.title('Cancer Mortality Rate Prediction')

    # Load your dataset
    df = pd.read_csv("Cancer_Mortality_Cleaned_data.csv")

    # Preprocess data for Streamlit app
    X_streamlit = preprocess_data_for_streamlit(df)

    # Display sidebar with input fields
    st.sidebar.header('Input Parameters')

    # Display all available columns for user input
    user_input = {}
    for col in X_streamlit.columns:
        user_input[col] = st.sidebar.number_input(col, value=X_streamlit[col].mean())

    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([user_input])

    # Check the format of input_data
    st.write("Input Data Format:")
    st.write(input_data)

    # Ensure the input data includes the same features as the model's training data
    if "const" not in input_data.columns:
        input_data = sm.add_constant(input_data, has_constant='add')

    # Load the trained model
    try:
        with open("trained_model.pkl", "rb") as f:
            lr_model = pickle.load(f)

        # Add a "Predict" button
        if st.button('Predict'):
            # Predict based on user input
            prediction = predict(lr_model, input_data)
            st.markdown(f'<h2>Predicted Cancer Mortality Rate: {prediction}</h2>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading or predicting with the model: {str(e)}")

if __name__ == "__main__":
    main()
