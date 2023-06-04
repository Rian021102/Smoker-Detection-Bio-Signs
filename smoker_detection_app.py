import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from functools import lru_cache

# Load the trained model
@lru_cache(maxsize=1)
def load_model():
    with open('/Users/rianrachmanto/pypro/project/smoker-detection/models/model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

def preprocess_data(df):
    # Perform any necessary preprocessing here (e.g., dropping columns, renaming columns)
    df.drop(['LDL', 'relaxation', 'waist(cm)'], axis=1, inplace=True)
    
    # Scale the data using StandardScaler
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    
    return df

def predict(df):
    # Preprocess the input data
    df2 = df.copy()
    df = preprocess_data(df)

    # Make predictions using the model
    predictions = model.predict(df)

    # Create a new column 'Prediction' with labels 'Non-Smoker' and 'Smoker'
    predictions_labels = ['Non-Smoker' if pred == 0 else 'Smoker' for pred in predictions]
    df2['Prediction'] = predictions_labels

    return df2

def main():
    st.title("Smoker Detection Prediction Using Machine Learning")

    # Allow the user to upload a CSV file
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display the original dataframe
        st.subheader("Original Data")
        st.write(df)

        # Make predictions and display the results
        if st.button('Predict'):
            predictions_df = predict(df)
            st.subheader("Predictions")
            st.write(predictions_df)

if __name__ == '__main__':
    main()
