import pandas as pd
import streamlit as st
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from joblib import load
from io import BytesIO

# Load the trained pipeline
@st.cache(allow_output_mutation=True)
def load_pipeline():
    url = "https://github.com/Rian021102/Smoker-Detection-Bio-Signs/raw/main/random_forest_pipeline.joblib"
    response = requests.get(url)
    return load(BytesIO(response.content))

pipeline = load_pipeline()


def preprocess_data(df):
    # Perform any necessary preprocessing here (e.g., dropping columns, renaming columns)
    df.drop(['LDL'], axis=1, inplace=True)
    df.drop(['relaxation'], axis=1, inplace=True)
    df.drop(['waist(cm)'],axis=1, inplace=True)
    return df

def predict(df):
    # Preprocess the input data
    df = preprocess_data(df)
    
    # Make predictions using the pipeline
    predictions = pipeline.predict(df)
    
    # Add the predictions as a new column in the dataframe
    df['Prediction'] = predictions
    
    return df

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


