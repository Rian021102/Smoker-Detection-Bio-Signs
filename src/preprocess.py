import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
import joblib

def preprocess(X_train, y_train, X_test):
    """
    A function for preprocessing data.

    Parameters:
    -----------
    X_train : numpy.ndarray
        The input features for training the model.
    y_train : numpy.ndarray
        The target variable for training the model.
    X_test : numpy.ndarray
        The input features for testing the model.
    feature_names : list
        The list of feature names.

    Returns:
    --------
    X_train_sm : numpy.ndarray
        The preprocessed input features for training the model.
    y_train_sm : numpy.ndarray
        The preprocessed target variable for training the model.
    X_test_scaled : numpy.ndarray
        The preprocessed input features for testing the model.
    """
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Remove outliers using Isolation Forest
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X_train)
    mask = yhat != -1
    X_train, y_train = X_train[mask, :], y_train[mask]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Resample the data
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)

    # Save the scaler as a pickle file
    joblib.dump(scaler, "/Users/rianrachmanto/pypro/project/smoker-detection/models/scaler.pkl")
    

   
    return X_train_sm, y_train_sm, X_test_scaled
