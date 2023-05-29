import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest

def preprocess(X_train,y_train,X_test):
    """
    A function for preprocessing data.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to be preprocessed.

    Returns:
    --------
    X_train_sm : pandas.DataFrame
        The preprocessed input features for training the model.
    y_train_sm : pandas.Series
        The preprocessed target variable for training the model.
    X_val_scaled : pandas.DataFrame
        The preprocessed input features for validating the model.
    y_val_sm : pandas.Series
        The preprocessed target variable for validating the model.
    """
    #impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)

    #remove outliers using Isolation Forest
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X_train)
    mask = yhat != -1
    X_train, y_train = X_train[mask, :], y_train[mask]



    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled= scaler.transform(X_test)

    # Resample the data
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)
    #print values after resampling
    print('After resampling:')
    print(y_train_sm.value_counts())


    return X_train_sm, y_train_sm, X_test_scaled