import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import mstats

def cleandata (X_train, X_val):
    """
    Clean numerical and categorical data in the input datasets X_train and X_val. 

    Parameters: 
    X_train (pandas.DataFrame): A pandas DataFrame representing training data. It contains both numerical (int64 and float64) and categorical (object) columns. 
    X_val (pandas.DataFrame): A pandas DataFrame representing validation data. It contains both numerical (int64 and float64) and categorical (object) columns. 

    Returns: 
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the cleaned training and validation data. 
    """
    #Select all numerical columns
    num_cols1 = X_train.select_dtypes(include=['int64', 'float64']).columns
    num_cols2 = X_val.select_dtypes(include=['int64', 'float64']).columns
    #Select all categorical columns
    cat_cols1 = X_train.select_dtypes(include=['object']).columns
    cat_cols2   = X_val.select_dtypes(include=['object']).columns
    #drop columns with half or more missing values
    X_train = X_train.drop(columns=X_train.columns[X_train.isnull().mean() > 0.5], inplace=False)
    X_val = X_val.drop(columns=X_val.columns[X_val.isnull().mean() > 0.5], inplace=False)
    #Impute missing values in numerical columns with mean
    imputer = SimpleImputer(strategy='mean')
    X_train[num_cols1] = imputer.fit_transform(X_train[num_cols1])
    X_val[num_cols2] = imputer.transform(X_val[num_cols2])
    #drop categorical columns with missing values
    X_train = X_train.drop(columns=cat_cols1, inplace=False)
    X_val = X_val.drop(columns=cat_cols2, inplace=False)
    #Remove outliers using Winsorization
    X_train[num_cols1] = mstats.winsorize(X_train[num_cols1].values, limits=[0.05, 0.05])
    return X_train.copy(), X_val.copy()
