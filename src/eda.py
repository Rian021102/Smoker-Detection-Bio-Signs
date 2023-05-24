import pandas as pd
import numpy as np

def eda(X_train, y_train):
    """
    A function for exploratory data analysis.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        The input features for training the model.
    y_train : pandas.Series
        The target variable for training the model.
        
    Returns:
    --------
    None
    """
    # Print the datafame shape
    print(f"Shape: {X_train.shape}")
    
    # Print dataframe info and datatypes
    print(X_train.info())
    print(X_train.dtypes)
    
    # Print the number of missing values
    print(f"Missing: {X_train.isnull().sum()}")

    # print missing values more than half of the total rows
    print(f"Missing more than half of the total rows: {X_train.columns[X_train.isnull().mean() > 0.5]}")
    
    # Print descriptive statistics
    print(X_train.describe())
    
    # Print the number of duplicates
    print(f"Duplicates: {X_train.duplicated().sum()}")
    
    # Print the distribution of target variable
    print(y_train.value_counts(normalize=True))
    
    return (X_train, y_train)

