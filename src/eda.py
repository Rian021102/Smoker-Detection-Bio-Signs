import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def eda(X_train,y_train):
    """
    A function for exploratory data analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe for analysis
        
    Returns:
    --------
    pandas.DataFrame
        The input dataframe for analysis, as is
    """
    
    # Print df 5 first rows
    print(X_train.head())

    # Print df 5 last rows
    print(X_train.tail())

    # Print the number of missing values
    print(f"Missing: {X_train.isnull().sum()}")
    
    # Print descriptive statistics
    print(X_train.describe())

    # Print Skewness
    print(X_train.skew())


    #print number of df['smoking'] using value_counts
    print(y_train.value_counts())

    #print tttest_ind for X_train
    for col in X_train.columns:
        print(ttest_ind(X_train[col], y_train))


    #plot kdeplot for in loop
    for col in X_train.columns:
        plt.figure()
        sns.kdeplot(X_train[col])
        plt.title(col)
    plt.show()

    #plot boxplot for in loop
    for col in X_train.columns:
        plt.figure()
        sns.boxplot(X_train[col])
        plt.title(col)
    plt.show()

    #plot countplot value_counts for y_train
    plt.figure()
    sns.countplot(y_train)
    plt.show()
    return X_train, y_train