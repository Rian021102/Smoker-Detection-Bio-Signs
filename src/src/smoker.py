import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle                                        

import logging as logger
def load_data():
    logger.info('Loading data')
    df = pd.read_csv('/Users/rianrachmanto/pypro/project/script/train_smokin.csv')
    logger.info('Data loaded')
    return df

import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)

def eda(df, target_col):
    """
    Conduct exploratory data analysis (EDA) on a Pandas dataframe.

    Args:
    df: Pandas dataframe containing the data.
    target_col: Name of the target column.

    Returns:
    None
    """
    try:
        # Drop columns
        logging.info("Dropping columns")
        df.drop(['relaxation', 'LDL', 'waist(cm)'], axis=1, inplace=True)
        
        # Check absence of predictors
        logging.info("Checking absence of predictors")
        expected_columns = ['age', 'height(cm)', 'weight(kg)', 'eyesight(left)', 'eyesight(right)',
                           'hearing(left)', 'hearing(right)', 'systolic', 'fasting blood sugar',
                           'Cholesterol', 'triglyceride', 'HDL', 'hemoglobin', 'Urine protein',
                           'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries', target_col]
        assert set(expected_columns) == set(df.columns), f"Expected columns {expected_columns} not found in data."
        
        # Check data types
        logging.info("Checking data types")
        print(df.dtypes)
        
        # Check for missing data
        logging.info("Checking for missing data")
        print(df.isnull().sum())
        
        # Check for outliers using boxplots
        logging.info("Checking for outliers")
        for column in df.columns:
            plt.figure()
            sns.boxplot(x=df[column])
            plt.title(column)
        
        # Check data range
        logging.info("Checking data range")
        print(df.describe())
        
        # Check for imbalanced data
        logging.info("Checking for imbalanced data")
        sns.countplot(x=target_col, data=df)
    except Exception as e:
        logging.exception(e)
  

def handle_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Handle missing values, missing columns, and outliers in a Pandas dataframe.

    Args:
        df (pd.DataFrame): Pandas dataframe containing the data.
        target_col (str): Name of the target column.

    Returns:
        pd.DataFrame: Pandas dataframe with missing values filled, missing columns added with NaN values, and outliers handled.
    """
    # Check absence of predictors
    expected_columns = [
        'age', 'height(cm)', 'weight(kg)', 'eyesight(left)', 'eyesight(right)',
        'hearing(left)', 'hearing(right)', 'systolic', 'fasting blood sugar',
        'Cholesterol', 'triglyceride', 'HDL', 'hemoglobin', 'Urine protein',
        'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries', target_col
    ]
    if not set(expected_columns).issubset(set(df.columns)):
        raise ValueError(
            f"Missing columns. Expected columns: {expected_columns}, got: {df.columns}"
        )

    # Handle missing values
    missing_cols = df.columns[df.isnull().sum() > 0]
    for col in missing_cols:
        if df[col].dtype == 'object':
            print(f"Missing values in categorical column {col}.")
            # Fill with mode
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            print(f"Missing values in numerical column {col}.")
            # Fill with mean
            df[col].fillna(df[col].mean(), inplace=True)

    # Handle outliers using the interquartile range (IQR) method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df.clip(lower_bound, upper_bound, axis=1)

    return df

def preprocess_data(df, target_col):
    """
    Preprocesses the data by setting the predictors and target variable.

    Args:
    df (pandas dataframe): Input dataframe containing the data.
    target_col (str): Target variable column name.

    Returns:
    X_train (numpy array): Predictor values for the training set.
    X_val (numpy array): Predictor values for the validation set.
    y_train (numpy array): Target values for the training set.
    y_val (numpy array): Target values for the validation set.
    """
    # Check if df is a dataframe
    assert isinstance(df, pd.DataFrame), "df should be a pandas dataframe"

    # Check if target_col is a string
    assert isinstance(target_col, str), "target_col should be a string"

    # Check if target_col is in the dataframe
    assert target_col in df.columns, "target_col should be in the dataframe"

    # Set X and y variables
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

def train_models(X_train, y_train, X_val, y_val):
    # Define models to train
    models = [
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('XGBoost', xgb.XGBClassifier()),
        ('AdaBoost', AdaBoostClassifier())
    ]
    
    # Define scaler
    scaler = StandardScaler()

    # Train and evaluate models
    scores = []
    reports = []
    for name, model in models:
        pipeline = Pipeline(steps=[
            ('scaler', scaler),
            ('model', model)
        ])
        try:
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=5, scoring='accuracy'
            )
        except ValueError as e:
            print(f"Error training {name}: {e}")
            cv_scores = [0.0]

        scores.append((name, cv_scores.mean()))

        try:
            pipeline.fit(X_train, y_train)
        except ValueError as e:
            print(f"Error training {name}: {e}")
            continue

        y_pred = pipeline.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True)
        reports.append((name, report))

    return scores, reports


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Trains a random forest classifier on the preprocessed training data and evaluates on the preprocessed validation data.

    Args:
    X_train (numpy array): Preprocessed training data features.
    y_train (numpy array): Preprocessed training data target variable.
    X_val (numpy array): Preprocessed validation data features.
    y_val (numpy array): Preprocessed validation data target variable.

    Returns:
    Tuple:
    - RandomForestClassifier: Trained random forest classifier.
    - float: Validation accuracy score.
    """
    
    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.info('Training random forest model')
    
    # Create a pipeline with a standard scaler and random forest classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier())
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate the pipeline on the validation data
    try:
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
    except Exception as e:
        logger.exception(f'Error while evaluating random forest model: {e}')
        accuracy = None
    
    # Log validation accuracy
    if accuracy is not None:
        logger.info(f'Validation accuracy: {accuracy:.3f}')
    
    return pipeline, accuracy


def predict_with_random_forest(pipeline, X_test):
    """
    Predicts the labels of a new dataset using a trained random forest pipeline.

    Args:
    pipeline (Pipeline): Trained random forest pipeline.
    X_test (numpy array): Preprocessed test data features.

    Returns:
    numpy array: Predicted target variable for the test data.
    """
    
    # Predict the labels for the new data
    y_pred = pipeline.predict(X_test)
    
    return y_pred




