import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
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

ssl._create_default_https_context = ssl._create_unverified_context

def load_data():
    """
    Load the data from a Google Drive link into a Pandas dataframe.

    Returns:
    df: Pandas dataframe containing the loaded data.
    """
    url_df = 'https://drive.google.com/file/d/1RoHxCPov-bGRKLiZ5a9t_Frc9VfqUAJh/view?usp=share_link'
    path_df = 'https://drive.google.com/uc?export=download&id='+url_df.split('/')[-2]
    df = pd.read_csv(path_df)
    return df
df = load_data()
# Drop columns
df.drop(['relaxation', 'LDL', 'waist(cm)'], axis=1, inplace=True)

def eda(df, target_col):
    # Check absence of predictors
    expected_columns = ['age', 'height(cm)', 'weight(kg)', 'eyesight(left)', 'eyesight(right)',
                       'hearing(left)', 'hearing(right)', 'systolic', 'fasting blood sugar',
                       'Cholesterol', 'triglyceride', 'HDL', 'hemoglobin', 'Urine protein',
                       'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries', target_col]
    assert set(expected_columns) == set(df.columns), f"Expected columns {expected_columns} not found in data."
    
    # Check data types
    print(df.dtypes)
    
    # Check for missing data
    print(df.isnull().sum())
    
    # Check for outliers using boxplots
    for column in df.columns:
        plt.figure()
        sns.boxplot(x=df[column])
        plt.title(column)
    
    # Check data range
    print(df.describe())
    
    # Check for imbalanced data
    sns.countplot(x=target_col, data=df)

eda(df, 'smoking')


def handle_data(df, target_col):
    """
    Handle missing values, missing columns, and outliers in a Pandas dataframe.

    Args:
    df: Pandas dataframe containing the data.
    target_col: Name of the target column.

    Returns:
    df: Pandas dataframe with missing values filled, missing columns added with NaN values, and outliers handled.
    """
    # Check absence of predictors
    expected_columns = ['age', 'height(cm)', 'weight(kg)', 'eyesight(left)', 'eyesight(right)',
                       'hearing(left)', 'hearing(right)', 'systolic', 'fasting blood sugar',
                       'Cholesterol', 'triglyceride', 'HDL', 'hemoglobin', 'Urine protein',
                       'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries', target_col]
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        for col in missing_columns:
            df[col] = np.nan
    
    # Handle missing values
    missing_cols = df.columns[df.isnull().sum() > 0]
    for col in missing_cols:
        if df[col].dtype == 'object':
            # Fill with mode
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
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
df = handle_data(df, 'smoking')

def prepare_data(df, target_col, test_size=0.2, random_state=42):
    """
    Preprocesses the data by setting the predictors and target variable, scaling the data,
    and splitting it into training and validation sets.

    Args:
    df (pandas dataframe): Input dataframe containing the data.
    target_col (str): Target variable column name.
    test_size (float, optional): Proportion of data to be used for validation. Defaults to 0.2.
    random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
    X_train (numpy array): Scaled predictor values for training set.
    X_val (numpy array): Scaled predictor values for validation set.
    y_train (numpy array): Target values for training set.
    y_val (numpy array): Target values for validation set.
    scaler (sklearn scaler object): Scaler object used to scale data.
    """
    # Set X and y variables
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_val, y_train, y_val, scaler

X_train, X_val, y_train, y_val, scaler = prepare_data(df, 'smoking')



def handle_imbalanced_data(X_train, y_train):
    """
    Resample the minority class using SMOTE to handle imbalanced data.

    Args:
    X_train: numpy array or Pandas dataframe containing the predictors for the training set.
    y_train: numpy array or Pandas series containing the target variable for the training set.

    Returns:
    X_train_resampled: numpy array containing the resampled predictors.
    y_train_resampled: numpy array containing the resampled target variable.
    """
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

X_train_resampled, y_train_resampled = handle_imbalanced_data(X_train, y_train)


def train_models(df, target_col):
    """
    Trains multiple classification models and returns their cross-validation scores and classification reports.

    Parameters:
    df: Pandas dataframe containing the data.
    target_col: Name of the target column.

    Returns:
    scores: List of cross-validation scores for each model.
    reports: List of classification reports for each model.
    """
    
    # Define models to train
    models = [
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('XGBoost', xgb.XGBClassifier()),
        ('AdaBoost', AdaBoostClassifier())
    ]

    # Train and evaluate models
    scores = []
    reports = []
    for name, model in models:
        pipeline = Pipeline(steps=[('model', model)])
        cv_scores = cross_val_score(pipeline, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
        scores.append((name, cv_scores.mean()))
        
        pipeline.fit(X_train_resampled, y_train_resampled)
        y_pred = pipeline.predict(X_train_resampled)
        report = classification_report(y_train_resampled, y_pred, output_dict=True)
        reports.append((name, report))

    return scores, reports

scores, reports = train_models(df, 'target')

scores, reports = train_models(df, 'target')
for name, score in scores:
    print(f'{name} cross-validation score: {score:.3f}')
    
for name, report in reports:
    print(f'{name} classification report:\n{report}\n')

def train_xgboost(X_train_resampled, y_train_resampled, X_val, y_val):
    """
    Trains an XGBoost classifier on the preprocessed training data and evaluates on the preprocessed validation data.

    Parameters:
    X_train_resampled: Preprocessed training data features after resampling and scaling.
    y_train_resampled: Preprocessed training data target variable after resampling.
    X_val: Preprocessed validation data features after scaling.
    y_val: Preprocessed validation data target variable.

    Returns:
    XGBClassifier: Trained XGBoost classifier.
    float: Validation accuracy score.
    """
    # Create an XGBoost classifier
    xgb_clf = xgb.XGBClassifier()

    # Train the classifier
    xgb_clf.fit(X_train_resampled, y_train_resampled)

    # Evaluate the classifier on the validation data
    y_pred = xgb_clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f'Validation accuracy: {accuracy:.3f}')

    return xgb_clf, accuracy

scores, reports = train_models(df, 'target')
for name, score in scores:
    print(f'{name} cross-validation score: {score:.3f}')
    
for name, report in reports:
    print(f'{name} classification report:\n{report}\n')

xgb_clf, accuracy = train_xgboost(X_train_resampled, y_train_resampled, X_val, y_val)

# Train and save the XGBoost classifier
xgb_clf, accuracy = train_xgboost(X_train_resampled, y_train_resampled, X_val, y_val)
with open('xgb.pkl', 'wb') as f:
    pickle.dump(xgb_clf, f)

# Load the XGBoost classifier
with open('xgb.pkl', 'rb') as f:
    xgb_clf = pickle.load(f)




