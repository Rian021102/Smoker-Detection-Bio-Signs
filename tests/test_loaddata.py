import pytest
from src.loaddata import load_data

def test_load_data(caplog):
    # Use os library to build the path dynamically
    pathfile = '/Users/rianrachmanto/pypro/project/smoker-detection/data/raw/train_smokin.csv'

    # Call the load_data function with the example data file
    X_train, X_test, y_train, y_test = load_data(pathfile)

    # Check if the expected columns are present in the data
    assert 'age' in X_train.columns
    assert 'height(cm)' in X_train.columns
    assert 'smoking' not in X_train.columns

    # Check if the data has been split into train and test sets
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

    # Print the captured log records
    for record in caplog.records:
        print(record.levelname, record.message)
