import pandas as pd
def make_features(X_train,X_val):
    #drop columns X_train and X_val
    X_train.drop(['relaxation', 'LDL', 'waist(cm)'], axis=1, inplace=True)
    X_val.drop(['relaxation', 'LDL', 'waist(cm)'], axis=1, inplace=True)
    return X_train, X_val
