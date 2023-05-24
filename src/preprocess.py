import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
def preprocess(X_train, y_train, X_val):
    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sm)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, y_train_sm, X_val_scaled