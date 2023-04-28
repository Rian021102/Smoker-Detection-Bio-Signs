from smoker import load_data, eda, handle_data
from smoker import preprocess_data, train_models
from smoker import train_random_forest
df=load_data()
print(df.head())
eda(df, 'smoking')
df=handle_data(df, 'smoking')
X_train, X_val, y_train, y_val = preprocess_data(df, 'smoking')
scores, reports = train_models(X_train, y_train, X_val, y_val)
for name, score in scores:
    print(f"{name} score: {score:.4f}")
for name, report in reports:
    print(f"{name} report:\n{report}")

trained_rf, val_accuracy = train_random_forest(X_train, y_train, X_val, y_val)
print(f'Validation accuracy: {val_accuracy:.3f}')

#predict using the trained model
import pandas as pd
df_test=pd.read_csv('/Users/rianrachmanto/pypro/project/script/test_smoking.csv')
df_test.drop(['LDL','relaxation','waist(cm)'],axis=1,inplace=True)
# Load the trained random forest model
trained_pipeline, _ = train_random_forest(X_train, y_train, X_val, y_val)

# Use the trained pipeline to predict labels for the new dataset
y_pred = trained_pipeline.predict(df_test)
print(y_pred)

# Train a random forest classifier
#pipeline, _ = train_random_forest(X_train, y_train, X_val, y_val)

# Predict the labels for a new dataset
#y_pred = predict_with_random_forest(pipeline, X_test)

import joblib
import os
from joblib import dump

# Train a random forest classifier
pipeline, _ = train_random_forest(X_train, y_train, X_val, y_val)

# Save the pipeline to disk
dump(pipeline, 'random_forest_pipeline.joblib')