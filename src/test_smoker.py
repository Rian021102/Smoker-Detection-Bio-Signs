import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from smoker import load_data, eda, handle_data
from smoker import preprocess_data, train_models
from smoker import train_random_forest,predict_with_random_forest

df=load_data()
print(df.head())
eda(df,'smoking')
df=handle_data(df,'smoking')
X_train, X_val, y_train, y_val = preprocess_data(df, 'smoking')
# assuming X_train, y_train, X_val, y_val have already been defined or loaded
scores, reports = train_models(X_train, y_train, X_val, y_val)
print(scores)
print(reports)
pipeline, accuracy = train_random_forest(X_train, y_train, X_val, y_val)
print(accuracy)
import pandas as pd
X_test=pd.read_csv('/Users/rianrachmanto/pypro/project/smoker-detection/data/raw/test_smoking.csv')
X_test.drop(['relaxation', 'LDL', 'waist(cm)'], axis=1, inplace=True)
# use the pipeline to make predictions on the test data
y_pred = predict_with_random_forest(pipeline, X_test)
print(y_pred)