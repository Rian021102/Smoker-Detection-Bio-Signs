import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report


def model_based(X_train_sm, y_train_sm, X_test, y_test):
    # build baseline model using logistic regression
    logreg = LogisticRegression()
    logreg.fit(X_train_sm, y_train_sm)
    y_pred = logreg.predict(X_test)
    print('Logistic Regression')
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    print('Precision: {:.2f}'.format(precision_score(y_test, y_pred)))
    print('Recall: {:.2f}'.format(recall_score(y_test, y_pred)))
    print('F1: {:.2f}'.format(f1_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))
    
    # build models
    models = ['Random Forest', RandomForestClassifier(), 'Gradient Boosting', GradientBoostingClassifier(), 
              'XGBoost', XGBClassifier()]
    for i in range(0, len(models), 2):
        model = models[i+1]
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test)
        print(models[i])
        print('Accuracy of {} classifier on test set: {:.2f}'.format(models[i], model.score(X_test, y_test)))
        print('Precision: {:.2f}'.format(precision_score(y_test, y_pred)))
        print('Recall: {:.2f}'.format(recall_score(y_test, y_pred)))
        print('F1: {:.2f}'.format(f1_score(y_test, y_pred)))
        print(classification_report(y_test, y_pred))
        print('\n')

    return None




    
