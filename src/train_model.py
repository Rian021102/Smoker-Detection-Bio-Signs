from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import joblib
from joblib import dump, load

def train (X_train_scaled, y_train_sm, X_val_scaled,y_val):
    # GridSearchCV
    pipe = Pipeline([('classifier', RandomForestClassifier())])
    param_grid = [
        {'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [10, 100, 1000],
        'classifier__max_features': [2, 3]},
    ]
    clf = RandomizedSearchCV(pipe, param_grid, cv=5, verbose=0, n_jobs=4)
    best_clf = clf.fit(X_train_scaled, y_train_sm)
    # Predict
    y_pred = best_clf.predict(X_val_scaled)
    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    print("Accuracy: {:.4%}".format(accuracy))
    print("Precision: {:.4%}".format(precision))
    print("Recall: {:.4%}".format(recall))
    print("F1: {:.4%}".format(f1))
    #save trained model
    joblib.dump(best_clf, '/Users/rianrachmanto/pypro/project/smoker-detection/models/model.joblib')
    return accuracy, precision, recall, f1