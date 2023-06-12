from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pickle

PARAM_GRID = [
    {
        'classifier__n_estimators': [10, 100, 1000],
        'classifier__max_features': [2, 3],
        'classifier__max_depth': [2, 5, 10],
    }
]

def train(X_train_sm, y_train_sm, X_test_scaled, y_test):
    # GridSearchCV
    pipe = Pipeline([('classifier', RandomForestClassifier())])
    clf = RandomizedSearchCV(pipe, PARAM_GRID, cv=5, verbose=0, n_jobs=4)
    best_clf = clf.fit(X_train_sm, y_train_sm)

    # Predict
    y_pred = best_clf.predict(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')

    # Save trained model
    with open('/Users/rianrachmanto/pypro/project/smoker-detection/models/model.pkl', 'wb') as f:
        pickle.dump(best_clf, f)


    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
