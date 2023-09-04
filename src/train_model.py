from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pickle

#set parameters
PARAM_GRID = [
    {
        #Hyperparameter Tuning for XGBoost
        'classifier__n_estimators': [100, 200, 300, 400, 500],
        'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'classifier__min_child_weight': [1, 2, 3, 4],
        'classifier__gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'classifier__colsample_bytree': [0.3, 0.4, 0.5, 0.7],
        'classifier__subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
        'classifier__reg_alpha': [0, 0.1, 0.2, 0.3, 0.4],
        'classifier__reg_lambda': [0.8, 0.9, 1.0],
        'classifier__scale_pos_weight': [1, 10, 25, 50, 75, 99, 100, 1000],
        'classifier__max_delta_step': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
        
    }
]

def train_model(X_train, y_train, X_test, y_test):
    # GridSearchCV
    pipe = Pipeline([('classifier', XGBClassifier())])
    clf = RandomizedSearchCV(pipe, PARAM_GRID, cv=5, verbose=0, n_jobs=4)
    best_clf = clf.fit(X_train, y_train)

    # Predict
    y_pred = best_clf.predict(X_test)

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
    print(classification_report(y_test, y_pred))

    # Save trained model
    with open('/Users/rianrachmanto/pypro/project/Smoker-Detection-Bio-Signs/models/model.pkl', 'wb') as f:
        pickle.dump(best_clf, f)


    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
