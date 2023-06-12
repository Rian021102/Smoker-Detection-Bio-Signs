from src.loaddata import load_data
from src.eda import eda
from src.preprocess import preprocess
from src.train_model import train



def main():
    # Load the data
    path='/Users/rianrachmanto/pypro/project/smoker-detection/data/raw/train_smokin.csv'
    X_train, X_test, y_train, y_test = load_data(path)
    print('Data loaded')
    
    # Perform EDA
    eda(X_train, y_train)
    print('EDA complete')

    # Preprocess the data
    X_train_sm, y_train_sm, X_test_scaled = preprocess(X_train, y_train, X_test)
    print('Data preprocessed')

    # Train the model
    train(X_train_sm, y_train_sm, X_test_scaled, y_test)
    print('Models trained')

    
    
if __name__ == '__main__':
    main()