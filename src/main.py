from loaddata import load_data
from eda import eda
from preprocess import preprocess
from train_model import train_model
from model_base import model_based
def main():
    # Load the data
    path='/Users/rianrachmanto/pypro/project/Smoker-Detection-Bio-Signs/data/raw/train_smoking.csv'
    X_train, X_test, y_train, y_test = load_data(path)
    print('Data loaded')
    
    # Perform EDA
    eda(X_train, y_train)
    print('EDA complete')

    # Preprocess the data
    X_train, y_train, X_test = preprocess(X_train, y_train, X_test)
    print('Data preprocessed')

    # Train the model
    train_model(X_train, y_train, X_test, y_test)
    print('Models trained')

    
    
if __name__ == '__main__':
    main()