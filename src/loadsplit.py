import pandas as pd
from sklearn.model_selection import train_test_split
import logging
print("Inside loadsplit module")

def load_data(pathfile):
    logging.info('Loading data')
    df = pd.read_csv(pathfile)
    logging.info('Data loaded')
    return df


def split_data(df):
    X = df.drop('smoking', axis=1)
    y = df['smoking']
    # Added stratify=y to make sure the train-test split has the same proportion of smoking and non-smoking labels
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_val, y_train, y_val


def main(pathfile):
    logging.basicConfig(level=logging.INFO)
    df = load_data(pathfile)
    X_train, X_val, y_train, y_val = split_data(df)
    # Do something with the data

if __name__ == '__main__':
    main()
