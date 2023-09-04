import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Initialize logging format and level
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def load_data(pathfile):
    # Log a message indicating that data loading is starting
    logging.info('Loading data')
    
    # Load the data using Pandas read_csv()
    df = pd.read_csv(pathfile)

    # Log a message indicating that data loading is complete
    logging.info('Data loaded')

    #setting new columns name
    new_columns={'height(cm)':'height', 'weight(kg)':'weight', 'waist(cm)':'waist', 'eyesight(left)':'eyesigth_left', 
                 'eyesight(right)':'eyesight_right', 'hearing(left)':'hearing_left', 
                 'hearing(right)':'hearing_right', 'fasting blood sugar':'fasting_blood_sugar', 
                 'Urine protein':'urine_protein', 'serum creatinine':'serum_creatinine', 
                 'dental caries':'dental_caries'}
    df=df.rename(columns=new_columns)

    # Display the first 5 rows
    print(df.head())

    # Display the last 5 rows
    print(df.tail())

    # Display the shape of the data
    print(df.shape)

    # Display the data types
    print(df.dtypes)

    #Drop relaxation, LDL, and waist(cm) columns

    df.drop(['relaxation', 'LDL', 'waist'], axis=1, inplace=True)

    # Determine the number of duplicated rows
    num_duplicates = df.duplicated().sum()
    
    # Log the number of duplicates found
    logging.info(f'{num_duplicates} duplicated rows found')

    # Remove the duplicated rows
    df.drop_duplicates(inplace=True)

    # Get the new shape of the DataFrame and log it
    new_shape = df.shape
    logging.info(f'The new shape of the DataFrame is {new_shape}')

    #set X and y
    X = df.drop('smoking', axis=1)
    y = df['smoking']


    #split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    return X_train, X_test, y_train, y_test