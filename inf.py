import requests
import pandas as pd
from datetime import timedelta, datetime, timezone
from bytewax.dataflow import Dataflow
from bytewax.connectors.files import CSVInput
from bytewax.connectors.stdio import StdOutput
from bytewax.window import SystemClockConfig, TumblingWindow

url = "http://127.0.0.1:8000/predict"

def process_data(row):
    df = pd.DataFrame([row])
    new_columns = {'height(cm)': 'height', 'weight(kg)': 'weight', 'waist(cm)': 'waist',
                   'eyesight(left)': 'eyesigth_left', 'eyesight(right)': 'eyesight_right',
                   'hearing(left)': 'hearing_left', 'hearing(right)': 'hearing_right',
                   'fasting blood sugar': 'fasting_blood_sugar', 'Urine protein': 'urine_protein',
                   'serum creatinine': 'serum_creatinine', 'dental caries': 'dental_caries'}
    df = df.rename(columns=new_columns)
    
    # Perform any necessary preprocessing here (e.g., dropping columns, renaming columns)
    df.drop(['LDL', 'relaxation', 'waist'], axis=1, inplace=True)
    
    return df

def predict(args, counter, total_rows):
    url, df = args
    
    # Convert DataFrame to CSV file-like object
    csv_file = df.to_csv(index=False).encode('utf-8')
    files = {"file": ("input.csv", csv_file)}
    
    # Predict the data using the model with the provided URL and files
    response = requests.post(url, files=files)
    # Convert response to DataFrame
    predict = pd.DataFrame(response.json())
    # Append predict to df
    df['predict'] = predict
    print(df)    

    # Increment counter
    counter += 1

    # Check if all rows are processed
    if counter == total_rows:
        print("All rows processed. Stopping the script.")
        exit()

    return response, df

clock_config = SystemClockConfig()
window_config = TumblingWindow(
    length=timedelta(seconds=2), align_to=datetime(2023, 1, 1, tzinfo=timezone.utc)
)

flow = Dataflow()
input_file_path = "/Users/rianrachmanto/pypro/project/Smoker-Detection-Bio-Signs/data/raw/test_smoking.csv"
total_rows = sum(1 for _ in open(input_file_path)) - 1  # Subtract 1 to exclude header
counter = 0  # Counter to track the number of processed rows

flow.input("inp", CSVInput(input_file_path))
flow.map(process_data)
flow.map(lambda row: predict((url, row), counter, total_rows))
flow.output("out", StdOutput())
