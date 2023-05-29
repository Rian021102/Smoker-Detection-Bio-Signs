# Smoker Detection Using Bio Signs
## Project Description
In this project, I make prediction whether or not a patient is a smoker using his/her bio signs. The goal of this project is to provide a prediction model to assist physician to prepare an intervention program for smoker.
## Folder Structure as Follow
.
├── API
│   └── python_api.py
├── README.md
├── __pycache__
├── data
│   ├── processed
│   └── raw
│       ├── test_smoking.csv
│       └── train_smokin.csv
├── docs
│   └── conf.py
├── folder_structure.txt
├── models
│   └── model.joblib
├── notebooks
│   ├── README.md
│   ├── Smoker_Status_Prediction.ipynb
│   ├── test_environment.py
│   └── tox.ini
├── random_forest_pipeline.joblib
├── reports
│   ├── Rangkuman_Project.pdf
│   └── figures
├── requirements.txt
├── setup.py
├── smoker_detection_app.py
├── src
│   ├── README.md
│   ├── __init__.py
│   ├── app.py
│   ├── docs
│   │   ├── Makefile
│   │   ├── commands.rst
│   │   ├── conf.py
│   │   ├── getting-started.rst
│   │   ├── index.rst
│   │   └── make.bat
│   ├── eda.py
│   ├── loaddata.py
│   ├── predict_model.py
│   ├── preprocess.py
│   ├── requirements.txt
│   ├── setup.py
│   ├── test_environment.py
│   ├── tox.ini
│   └── train_model.py
├── src.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   └── top_level.txt
├── test_environment.py
└── tox.ini
## Source Code
The source codes are located in smoker-detection/src, you will find
### loaddata.py
This module to load, inspect and split the data into train and validation set
### eda.py
This module is to perform exploratary data analysis
### preprocess.py
This module is to perform data cleaning (imputing), outliers removal, oversampling for imbalanced data and data scalling
### train_model.py
This module is to train the module. I train randomforest for this project
### app.py
This module is to test all the modules

## ML Workflow

```mermaid
flowchart TD
dataset[(Dataset)]-->loaddata["loaddata.py"]
loaddata-->eda["eda.py"]
eda-->preprocess["preprocess.py"]
preprocess-->trainmodel["train_model.py"]
trainmodel-->predict["build-prediction using trained model"]


 ```
