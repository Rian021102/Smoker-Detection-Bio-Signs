# Report on Smoking Prediction using Machine Learning

## 1. Problem Statement:

Smoking is widely recognized as a significant contributor to various diseases and health issues, including heart disease, lung cancer, and respiratory problems. Unfortunately, many individuals who smoke are not forthcoming about their smoking habits when questioned by healthcare providers. This discrepancy between self-reported smoking status and actual smoking behavior poses a significant challenge for healthcare professionals. It can lead to misdiagnoses, inappropriate treatments, and missed opportunities for smoking cessation interventions. Therefore, there is a pressing need for a reliable method to assess an individual's smoking status accurately.

## 2. Objective:

The primary objective of this project is to develop a machine learning model that can predict whether an individual is a smoker or non-smoker based on their biometric data. By leveraging features such as blood pressure, glucose levels, cholesterol levels, and other relevant medical indicators, we aim to create a robust predictive model. This model will enable healthcare providers to make more informed decisions about patient care, even when patients are reluctant or dishonest about their smoking habits.

### 3. FastAPI Integration:
Once the machine learning model was trained and validated, we integrated it into a FastAPI application. FastAPI is a Python-based web framework that allows for easy and efficient serving of machine learning models through a RESTful API. This integration made the model accessible via HTTP requests, enabling healthcare providers to interact with it seamlessly.

### 4. Addition of Bytewax Data Streaminng Framework
Bytewax is an open source framwork that enables us to build highly scalable dataflows in a streaming or batch context. Simply put, bytewax is designed to build stream processing data pipeline. For this project, we first trained and deployed model served with Fastapi (app.py). You can run the Fastapi using command: uvicorn app:app --reload. Since the FastApi run locally you can access, http://127.0.0.1:8000/predict. Then you can run the bytewax (inf.py) with command: python -m bytewax.run inf:flow. This part is basically simulation of data processing streaming pipelines and prediction, which in this case is CSV file.



## ML Workflow

```mermaid
flowchart TD
dataset[(Dataset)]-->loaddata["loaddata.py"]
loaddata-->eda["eda.py"]
eda-->preprocess["preprocess.py"]
preprocess-->trainmodel["train_model.py"]
trainmodel-->predict["prediction_model.py"]
newdata["Test-Data"]-->predict


 ```
