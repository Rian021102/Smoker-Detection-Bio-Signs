# Report on Smoking Prediction using Machine Learning

## 1. Problem Statement:

Smoking is widely recognized as a significant contributor to various diseases and health issues, including heart disease, lung cancer, and respiratory problems. Unfortunately, many individuals who smoke are not forthcoming about their smoking habits when questioned by healthcare providers. This discrepancy between self-reported smoking status and actual smoking behavior poses a significant challenge for healthcare professionals. It can lead to misdiagnoses, inappropriate treatments, and missed opportunities for smoking cessation interventions. Therefore, there is a pressing need for a reliable method to assess an individual's smoking status accurately.

## 2. Objective:

The primary objective of this project is to develop a machine learning model that can predict whether an individual is a smoker or non-smoker based on their biometric data. By leveraging features such as blood pressure, glucose levels, cholesterol levels, and other relevant medical indicators, we aim to create a robust predictive model. This model will enable healthcare providers to make more informed decisions about patient care, even when patients are reluctant or dishonest about their smoking habits.

## 3. Methodology:

### Data Collection:
To build the machine learning model, we collected a diverse dataset from healthcare providers' records. This dataset included a wide range of biometric measurements, demographic information, and smoking status labels (smoker or non-smoker) for each patient. The dataset spanned a significant period to capture variations in patients' health and behaviors over time.

### Data Cleaning:
Data preprocessing is a critical step in ensuring the quality and reliability of the machine learning model. In this project, we conducted the following data cleaning activities:

- **Handling Missing Values:** We addressed missing values in the dataset through various techniques, including mean imputation and forward/backward filling. This step ensured that the model had complete data for analysis.

- **Outliers Detection and Removal:** Outliers can skew the model's predictions. To enhance the model's robustness, we employed the Isolation Forest algorithm to identify and remove outliers from the dataset.

### Model Training and Deployment:

#### 1. FastAPI Integration:
Once the machine learning model was trained and validated, we integrated it into a FastAPI application. FastAPI is a Python-based web framework that allows for easy and efficient serving of machine learning models through a RESTful API. This integration made the model accessible via HTTP requests, enabling healthcare providers to interact with it seamlessly.

#### 2. Docker Containerization:
To ensure that the entire application, including the model, dependencies, and server configuration, could be easily reproduced in various environments, we containerized the FastAPI application using Docker. Containerization enhances portability, consistency, and scalability of the application.

#### 3. Setting up AWS EC2 Instances:
To deploy the Dockerized FastAPI application and make the model accessible over the internet, we leveraged Amazon Web Services (AWS) Elastic Compute Cloud (EC2) instances. EC2 instances provide a scalable and reliable infrastructure for hosting web applications, ensuring that the model can handle a large number of requests while maintaining uptime and performance.

#### 4. CI/CD Using GitHub Workflow:
Continuous Integration and Continuous Deployment (CI/CD) pipelines automate the process of building, testing, and deploying software. In our project, we established a GitHub workflow that was triggered automatically whenever changes were pushed to the repository. This workflow included building and testing the Docker image and updating the AWS EC2 instance with the latest version of the application. This automated approach streamlined the development and deployment process, reducing the risk of human error.

#### 5. Access via HTTP Protocol:
With the model deployed on AWS EC2 instances and accessible through a FastAPI-powered RESTful API, healthcare providers and end-users could access the model via the HTTP protocol. This user-friendly and standardized approach allowed for easy integration into various healthcare systems and applications.

## Conclusion:

In conclusion, this comprehensive solution for predicting smoking status based on biometric data addresses the critical issue of underreporting of smoking habits among patients. By leveraging machine learning, data preprocessing, FastAPI integration, Docker containerization, AWS deployment, and a robust CI/CD pipeline, we have provided healthcare professionals with a powerful tool to assess smoking status accurately.

This project not only improves patient care and outcomes but also contributes to the larger goal of reducing the impact of smoking-related diseases on public health. By creating a reliable and accessible method for assessing smoking status, healthcare providers can take proactive measures to help individuals quit smoking and mitigate the associated health risks. This end-to-end machine learning solution represents a significant step toward more informed and effective healthcare interventions.

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
