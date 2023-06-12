# Smoker Detection Using Bio Signs
## Project Description
In this project I train machine learning model to predict if someone is whether or not a smoker using bio signs. We trained multiple models and Random Forest gives the best accuracy, hence the prediction is using Random Forest (we trained the pipeline with RF model and StandardScaler inside the pipeline)
## File Structures
1. smoker.py is source code. The code is writen in modular format so it will be easy to make it into separate modular files for examples (load_data.py, eda.py, handle_data.py, etc)
2. try_smoker_detection.py is file to test that the source code is indeed working. This is just for testing but I also add save model (in this case pipeline where there is random forest model and standard scaler function inside the pipeline)
3. smoker_detection_app.py is streamlit for model deployment. The idea is to predict a smoker based on their bio signs using machine learning. You just need load new dataset and predict the output based on the saved model pipeline.
All the three files above are located in folders **src**
I also include notebook called **Smoker_Status_Prediction.ipynb** which basically contains the analytics/experimentation work of this project, before refactoring the code inside the notebook to more readable code
## ML Workflow

```mermaid
flowchart TD
dataset[(Dataset)]-->sourcecode["smoker.py"]
sourcecode-->eda["EDA"]
eda-->handledata["Handle Data"]
handledata-->preprocess["Pre Process Data"]
preprocess-->trainmodel["Experimenting with Multiple Models"]
trainmodel-->randomforest["Retrain Random Forest"]
randomforest-->prediction["Predict"]
newdata[(New Data)]-->handledata
preprocess--newdata-->prediction
newdataset[(New Dataset)]-->sourcecode
preprocess--newdataset-->randomforest
 ```
