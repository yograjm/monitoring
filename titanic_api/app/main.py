import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio
from fastapi import FastAPI, Request, Response

import random
import numpy as np
import pandas as pd
from titanic_model.processing.data_manager import load_pipeline, pre_pipeline_preparation, load_dataset
from titanic_model import __version__ as _version
from titanic_model.config.core import config
from sklearn.model_selection import train_test_split
from titanic_model.predict import make_prediction

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# FastAPI object
app = FastAPI()


################################# Prometheus related code START ######################################################
import prometheus_client as prom

acc_metric = prom.Gauge('titanic_accuracy_score', 'Accuracy score for few random 100 test samples')
f1_metric = prom.Gauge('titanic_f1_score', 'F1 score for few random 100 test samples')
precision_metric = prom.Gauge('titanic_precision_score', 'Precision score for few random 100 test samples')
recall_metric = prom.Gauge('titanic_recall_score', 'Recall score for few random 100 test samples')

# Load trained pipeline
pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
titanic_pipe= load_pipeline(file_name=pipeline_file_name)

# LOAD TEST DATA
import psycopg2
from psycopg2 import sql

# Database connection parameters
db_params = {
    'dbname': 'storedb',
    'user': 'postgres',
    'password': 'mypassword',
    'host': 'add-public-IP-here',  # EC2 public IP # or your database host
    'port': '5432'  #'5432'        # default PostgreSQL port
}
# Connect to the PostgreSQL database
try:
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Read existing data from the table
    query = "SELECT * FROM titanic;"  # SQL query to select all data from the table
    cursor.execute(query)

    # Fetch all results
    rows = cursor.fetchall()

    # Get column names from the cursor
    column_names = [desc[0] for desc in cursor.description]

    # Create a DataFrame from the fetched data
    df = pd.DataFrame(rows, columns=column_names)
    df.columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    print(f"Existing rows in db: {len(df)}")

    # Display the DataFrame
    #print(df.head())

except Exception as e:
    print(f"An error occurred while getting data from DB: {e}")

finally:
    # Close the cursor and connection
    if cursor:
        cursor.close()
    if conn:
        conn.close()

# Preprocess Test Data
test_data = pre_pipeline_preparation(data_frame = df) 

# Reorder columns
new_order = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Has_cabin', 'Title', 'Survived']
test_data = test_data[new_order]


# Function for updating metrics
def update_metrics():
    global test_data
    # Performance on test set
    test = test_data.copy()
    y_pred = titanic_pipe.predict(test.drop(['Survived'], axis=1))     # prediction
    acc = accuracy_score(test['Survived'], y_pred)                     # accuracy score
    f1 = f1_score(test['Survived'], y_pred)                            # F1 score
    precision = precision_score(test['Survived'], y_pred)              # Precision score
    recall = recall_score(test['Survived'], y_pred)                    # Recall score
    
    acc_metric.set(round(acc, 3))
    f1_metric.set(round(f1, 3))
    precision_metric.set(round(precision, 3))
    recall_metric.set(round(recall, 3))


@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())

################################# Prometheus related code END ######################################################


# UI - Input components
in_Pid = gradio.Textbox(lines=1, placeholder=None, value="79", label='Passenger Id')
in_Pclass = gradio.Radio(['1', '2', '3'], type="value", label='Passenger class')
in_Pname = gradio.Textbox(lines=1, placeholder=None, value="Caldwell, Master. Alden Gates", label='Passenger Name')
in_sex = gradio.Radio(["Male", "Female"], type="value", label='Gender')
in_age = gradio.Textbox(lines=1, placeholder=None, value="14", label='Age of the passenger in yrs')
in_sibsp = gradio.Textbox(lines=1, placeholder=None, value="0", label='No. of siblings/spouse of the passenger aboard')
in_parch = gradio.Textbox(lines=1, placeholder=None, value="2", label='No. of parents/children of the passenger aboard')
in_ticket = gradio.Textbox(lines=1, placeholder=None, value="248738", label='Ticket number')
in_cabin = gradio.Textbox(lines=1, placeholder=None, value="A5", label='Cabin number')
in_embarked = gradio.Radio(["Southampton", "Cherbourg", "Queenstown"], type="value", label='Port of Embarkation')
in_fare = gradio.Textbox(lines=1, placeholder=None, value="29", label='Passenger fare')

# UI - Output component
out_label = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox")

# Label prediction function
def get_output_label(in_Pid, in_Pclass, in_Pname, in_sex, in_age, in_sibsp, in_parch, in_ticket, in_cabin, in_embarked, in_fare):
    
    input_df = pd.DataFrame({"PassengerId": [in_Pid], 
                             "Pclass": [int(in_Pclass)], 
                             "Name": [in_Pname],
                             "Sex": [in_sex.lower()], 
                             "Age": [float(in_age)], 
                             "SibSp": [int(in_sibsp)],
                             "Parch": [int(in_parch)], 
                             "Ticket": [in_ticket], 
                             "Cabin": [in_cabin],
                             "Embarked": [in_embarked[0]], 
                             "Fare": [float(in_fare)]})
    
    result = make_prediction(input_data=input_df.replace({np.nan: None}))["predictions"]
    label = "Survived" if result[0]==1 else "Not Survived"
    return label


# Create Gradio interface object
iface = gradio.Interface(fn = get_output_label,
                         inputs = [in_Pid, in_Pclass, in_Pname, in_sex, in_age, in_sibsp, in_parch, in_ticket, in_cabin, in_embarked, in_fare],
                         outputs = [out_label],
                         title="Titanic Survival Prediction API  ⛴",
                         description="Predictive model that answers the question: “What sort of people were more likely to survive?”",
                         flagging_mode='never',
                         )

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
