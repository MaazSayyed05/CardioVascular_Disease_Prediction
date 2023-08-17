import os,sys
import pickle
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import  LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import  accuracy_score,precision_score,classification_report,confusion_matrix,recall_score,f1_score

def save_obj(file_path,obj):
    try:
        logging.info("Object Saving Process Initiated.")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj, file)
        
        logging.info("Object Saving Process Terminated Successfully.")
    
    except Exception as e:
        logging.info("Error occured in Object Saving Process.")
        raise CustomException(e, sys)


def evaluate_models(X_train,X_test,y_train,y_test,models):
    try:
        logging.info("Initiate Evaluation of Models.")
        accuracy_score_list = []
        precision_score_list = []
        recall_score_list = []
        f1_score_list = []

        for i in range(len(list(models.keys()))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            accuracy_score_list.append(accuracy_score(y_pred=y_pred, y_true=y_test))
            precision_score_list.append(precision_score(y_pred=y_pred, y_true=y_test))
            recall_score_list.append(recall_score(y_pred=y_pred, y_true=y_test))
            f1_score_list.append(f1_score(y_pred=y_pred, y_true=y_test))

        logging.info("Evaluation of Models Terminated Successfully.")
        return (
            accuracy_score_list,
            precision_score_list,
            recall_score_list,
            f1_score_list
        )


    except Exception as e:
        logging.info("Error occured in Model Evaluation Process.")
        raise CustomException(e, sys)


def load_obj(file_path,obj):
    try:
        # logging.info("Object Loading Process Initiated.")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
        # logging.info("Object Saving Process Terminated Successfully.")
    
    except Exception as e:
        logging.info("Error occured in Object Saving Process.")
        raise CustomException(e, sys)



# def prediction_dataset_mapping(dataset):
#     try:
#         logging.info("Mapping of Catgerocal Features Initiat.")

#         General_Health_map =    {"Poor":0,'Fair':1,'Good':2,'Very Good':3,'Excellent':4} 
#         Checkup_map =           {'Never':0,'Within the past year':1,'Within the past 2 years':2,'Within the past 5 years':3,'5 or more years ago':4}
#         Exercise_map =          {'No':0,'Yes':1}
#         Skin_Cancer_map =       {'No':0,'Yes':1}
#         Other_Cancer_map =      {'No':0,'Yes':1}
#         Depression_map =        {'No':0,'Yes':1}
#         Diabetes_map =          {'No':0,'No, pre-diabetes or borderline diabetes':1,'Yes, but female told only during pregnancy':2,'Yes':3}
#         Arthritis_map =         {'No':0,'Yes':1}
#         Sex_map =               {'Male':0,'Female':1}
#         Age_Category_map =      {'18-24':0,'25-29':1,'30-34':2,'35-39':3,'40-44':4,'45-49':5,'50-54':6,'55-59':7,'60-64':8,'65-69':9,'70-74':10,'75-79':11,'80+':12}
#         Smoking_History_map =   {'No':0,'Yes':1}

#         cat_cols = [cols for cols in dataset.columns if dataset[cols].dtype == 'O']

#         dataset[cat_cols[0]] = dataset[cols[0]].map(General_Health_map)
#         dataset[cat_cols[1]] = dataset[cols[1]].map(Checkup_map)
#         dataset[cat_cols[2]] = dataset[cols[2]].map(Exercise_map)
#         dataset[cat_cols[3]] = dataset[cols[3]].map(Skin_Cancer_map)
#         dataset[cat_cols[4]] = dataset[cols[4]].map(Other_Cancer_map)
#         dataset[cat_cols[5]] = dataset[cols[5]].map(Depression_map)
#         dataset[cat_cols[6]] = dataset[cols[6]].map(Diabetes_map)
#         dataset[cat_cols[7]] = dataset[cols[7]].map(Arthritis_map)
#         dataset[cat_cols[8]] = dataset[cols[8]].map(Sex_map)
#         dataset[cat_cols[9]] = dataset[cols[9]].map(Age_Category_map)
#         dataset[cat_cols[10]] = dataset[cols[10]].map(Smoking_History_map)

#         logging.info("Mapping of Categorical Features Terminated Succesfully.")
#         return dataset
    
#     except Exception as e:
#         logging.info("Error occured in Mapping of Categorical Features.")
#         raise CustomException(e, sys)











