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






