import os,sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import  accuracy_score,precision_score,classification_report,confusion_matrix,recall_score,f1_score
from dataclasses import dataclass
from src.utils import evaluate_models,save_obj


@dataclass
class ModelTrainingConfig:
    model_path = os.path.join('artifacts','model.pkl')

class ModelTraining:

    def __init__(self):
        self.model_training_config = ModelTrainingConfig()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Initiate Model Training Process.")

            X_train, X_test, y_train, y_test = train_arr[:,:-1], test_arr[:,:-1], train_arr[:,-1], test_arr[:,-1]
            models = {
            'Logistic Regression'      : LogisticRegression(),
            'Decision Tree'            : DecisionTreeClassifier(),
            'Naive Bayes'              : GaussianNB(),
            'Random Forest Classifier' : RandomForestClassifier()
            }

            accuracy_score_list, precision_score_list, recall_score_list, f1_score_list, model_train_list = evaluate_models(X_train,X_test,y_train,y_test,models)
            best_score_index = accuracy_score_list.index(max(accuracy_score_list))

            best_accuracy_score = accuracy_score_list[best_score_index]
                        
            model_precsison_score = precision_score_list[best_score_index]
                        
            model_recall_score = recall_score_list[best_score_index]
                        
            model_f1_score = f1_score_list[best_score_index]

            model_name = list(models.keys())[best_score_index]

            best_model = model_train_list[best_score_index]

            logging.info(f"Best Model:{model_name}\nAccuracy Score: {best_accuracy_score}\tPrecision Score: {model_precsison_score}\tRecall Score:{model_recall_score}\tF1 Score: {model_f1_score}")

            save_obj(
                file_path=self.model_training_config.model_path,
                obj=best_model
            )

            logging.info("Model Training Process Terminated Successfully.")

        except Exception as e:
            logging.info("Error occured in Model Training Process.")
            raise CustomException(e, sys)

        finally:
            pass




















