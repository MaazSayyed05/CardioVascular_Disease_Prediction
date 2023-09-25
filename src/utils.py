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
# -----------------------------------------------------------------------------------

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.applications import  VGG16
from keras.applications import ResNet50
from keras.applications import InceptionV3



def save_obj(file_path,obj):
    try:
        logging.info("Object Saving Process Initiated.")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj, file_obj)
        
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
        model_train_list = []

        for i in range(len(list(models.keys()))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)


            model_train_list.append(model)
            accuracy_score_list.append(accuracy_score(y_pred=y_pred, y_true=y_test))
            precision_score_list.append(precision_score(y_pred=y_pred, y_true=y_test))
            recall_score_list.append(recall_score(y_pred=y_pred, y_true=y_test))
            f1_score_list.append(f1_score(y_pred=y_pred, y_true=y_test))

        logging.info("Evaluation of Models Terminated Successfully.")
        return (
            accuracy_score_list,
            precision_score_list,
            recall_score_list,
            f1_score_list,
            model_train_list
        )


    except Exception as e:
        logging.info("Error occured in Model Evaluation Process.")
        raise CustomException(e, sys)


def load_obj(file_path):
    try:
        # logging.info("Object Loading Process Initiated.")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
        # logging.info("Object Saving Process Terminated Successfully.")
    
    except Exception as e:
        logging.info("Error occured in Object Saving Process.")
        raise CustomException(e, sys)



def prediction_dataset_mapping(dataset):
    try:
        logging.info("Mapping of Catgerocal Features Initiat.")

        General_Health_map =    {"Poor":0,'Fair':1,'Good':2,'Very Good':3,'Excellent':4} 
        Checkup_map =           {'Never':0,'Within the past year':1,'Within the past 2 years':2,'Within the past 5 years':3,'5 or more years ago':4}
        Exercise_map =          {'No':0,'Yes':1}
        Skin_Cancer_map =       {'No':0,'Yes':1}
        Other_Cancer_map =      {'No':0,'Yes':1}
        Depression_map =        {'No':0,'Yes':1}
        Diabetes_map =          {'No':0,'No, pre-diabetes or borderline diabetes':1,'Yes, but female told only during pregnancy':2,'Yes':3}
        Arthritis_map =         {'No':0,'Yes':1}
        Sex_map =               {'Male':0,'Female':1}
        Age_Category_map =      {'18-24':0,'25-29':1,'30-34':2,'35-39':3,'40-44':4,'45-49':5,'50-54':6,'55-59':7,'60-64':8,'65-69':9,'70-74':10,'75-79':11,'80+':12}
        Smoking_History_map =   {'No':0,'Yes':1}

        cat_cols = [cols for cols in dataset.columns if dataset[cols].dtype == 'O']

        dataset['General_Health'] = dataset['General_Health'].map(General_Health_map)
        dataset['Checkup'] = dataset['Checkup'].map(Checkup_map)
        dataset['Exercise'] = dataset['Exercise'].map(Exercise_map)
        dataset['Skin_Cancer'] = dataset['Skin_Cancer'].map(Skin_Cancer_map)
        dataset['Other_Cancer'] = dataset['Other_Cancer'].map(Other_Cancer_map)
        dataset['Depression'] = dataset['Depression'].map(Depression_map)
        dataset['Diabetes'] = dataset['Diabetes'].map(Diabetes_map)
        dataset['Arthritis'] = dataset['Arthritis'].map(Arthritis_map)
        dataset['Sex'] = dataset['Sex'].map(Sex_map)
        dataset['Age_Category'] = dataset['Age_Category'].map(Age_Category_map)
        dataset['Smoking_History'] = dataset['Smoking_History'].map(Smoking_History_map)

        logging.info("Mapping of Categorical Features Terminated Succesfully.")
        return dataset
    
    except Exception as e:
        logging.info("Error occured in Mapping of Categorical Features.")
        raise CustomException(e, sys)






# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

def model_compile_fit(base_models,train_data_set,validation_data_set,test_data_set):
    
        # accuracy_score_list = []
        # loss_score_list = []
        # model_fit_list = []

        # num_output_class = len(train_data_set.class_indices)

        # for base_model in list(base_models.values()):
        #     model = Sequential()
        #     model.add(base_model)
            
        #     # Add Fully Connected Layers
        #     model.add(Flatten())
        #     model.add(Dense(units=256,activation='relu'))
        #     model.add(Dropout(0.4))
        #     model.add(Dense(units=num_output_class,activation='softmax'))

        #     # Compile the Model
        #     model.compile(
        #         loss='categorical_crossentropy',
        #         optimizer='adam',
        #         metrics=['accuracy']
        #     )

        #     model.fit(
        #         train_data_set,
        #         validation_data=validation_data_set,
        #         epochs=7,
        #         steps_per_epoch=len(train_data_set),
        #         validation_steps= len(validation_data_set)
        #     )

        #     model_fit_list.append(model)

        #     # logging.info(Add a dataframe to show each model evaluation using pd.DataFrame(model.history) [loss, acc, val_lss, val_acc])



        #     # Make Predictions
        #     # predictions = model.predict(test_data_set)

        #     # Calculate Accuracy and Loss
        #     # true_labels = test_data_set.classes
        #     # predicted_labels = np.argmax(predictions, axis=1)


        #     # Calculate accuracy(Since we have not used One-Hot Encoder)
        #     # If you're using categorical cross-entropy loss during training

        #     loss, accuracy = model.evaluate(test_data_set)

        #     # print(f'Loss: {loss[0]}')
        #     # accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)
        #     # print(f'Accuracy: {accuracy}')

        #     accuracy_score_list.append(accuracy)
        #     loss_score_list.append(loss)

        # # ------------------------------------------------

        #     # To determine which class is given which label
        #     # class_indices = train_generator.class_indices
        #     # class_indices 


        #     # class_mapping = {
        #     #     0: 'type1',
        #     #     1: 'type2',
        #     #     2: 'type3',
        #     #     3: 'type4',
        #     #     4: 'type5'
        #     # }

        #     # # Assuming 'predicted_labels' is the output of your model
        #     # predicted_class_names = [class_mapping[label] for label in predicted_labels]

        # # -----------------------------------------------------


        
        # return (
        #     accuracy_score_list,
        #     loss_score_list,
        #     model_fit_list
        # )


    accuracy_score_list = []
    loss_score_list = []
    model_fit_list = []
    model_history_list = []

    num_output_class = len(train_data_set.class_indices)

    for base_model in list(base_models.values()):

        for layer in base_model.layers:
            layer.trainable = False
            
        model = Sequential()
        model.add(base_model)

        # Add Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(units=256,activation='relu'))
        model.add(Dropout(0.40))
        model.add(Dense(units=num_output_class,activation='softmax'))

        # Compile the Model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        model.fit(
            train_data_set,
            validation_data=validation_data_set,
            epochs=25,
            steps_per_epoch=len(train_data_set),
            validation_steps= len(validation_data_set)
        )

        model_fit_list.append(model)
        model_history_list.append(model.history)

        # logging.info(Add a dataframe to show each model evaluation using pd.DataFrame(model.history) [loss, acc, val_lss, val_acc])



        # Make Predictions
        # predictions = model.predict(test_data_set)

        # Calculate Accuracy and Loss
        # true_labels = test_data_set.classes
        # predicted_labels = np.argmax(predictions, axis=1)


        # Calculate accuracy(Since we have not used One-Hot Encoder)
        # If you're using categorical cross-entropy loss during training

        loss, accuracy = model.evaluate(test_data_set)

        # print(f'Loss: {loss[0]}')
        # accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)
        # print(f'Accuracy: {accuracy}')

        accuracy_score_list.append(accuracy)
        loss_score_list.append(loss)

    # ------------------------------------------------

        # To determine which class is given which label
        # class_indices = train_generator.class_indices
        # class_indices


        # class_mapping = {
        #     0: 'type1',
        #     1: 'type2',
        #     2: 'type3',
        #     3: 'type4',
        #     4: 'type5'
        # }

        # # Assuming 'predicted_labels' is the output of your model
        # predicted_class_names = [class_mapping[label] for label in predicted_labels]

    # -----------------------------------------------------



    return (
        accuracy_score_list,
        loss_score_list,
        model_fit_list,
        model_history_list
    )

















































