import pandas as pd
import numpy as np
import os,sys

from src.exception import CustomException
from src.logger import logging

from src.utils import load_obj

# -------------------------------------------------------------
from keras.models import load_model
import tensorflow as tf
from tf.keras.preprocessing.utils import load_img


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,dataset):
        try:
            logging.info("Initiate Prediction Process.")
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

            model = load_obj(model_path)
            preprocessor = load_obj(preprocessor_path)
            logging.info("Model object and preprocessor object fetched successfully.")

            dataset_preprocessed = preprocessor.transform(dataset)

            pred = model.predict(dataset_preprocessed)

            logging.info("Prediction Process Terminated Successfully.")

            return pred
        
        except Exception as e:
            logging.info("Error occured in Prediciton Process.")
            raise CustomException(e, sys)
    

    def image_prediction(self,input_image_path):
        # path to ftech user input image
        # image preprocessing
        # load model
        # prediction
        # return results of prediction

        logging.info("Initiate Prediction Process.")
        image_model_path   = os.path.join('/config/workspace/CardioVascular_Disease_Prediction/artifacts','image_model.h5')

        # Load and preprocess the image
        img = image.load_img(input_image_path, target_size=(224, 224), color_mode='rgb')  # Adjust target_size as per your model
        image_array = image.img_to_array(img)
        image_dims = np.expand_dims(image_array, axis=0)
        image_preprocessed = image_dims / 255.0  # Rescale pixel values to [0,1]
        logging.info("User Input Image Preprocessed Successfully.")


        image_model = load_model(image_model_path)
        logging.info("CNN Model Loaded Successfully.")
        predicted_class_array = image_model.predict(image_preprocessed)
        predicted_image_class = np.argmax(predicted_class_array[0])
        class_labels_map = {
            0: 'class_0',
            1: 'class_1',
            2: 'class_2',
            3: 'class_3',
            4: 'class_4',
            5: 'class_5',
        }
        logging.info("Image Model Training Completed Successfully.")
        return class_labels_map[predicted_image_class]




# Height_(cm),Weight_(kg),BMI,Alcohol_Consumption,Fruit_Consumption,Green_Vegetables_Consumption,FriedPotato_Consumption,General_Health,Checkup,Exercise,Skin_Cancer,Other_Cancer,Depression,Diabetes,Arthritis,Sex,Age_Category,Smoking_History,Heart_Disease

# 0   General_Health                308854 non-null  object 
#  1   Checkup                       308854 non-null  object 
#  2   Exercise                      308854 non-null  object 
#  3   Heart_Disease                 308854 non-null  object 
#  4   Skin_Cancer                   308854 non-null  object 
#  5   Other_Cancer                  308854 non-null  object 
#  6   Depression                    308854 non-null  object 
#  7   Diabetes                      308854 non-null  object 
#  8   Arthritis                     308854 non-null  object 
#  9   Sex                           308854 non-null  object 
#  10  Age_Category                  308854 non-null  object 
#  11  Height_(cm)                   308854 non-null  int64  
#  12  Weight_(kg)                   308854 non-null  float64
#  13  BMI                           308854 non-null  float64
#  14  Smoking_History               308854 non-null  object 
#  15  Alcohol_Consumption           308854 non-null  int64  
#  16  Fruit_Consumption             308854 non-null  int64  
#  17  Green_Vegetables_Consumption  308854 non-null  int64  
#  18  FriedPotato_Consumption       308854 non-null  int64 


class CustomData:
    def __init__(self,
                Height_:float,
                Weight_:float,
                BMI:float,
                Alcohol_Consumption:float,
                Fruit_Consumption:float,
                Green_Vegetables_Consumption:float,
                FriedPotato_Consumption:float,
                General_Health:str,
                Checkup:str,
                Exercise:str,
                Skin_Cancer:str,
                Other_Cancer:str,
                Depression:str,
                Diabetes:str,
                Arthritis:str,
                Sex:str,
                Age_Category:str,
                Smoking_History:str,
                ):

        self.Height_ = Height_
        self.Weight_ = Weight_
        self.BMI = BMI
        self.Alcohol_Consumption = Alcohol_Consumption
        self.Fruit_Consumption = Fruit_Consumption
        self.Green_Vegetables_Consumption = Green_Vegetables_Consumption
        self.FriedPotato_Consumption = FriedPotato_Consumption
        self.General_Health = General_Health
        self.Checkup = Checkup
        self.Exercise = Exercise
        self.Skin_Cancer = Skin_Cancer
        self.Other_Cancer = Other_Cancer
        self.Depression = Depression
        self.Diabetes = Diabetes
        self.Arthritis = Arthritis
        self.Sex = Sex
        self.Age_Category = Age_Category
        self.Smoking_History = Smoking_History
    

    def get_data_as_dataframe(self):
        try:
            logging.info("Initiate Process to gather data as DataFrame.")
            # self.Heigth_ is input data from user.
            custom_data_input_dict = {
                'Height_(cm)'                       : [self.Height_],
                'Weight_(kg)'                       : [self.Weight_],
                'BMI'                               : [self.BMI],
                'Alcohol_Consumption'               : [self.Alcohol_Consumption],
                'Fruit_Consumption'                 : [self.Fruit_Consumption],
                'Green_Vegetables_Consumption'      : [self.Green_Vegetables_Consumption],
                'FriedPotato_Consumption'           : [self.FriedPotato_Consumption],
                'General_Health'                    : [self.General_Health],
                'Checkup'                           : [self.Checkup],
                'Exercise'                          : [self.Exercise],
                'Skin_Cancer'                       : [self.Skin_Cancer],
                'Other_Cancer'                      : [self.Other_Cancer],
                'Depression'                        : [self.Depression],
                'Diabetes'                          : [self.Diabetes],
                'Arthritis'                         : [self.Arthritis],
                'Sex'                               : [self.Sex],
                'Age_Category'                      : [self.Age_Category],
                'Smoking_History'                   : [self.Smoking_History]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info("Error in DataFrame Class.")
            raise CustomException(e,sys)







