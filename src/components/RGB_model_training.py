
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.applications.VGG16 import  VGG16
from keras.applications.ResNet50 import ResNet50
from keras.applications.InceptionV3 import IncepitonV3

import os,sys
from src.exception import CustomException
from src.logger import logging

from src.utils import model_compile_fit



class ModelTrainingConfig:
    model_path = os.path.join('/config/workspace/CardioVascular_Disease_Prediction/artifacts', 'RGB_model.h5')


class RGB_ModelTraining:

    def __init__(self):
        self.model_train_config = ModelTrainingConfig()

    
    def model_training(self, train_data_set, validation_data_set, test_data_set):

        try: 
            logging.info("Initiate RGB Image Model Training.")
            IMAGE_SIZE = [224,224]
            CHANNELS = [3]  # RGB 
            

            # Model Dict.
            # Define model fit, compile, define base_model  in utils

            base_models = {
                'VGG16'           : VGG16(input_shape=IMAGE_SIZE+CHANNELS, weights='imagenet', include_top=False),
                'RESNET50'        : ResNet50(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE+CHANNELS),
                'InceptionV3'     : InceptionV3(input_shape=IMAGE_SIZE+CHANNELS, weights='imagenet', include_top=False),
            }


            accuracy_score_list, loss_score_list, model_fit_list = model_compile_fit(base_models,train_data_set,validation_data_set,test_data_set)

            logging.info("Defining of Model, Fully Connected Layers, Conpilation and Fitting of Models Successful.")

            best_score_index = accuracy_score_list.index(max(accuracy_score_list))
            
            best_model_name = list(base_models.keys())[best_score_index]

            best_model = model_fit_list[best_score_index]

            best_model_loss = loss_score_list[best_score_index]

            best_model_accuracy = accuracy_score_list[best_score_index]

            logging.info(f"Best RGB Model: {best_model_name}\nModel Accuracy: {best_model_accuracy}\t    Model Loss: {best_model_loss}")
            
            # Save Model
            best_model.save(self.model_train_config.model_path)

            logging.info("Model Training Process and Model Saving Process Completed Succefully")


        except Exception as e:
            logging.info("Error occured in RGB Image Model Training Process.")
            raise CustomException(e, sys)







