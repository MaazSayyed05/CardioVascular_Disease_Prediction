
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from glob import glob

import os,sys
from src.exception import CustomException
from src.logger import logging



class ImagePreprocessingConfig:

    train_path      =  os.path.join('/config/workspace/CardioVascular_Disease_Prediction/notebooks/data/Images_Dataset', 'train')

    validation_path =  os.path.join('/config/workspace/CardioVascular_Disease_Prediction/notebooks/data/Images_Dataset', 'validation')

    test_path       =  os.path.join('/config/workspace/CardioVascular_Disease_Prediction/notebooks/data/Images_Dataset', 'test')


class ImagePreprocessing:
    def __init__(self):
        self.image_preprocess_config = ImagePreprocessingConfig()
    
    def get_rgb_image_data(self):
        try:
            logging.info("Initiate Image Preprocessing Process.")
            # Preprocessing Part
            train_data_gen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            )

            validation_data_gen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            )

            test_data_gen = ImageDataGenerator(rescale=1./255)


            logging.info("Image Data Generators Initiated Successfully.")

            # Preprocess the data
            train_data_set = train_data_gen.flow_from_directory(
                self.image_preprocess_config.train_path,
                target_size=(224,224),
                batch_size=32,
                class_mode='categorical',
                color_mode='rgb',
                shuffle=True
            )

            validation_data_set = validation_data_gen.flow_from_directory(
                self.image_preprocess_config.validation_path,
                target_size=(224,224),
                batch_size=32,
                class_mode='categorical',
                color_mode='rgb',
                shuffle=True
            )

            test_data_set = test_data_gen.flow_from_directory(
                self.image_preprocess_config.test_path,
                target_size=(224,224),
                batch_size=32,
                class_mode='categorical',
                color_mode='rgb',
                shuffle=True
            )

            logging.info("Preporcessing of Images Completed Successful.")

            return (
                train_data_set,
                validation_data_set,
                test_data_set
            )
        
        except Exception as e:
            logging.info("Error occured in Image Preprocessing.")
            raise CustomException(e, sys)











