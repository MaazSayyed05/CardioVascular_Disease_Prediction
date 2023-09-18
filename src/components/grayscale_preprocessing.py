
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from glob import glob

import os,sys
from src.exception import CustomException
from src.logger import logging



class grayscale_ImagePreprocessingConfig:

    train_path      =  os.path.join('/config/workspace/CardioVascular_Disease_Prediction/notebooks/data/Lung_Cancer_cells', 'train')

    validation_path =  os.path.join('/config/workspace/CardioVascular_Disease_Prediction/notebooks/data/Lung_Cancer_cells', 'validation')

    test_path       =  os.path.join('/config/workspace/CardioVascular_Disease_Prediction/notebooks/data/Lung_Cancer_cells', 'test')


class grayscale_ImagePreprocessing:
    def __init__(self):
        self.image_preprocess_config = grayscale_ImagePreprocessingConfig()
    
    def get_grayscale_image_data(self):
        try:
            logging.info("Initiate grayscale Image Preprocessing Process.")
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


            # logging.info("Image Data Generator Initiated Successfully.")

            # Preprocess the data
            train_data_set = train_data_gen.flow_from_directory(
                self.image_preprocess_config.train_path,
                target_size=(224,224),
                batch_size=32,
                class_mode='categorical',
                color_mode='grayscale',
                shuffle=True
            )

            validation_data_set = validation_data_gen.flow_from_directory(
                self.image_preprocess_config.validation_path,
                target_size=(224,224),
                batch_size=32,
                class_mode='categorical',
                color_mode='grayscale',
                shuffle=True
            )

            test_data_set = test_data_gen.flow_from_directory(
                self.image_preprocess_config.test_path,
                target_size=(224,224),
                batch_size=32,
                class_mode='categorical',
                color_mode='grayscale',
                shuffle=True
            )

            logging.info("Preporcessing of grayscale Images  Successful.")

            return (
                train_data_set,
                validation_data_set,
                test_data_set
            )
        
        except Exception as e:
            logging.info("Error occured in grayscale Image Preprocessing.")
            raise CustomException(e, sys)





















