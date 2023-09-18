
import os,sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformaiton import DataTransformation
from src.components.model_trainer import ModelTraining
# ----------------------------------------------------------------------------
from src.components.RGB_preprocessing import RGB_ImagePreprocessing
from src.components.RGB_model_training import RGB_ModelTraining
# ----------------------------------------------------------------------------
from src.components.grayscale_preprocessing import grayscale_ImagePreprocessing
from src.components.grayscale_model_training import grayscale_ModelTraining

from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np

if __name__ == '__main__':
    try:

        data_ingestion_obj = DataIngestion()
        train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()

        data_transformation_obj = DataTransformation()
        train_arr, test_arr = data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)

        model_trainer_obj = ModelTraining()
        model_trainer_obj.initiate_model_training(train_arr,test_arr)
        
#  ---------------------------------------------------------------------------------------------------------------

        RGB_image_obj = RGB_ImagePreprocessing()
        RGB_train_data_set, RGB_validation_data_set, RGB_test_data_set = RGB_image_obj.get_rgb_image_data()

        RGB_image_model_trainer_obj = RGB_ModelTraining()
        RGB_image_model_trainer_obj.rgb_model_training(RGB_train_data_set, RGB_validation_data_set, RGB_test_data_set)

# ------------------------------------------------------------------------------------------------------------------

        grayscale_image_obj = grayscale_ImagePreprocessing()
        grayscale_train_data_set, grayscale_validation_data_set, grayscale_test_data_set = grayscale_image_obj.get_grayscale_image_data()

        grayscale_image_model_trainer_obj = grayscale_ModelTraining()
        grayscale_image_model_trainer_obj.grayscale_model_training(grayscale_train_data_set, grayscale_validation_data_set, grayscale_test_data_set)


    except Exception as e:
        logging.info("Error occured in training Pipeline.")
        raise CustomException(e, sys)







