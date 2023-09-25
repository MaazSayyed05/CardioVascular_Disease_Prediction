
import os,sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformaiton import DataTransformation
from src.components.model_trainer import ModelTraining
# ----------------------------------------------------------------------------
from src.components.RGB_preprocessing import ImagePreprocessing
from src.components.RGB_model_training import Image_ModelTraining

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
#  ---------------------------------------------------------------------------------------------------------------
#  ---------------------------------------------------------------------------------------------------------------

        
        image_obj = ImagePreprocessing()
        train_data_set, validation_data_set, test_data_set = image_obj.get_rgb_image_data()

        image_model_trainer_obj = Image_ModelTraining()
        image_model_trainer_obj.model_training(train_data_set, validation_data_set, test_data_set)


    except Exception as e:
        logging.info("Error occured in training Pipeline.")
        raise CustomException(e, sys)







