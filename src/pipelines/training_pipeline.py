import os,sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformaiton import DataTransformation
from src.components.model_trainer import ModelTraining
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
        


    except Exception as e:
        logging.info("Error occured in training Pipeline.")
        raise CustomException(e, sys)














