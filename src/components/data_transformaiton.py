import pandas as pd
import numpy as np
import os,sys
from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass

from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    num_preprocessor_path = os.path.join('artifacts','numerical_preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    

    def get_data_transformation_object(self):
        try:
            logging.info("Initiate Data Tranformation Object Process.")
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            save_obj(
                file_path=self.data_transformation_config.num_preprocessor_path,
                obj=num_pipeline
            )

            logging.info("Data Transformation Object Process Terminated Successfully.")

            return num_pipeline

        
        except Exception as e:
            logging.info("Error in Data Tranformation Object Process.")
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info("Initiate Data Transformation Process.")
            train_data_set = pd.read_csv(train_data_path)
            test_data_set = pd.read_csv(test_data_path)
            logging.info("Successfully fetched train dataset and test dataset.")
 
            target_feature = 'Heart_Disease'

            input_feature_train_data = train_data_set.drop(target_feature,axis=1)
            input_feature_test_data  = test_data_set.drop(target_feature,axis=1)

            target_feature_train_data = train_data_set[[target_fearure]]
            target_feature_test_data  = test_data_set[[target_feature]]
            logging.info("Successfully Segregated Dependent and Independent features from train and test data.")

            preprocessor = get_data_transformation_object()
            input_feature_train_data_arr = preprocessor.fit_transform(input_feature_train_data,columns=preprocessor.get_feature_names_out())
            input_feature_test_data_arr  = preprocessor.transform(input_feature_test_data,columns=preprocessor.get_feature_names_out())
            logging.info("Preprocessing of Dependent Features Completed Succesfully.")

            train_arr = np.c_[input_feature_train_data_arr,np.array(target_feature_train_data)]
            test_arr  = np.c_[input_feature_test_data_arr,np.aray(target_feature_test_data)]

            logging.info("Data Tranformation Process Terminated Successfully.")

            return (
                train_arr,
                test_arr
            )
        
        except Exception as e:
            logging.info("Error occured in Data Transformation Process.")
            raise CustomException(e, sys)








    



























