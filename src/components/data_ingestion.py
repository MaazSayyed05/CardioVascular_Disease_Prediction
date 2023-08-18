import pandas as pd
import numpy as np
import os,sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts','raw.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')


class DataIngestion:
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
     

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiate Data Ingestion Process.")
            dataset_as_df = pd.read_csv(os.path.join('/config/workspace/notebooks/data','CVD_resampled_labeled.csv'))
            dataset_as_df = dataset_as_df.drop('Unnamed: 0',axis=1) 
            logging.info("Dataset read as dataframe.")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            dataset_as_df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Train-Test split of dataset.")
            train_set, test_set = train_test_split(dataset_as_df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion Process Terminated Successfully.")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Error in Data Ingestion Process.")
            raise CustomException(e, sys)



