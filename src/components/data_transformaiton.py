import pandas as pd
import numpy as np
import os,sys
from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import  ColumnTransformer

import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass

from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    

    def get_data_transformation_object(self):
        try:
            logging.info("Initiate Data Tranformation Object Process.")
            
            categorical_features = ['General_Health','Checkup','Exercise','Skin_Cancer','Other_Cancer','Depression','Diabetes','Arthritis','Sex','Age_Category','Smoking_History']
            numerical_features   = ['Height_(cm)','Weight_(kg)','BMI','Alcohol_Consumption','Fruit_Consumption','Green_Vegetables_Consumption','FriedPotato_Consumption']

            General_Health_category = ['Poor','Fair','Good','Very Good','Excellent'] 
            Checkup_category = ['Never','Within the past year','Within the past 2 years','Within the past 5 years','5 or more years ago']
            Exercise_category = ['No','Yes']
            Skin_Cancer_category = ['No','Yes']
            Other_Cancer_category = ['No','Yes']
            Depression_category = ['No','Yes']
            Diabetes_category = ['No','No, pre-diabetes or borderline diabetes','Yes, but female told only during pregnancy','Yes']
            Arthritis_category = ['No','Yes']
            Sex_category = ['Male','Female']
            Age_Category_category = ['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80+']
            Smoking_History_category = ['No','Yes']

            num_pipeline = Pipeline(
            steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ]   
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OrdinalEncoder(categories=[General_Health_category,Checkup_category,Exercise_category,Skin_Cancer_category,Other_Cancer_category,Depression_category,Diabetes_category,Arthritis_category,Sex_category,Age_Category_category,Smoking_History_category])),
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_features),
                ('cat_pipeline',cat_pipeline,categorical_features)
            ])


            # save_obj(
            #     file_path=self.data_transformation_config.preprocessor_path,
            #     obj=preprocessor
            # )

            logging.info("Data Transformation Object Process Terminated Successfully.")

            return preprocessor

        
        except Exception as e:
            logging.info("Error in Data Tranformation Object Process.")
            raise CustomException(e, sys)
        
        finally:
            pass


    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info("Initiate Data Transformation Process.")
            train_data_set = pd.read_csv(train_data_path)
            test_data_set = pd.read_csv(test_data_path)
            logging.info("Successfully fetched train dataset and test dataset.")
 
            target_feature = 'Heart_Disease'

            input_feature_train_data = train_data_set.drop(target_feature,axis=1)
            input_feature_test_data  = test_data_set.drop(target_feature,axis=1)

            target_feature_train_data = train_data_set[[target_feature]]  # MAPPING
            target_feature_test_data  = test_data_set[[target_feature]]   # MAPPING

            target_catgeory = {'No':0,'Yes':1}
            target_feature_train_data[target_feature] = target_feature_train_data[target_feature].map(target_catgeory)
            target_feature_test_data[target_feature]  = target_feature_test_data[target_feature].map(target_catgeory)

            logging.info("Successfully Segregated Dependent and Independent features from train and test data.")

            preprocessor = self.get_data_transformation_object()
            input_feature_train_data_arr = preprocessor.fit_transform(input_feature_train_data)
            input_feature_test_data_arr  = preprocessor.transform(input_feature_test_data)
            save_obj(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessor
            )
            logging.info("Preprocessing of Dependent Features Completed Succesfully.")

            train_arr = np.c_[input_feature_train_data_arr,np.array(target_feature_train_data)]
            test_arr  = np.c_[input_feature_test_data_arr,np.array(target_feature_test_data)]

            logging.info("Data Tranformation Process Terminated Successfully.")

            return (
                train_arr,
                test_arr
            )
        
        except Exception as e:
            logging.info("Error occured in Data Transformation Process.")
            raise CustomException(e, sys)
        
        finally:
            pass








    



























