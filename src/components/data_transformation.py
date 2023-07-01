import os
import sys
import pandas as pd
import numpy as np
sys.path.append("./src")
from logger import logging
from exception import CustomException
from utils import save_object
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline

class Imputer(BaseEstimator, TransformerMixin):
    def fit(self, df):
        return self
    def transform(self, df):
        df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median())
        df['Delivery_person_Age'] = df['Delivery_person_Age'].fillna(df['Delivery_person_Age'].mean())
        df['multiple_deliveries'] = df['multiple_deliveries'].fillna(df['multiple_deliveries'].mode()[0])
        df['Weather_conditions'] = df['Weather_conditions'].fillna(df['Weather_conditions'].mode()[0])
        df['Road_traffic_density'] = df['Road_traffic_density'].fillna(df['Road_traffic_density'].mode()[0])
        df['Festival'] = df['Festival'].fillna(df['Festival'].mode()[0])
        df['City'] = df['City'].fillna(df['City'].mode()[0])
        return df

class drop_feature(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    def fit(self, df):
        return self
    def transform(self, df):
        for i in columns_to_drop:
            df.drop(i, axis=1, inplace=True)
        return df


class OrdinalEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, column_mappings):
        self.column_mappings = column_mappings
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for col, mapping in self.column_mappings.items():
            X[col] = X[col].map(mapping)
        return X

class OneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoder = None
    
    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse=False, drop='first').fit(X[self.columns])
        return self
    
    def transform(self, X, y=None):
        encoded = self.encoder.transform(X[self.columns])
        # Convert boolean values to integers (1 and 0)
        encoded = encoded.astype(int)
        
        # Create new column names
        new_columns = self.encoder.get_feature_names_out(self.columns)
        
        # Create a new DataFrame with the encoded features
        encoded_df = pd.DataFrame(encoded, columns=new_columns, index=X.index)
        
        # Drop the original columns from the input DataFrame
        X.drop(self.columns, axis=1, inplace=True)
        
        # Concatenate the original DataFrame with the encoded DataFrame
        X_encoded = pd.concat([X, encoded_df], axis=1)
        
        return X_encoded

class FeatureScaling(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_scaled = X.copy()
        scaler = StandardScaler()
        X_scaled[self.columns] = scaler.fit_transform(X[self.columns])
        return X_scaled

columns_to_drop=['ID','Delivery_person_ID','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Order_Date','Time_Orderd','Time_Order_picked','Type_of_order']

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated") 

            # Define the column mappings
            column_mappings = {
                'Road_traffic_density': {'Jam': 4, 'High': 3, 'Medium': 2, 'Low': 1},
                'Festival': {'No': 0, 'Yes': 1},
                'City': {'Metropolitian': 2, 'Urban': 1, 'Semi-Urban': 3}
            }

        

            # Define the pipeline
            pipe = Pipeline([
                ('Imputer',Imputer()),
                ('drop_feature',drop_feature(columns_to_drop)),
                ('ordinal_encoding', OrdinalEncoding(column_mappings)),
                ('one_hot_encoding', OneHotEncoding(columns=['Weather_conditions', 'Type_of_vehicle'])),
                ('feature_scaling', FeatureScaling(columns=['Delivery_person_Age', 'Delivery_person_Ratings', 'distance']))
            ])

            return pipe



        except Exception as error:
            logging.info("Error Occured in Data Transformation")
            raise CustomException(error, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation initiated")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read the train and test data")
            logging.info(f"Train Dataframe head: \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe head: \n{test_df.head().to_string()}")

            logging.info("Getting processing object")

            preprocessor_obj = self.get_data_transformation_object()
            logging.info("Data transformation object initiated")

            target_column = 'Time_taken (min)'
            drop_columns = [target_column]
            
            input_column_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_column_train_df = train_df[target_column]

            logging.info("train test splitted")

            input_column_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_column_test_df = test_df[target_column]

            logging.info("fitting to begin")
            input_column_train_arr = preprocessor_obj.fit_transform(input_column_train_df)
            input_column_test_arr = preprocessor_obj.transform(input_column_test_df)
            logging.info("fitting complete_abc")


            train_arr = np.c_[input_column_train_arr, np.array(target_column_train_df)]
            test_arr = np.c_[input_column_test_arr, np.array(target_column_test_df)]


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            logging.info("Data transformation completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as error:
            logging.info("Exception occured at initiation of data transformation")
            raise CustomException(error, sys)