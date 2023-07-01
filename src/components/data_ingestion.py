import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
import math
from src.components.data_transformation import DataTransformation


## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df=pd.read_csv('https://raw.githubusercontent.com/ssharma2323/Swati_Assignment/main/FinalTrain.csv')
            logging.info("Dataset read as pandas Dataframe")

            logging.info("Process started of converting Longititude and Latitude into displacement of source and destination")

            def calculate_distance(row):
                try: 
                    lat1 = row['Restaurant_latitude']
                    lon1 = row['Restaurant_longitude']
                    lat2 = row['Delivery_location_latitude']
                    lon2 = row['Delivery_location_longitude']
                    
                    '''
                    Since for few locations where lat long is negative, Restaurant location was of sea which is not possible and if we make it 
                    positive then it is at feasible distance to delivery location, hence making the adjustment accordingly
                    '''
                    if lat1<0:
                        lat1=lat1*(-1)
                    if lon1<0:
                        lon1=lon1*(-1)
                    # Convert degrees to radians
                    lat1_rad = math.radians(lat1)
                    lon1_rad = math.radians(lon1)
                    lat2_rad = math.radians(lat2)
                    lon2_rad = math.radians(lon2)

                    # Radius of the Earth in kilometers
                    radius = 6371

                    # Calculate the differences between the coordinates
                    dlat = lat2_rad - lat1_rad
                    dlon = lon2_rad - lon1_rad

                    # Haversine formula
                    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2) * math.sin(dlon/2)
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    distance = radius * c

                    return distance
                except (TypeError, ValueError):
                    return None

            # Calculate distances and assign them to a new column
            df['distance'] = df.apply(calculate_distance, axis=1)


            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)





if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_data_path, test_data_path)

