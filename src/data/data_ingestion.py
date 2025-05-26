import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.connections import s3_connection

def load_params(params_path):
    try:
        with open(params_path,"r") as file:
            params=yaml.safe_load(file)
        return params 
    except FileNotFoundError:
        logging.error('File not found : %s',params_path)
        raise 
    except yaml.YAMLError as e:
        logging.error("YAML error : %s",e)
        raise 
    except Exception as e:
        logging.error('Unexpected error : %s',e)
        raise 

def load_data(data_url):
    try:
        df=pd.read_csv(data_url)
        logging.info("data loaded from %s",data_url)
        return df 
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the csv file : %s",e)
        raise 
    except Exception as e:
        logging.error("Unexpected error occured while loading the data : %s",e)
        raise 


def preprocess_data(df):
    try:
        logging.info("pre-processing")
        final_df=df[df['sentiment'].isin(['positive','negative'])]
        final_df['sentiment']=final_df['sentiment'].replace({'positive':1,'negative':0})
        logging.info("Data preprocessing complete")
        return final_df
    except KeyError as e:
        logging.error("Missing column in the dataframe : %s",e)
        raise 
    except Exception as e:
        logging.error("Unexpected error during preprocessing : %s",e)
        raise 

def save_data(train_data,test_data,data_path):
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logging.debug("Train and test data saved to %s",raw_data_path)
    except Exception as e:
        logging.error("Unexpected error occured while saving the data : %s",e)
        raise 

def main():
    try:
        params=load_params("params.yml")
        test_size=params["data_ingestion"]["test_size"]
        df=load_data(data_url="https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv")
        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_path="./data")
    except Exception as e:
        logging.error("Failed to complete the data ingestion process ")
        print("Error : {e}")

if __name__ == "__main__":
    main()


