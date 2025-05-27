import numpy as np
import pandas as pd
import os 
from sklearn.feature_extraction.text import CountVectorizer
import yaml 
from src.logger import logging
import pickle

def load_params(params_path):
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logging.debug("Parameters retreived from %s",params_path)
        return params
    
    except FileNotFoundError:
        logging.error("File not found %s",params_path)
        raise 
    except yaml.YAMLError as e:
        logging.error('YAML error : %s',e)
        raise 
    except Exception as e:
        logging.error("Unexpected error : %s",e)
        raise 

def load_data(file_path):
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        return df 
    except pd.errors.ParserWarning as e:
        logging.error("Failed to parse the csv file : %s",e)
        raise 
    except Exception as e:
        logging.error("Unexpected error occurred while loading the data : %s",e)
        raise

def apply_bow(train_data,test_data,max_features):
    try:
        logging.info("Applying BOW")
        vectorizer=CountVectorizer(max_features=max_features)

        X_train=train_data['review'].values
        y_train=train_data['sentiment'].values 
        X_test=test_data['review'].values
        y_test=test_data['sentiment'].values 

        X_train_Bow=vectorizer.fit_transform(X_train)
        X_test_Bow=vectorizer.transform(X_test)

        train_df=pd.DataFrame(X_train_Bow.toarray())
        train_df['label']=y_train

        test_df=pd.DataFrame(X_test_Bow.toarray())
        test_df['label']=y_test

        os.makedirs('models', exist_ok=True) 
        logging.debug("Saving vectorizer to ./models/vectorizer.pkl")

        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        logging.info('bag of worlds applied and data transformed')

        return train_df,test_df
    
    except Exception as e:
        logging.error("Error during bag of words transformation : %s",e)
        raise 

def save_data(df,file_path):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logging.info("Data saved to %s",file_path)
    except Exception as e:
        logging.error("Unexpected error occurred while saving the data : %s",e)
        raise 


def main():
    try:
        params=load_params('params.yaml')
        max_features=params['feature_engineering']['max_features']

        train_data=load_data('./data/interim/train_processed.csv')
        test_data=load_data('./data/interim/test_processed.csv')

        train_df,test_df=apply_bow(train_data,test_data,max_features)

        save_data(train_df,os.path.join("./data","processed","train_bow.csv"))
        save_data(test_df,os.path.join("./data","processed","test_bow.csv"))
    except Exception as e:
        logging.error("Failed to complete the feature engineering : %s",e)

if __name__ == "__main__":
    main()