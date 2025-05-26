import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml 
from src.logger import logging


def load_data(filepath):
    try:
        df=pd.read_csv(filepath)
        logging.info("Data loaded from path")
        return df 
    except pd.errors.ParserError as e:
        logging.error("Error occured while parsing the csv ",e)
        raise 
    except Exception as e:
        logging.error("Error occured while reading the file")
        raise 

def train_model(X_train,y_train):
    try:
        clf=LogisticRegression(C=1,solver='liblinear',penalty='l1')
        clf.fit(X_train,y_train)
        logging.info("Model training completed")
        return clf 
    except Exception as e:
        logging.error("Error during model training ",e)
        raise 

def save_model(model,file_path):
    try:
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logging.info("Model saved successfully ")
    except Exception as e:
        logging.error("Error occurred while saving the model ")
        raise 

def main():
    try:
        train_data=load_data("./data/processed/train_bow.csv")
        X_train=train_data.iloc[:,:-1].values 
        y_train=train_data.iloc[:,-1].values 

        clf=train_model(X_train,y_train)

        save_model(clf,"models/model.pkl")

    except Exception as e:
        logging.error("Failed to complete the model building process")
        print(f"Error : {e}")

if __name__ == "__main__":
    main()