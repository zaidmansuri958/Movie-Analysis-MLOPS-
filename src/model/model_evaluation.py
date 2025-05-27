import mlflow.sklearn
import numpy as np
import pandas as pd 
import pickle 
import json 
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import os 
from src.logger import logging


mlflow.set_tracking_uri("file:///C:/Users/ZAID/Desktop/MLops/Movie-Analysis-MLOPS/mlruns")


def load_model(file_path):
    try:
        with open(file_path,'rb') as f:
            model=pickle.load(f)
        logging.info("Model loaded")
        return model 
    except FileNotFoundError:
        logging.error("File not found ",file_path)
        raise 
    except Exception as e:
        logging.error("Unexpected error occurred while loading the model")
        raise

def load_data(file_path):
    try:
        df=pd.read_csv(file_path)
        logging.info("Data loaded from the %s",file_path)
        return df 
    except pd.errors.ParserError as e:
        logging.error("Failed to load the data ",e)
        raise 
    except Exception as e:
        logging.error("Unexpected error occured while loading the data")
        raise 

def evaluate_model(clf,X_test,y_test):
    try:
        y_pred=clf.predict(X_test)
        y_pred_proba=clf.predict_proba(X_test)[:,1]

        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_proba)

        metrics={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }

        return metrics
    except Exception as e:
        logging.error("Error during model evaluation : %s",e)
        raise 

def save_metrics(metrics,filepath):
    try:
        with open(filepath,'w') as file:
            json.dump(metrics,file)
        logging.info("Metrics saved to %s",filepath)
    except Exception as e:
        logging.error("Error occured while saving the metrics")
        raise 

def save_model_info(run_id,model_path,file_path):
    try:
        model_info={'run_id':run_id,'model_path':model_path}
        with open(file_path,'w') as file:
            json.dump(model_info,file,indent=4)
        logging.debug("model info saved to %s",file_path)
    except Exception as e:
        logging.error("Error occurred while saving the model info")
        raise

def main():
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            clf=load_model("./models/model.pkl")
            test_data=load_data("./data/processed/test_bow.csv")

            X_test=test_data.iloc[:,:-1].values
            y_test=test_data.iloc[:,-1].values 

            metrics=evaluate_model(clf,X_test,y_test)

            save_metrics(metrics,'reports/metrics.json')

            for metric_name,metric_value in metrics.items():
                mlflow.log_metric(metric_name,metric_value)
            
            if hasattr(clf,'get_params'):
                params=clf.get_params()
                for param_name,param_value in params.items():
                    mlflow.log_param(param_name,param_value)
            
            mlflow.sklearn.log_model(clf,"model")
            
            save_model_info(run.info.run_id,"model",'reports/experiment_info.json')

            mlflow.log_artifact('reports/metrics.json')
        
        except Exception as e:
            logging.error("Failed to complete the model evaluation : %s",e)

if __name__ == "__main__":
    main()
