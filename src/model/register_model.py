import json 
import mlflow 
import logging

import mlflow.tracking 
from src.logger import logging
import os



mlflow.set_tracking_uri("file:///C:/Users/ZAID/Desktop/MLops/Movie-Analysis-MLOPS/mlruns")


def load_model_info(file_path):
    try:
        with open(file_path,'r') as file:
            model_info=json.load(file)
        logging.debug("Model info loaded from %s",file_path)
        return model_info
    except FileNotFoundError:
        logging.error("File not found : %s",file_path)
        raise 
    except Exception as e:
        logging.error("Unexpected error occurred while loading the model %s",e)
        raise 

def register_model(model_name,model_info):
    try:
        model_uri=f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        model_version=mlflow.register_model(model_uri,model_name)

        client=mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logging.debug(f"Model {model_name} version {model_version.version} registered and transitioned to staging")
    except Exception as e:
        logging.error("Error during model registration %s",e)
        raise 

def main():
    try:
        model_info_path='reports/experiment_info.json'
        model_info=load_model_info(model_info_path)

        model_name="my_model"
        register_model(model_name,model_info)

    except Exception as e:
        logging.error("Failed to complete the model registration process %s",e)

if __name__ == "__main__":
    main()