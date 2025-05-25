import os
import re
import string 
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse


config={
    "data_path":"notebooks/data.csv",
    "test_size":0.2,
    "mlflow_tracking_url":"file:///C:/Users/ZAID/Desktop/MLops/Movie-Analysis-MLOPS/mlruns",
    "experiment_name":"Bow Vs TFIDF"
}

mlflow.set_tracking_uri(config["mlflow_tracking_url"])
mlflow.set_experiment(config["experiment_name"])



def lemmatization(text):
    lemmatizer=WordNetLemmatizer()
    text=text.split()
    text=[lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words=set(stopwords.words("english"))
    text=[word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    text=''.join([char for char in text if not char.isdigit()])
    return text 

def lower_case(text):
    text=text.split()
    text=[word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    text=re.sub('[%s]' % re.escape(string.punctuation),' ',text)
    text=re.sub('\s+',' ',text).strip()
    return text 

def removing_urls(text):
    url_pattern=re.compile(r'https?://S+|www\.\S+')
    return url_pattern.sub(r'',text)

def normalize_text(df):
    try:
        df['review']=df['review'].apply(lower_case)
        df['review']=df['review'].apply(remove_stop_words)
        df['review']=df['review'].apply(removing_numbers)
        df['review']=df['review'].apply(removing_punctuations)
        df['review']=df['review'].apply(removing_urls)
        df['review']=df['review'].apply(lemmatization)
        return df 
    
    except Exception as e:
        print(f"Error during text normalization {e}")
        raise

def load_data(file_path):
    try:
        df=pd.read_csv(file_path)
        df=normalize_text(df)
        df=df[df['sentiment'].isin(['positive','negative'])]
        df['sentiment']=df['sentiment'].replace({'negative':0,'positive':1}).infer_objects(copy=True)
        return df 
    except Exception as e:
        print(f"error while loading data {e}")
        raise 


VECTORIZERS={
    'BoW':CountVectorizer(),
    'TF-IDF':TfidfVectorizer()
}

ALGORITHMS={
    'LogisticRegression':LogisticRegression(),
    'MultinomiaNB':MultinomialNB(),
    'XGBoost':XGBClassifier(),
    'RandomForest':RandomForestClassifier(),
    'GradientBoosting':GradientBoostingClassifier()
}

def train_and_evaluate(df):
    with mlflow.start_run(run_name="All experiments") as parent_run:
        for algo_name,algorithm in ALGORITHMS.items():
            for vec_name , vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}",nested=True) as child_run:
                    try:
                        X=vectorizer.fit_transform(df['review'])
                        y=df['sentiment']
                        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=config["test_size"],random_state=42)
                        mlflow.log_params({
                            "vectorizer":vec_name,
                            "algorithm":algo_name,
                            "test_size":config["test_size"]
                        })

                        model=algorithm
                        model.fit(X_train,y_train)

                        log_model_params(algo_name,model)

                        y_pred=model.predict(X_test)

                        metrics={
                            "accuracy":accuracy_score(y_test,y_pred),
                            "precision":precision_score(y_test,y_pred),
                            "recall":recall_score(y_test,y_pred),
                            "f1":f1_score(y_test,y_pred)
                        }

                        mlflow.log_metrics(metrics)

                        input_example=X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                        mlflow.sklearn.log_model(model,"model",input_example=input_example)

                        print(f"Algorithm {algo_name} , vectorizer:{vec_name}")
                        print(f"Metrics : {metrics}")

                    except Exception as e :
                        print(f"Error in trainin {algo_name} with {vec_name} : {e}")
                        mlflow.log_param("error",str(e))

def log_model_params(algo_name,model):
    params_to_log={}

    if algo_name == 'LogisticRegression':
        params_to_log["C"]=model.C
    elif algo_name == "MultinominalNB":
        params_to_log["alpha"]=model.alpha
    elif algo_name == "XGBoost":
        params_to_log["n_estimators"]=model.n_estimators
        params_to_log["learning_rate"]=model.learning_rate
    elif algo_name== "RandomForest":
        params_to_log["n_estimators"]=model.n_estimators
        params_to_log["max_depth"]=model.max_depth
    elif algo_name == "GradientBoosting":
        params_to_log["n_estimators"]=model.n_estimators
        params_to_log["learning_rate"]=model.learning_rate
        params_to_log["max_depth"]=model.max_depth

    mlflow.log_params(params_to_log)


if __name__ == "__main__":
    df=load_data(config["data_path"])
    train_and_evaluate(df)