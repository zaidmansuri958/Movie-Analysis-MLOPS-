import os 
import re
import string 
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

config={
    "data_path":"notebooks/data.csv",
    "test_size":0.2,
    "mlflow_tracking_url":"file:///C:/Users/ZAID/Desktop/MLops/Movie-Analysis-MLOPS/mlruns",
    "experiment_name":"LoR Hyperparameter Tuning"
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
        df['sentiment']=df['sentiment'].replace({'negative':0,'positive':1})
        vectorizer=TfidfVectorizer()
        X=vectorizer.fit_transform(df["review"])
        y=df["sentiment"]
        return train_test_split(X,y,test_size=0.2,random_state=42),vectorizer
    except Exception as e:
        print(f"error while loading data {e}")
        raise 

def train_and_log_model(X_train,X_test,y_train,y_test,vectorizer):
    param_grid={
        "C":[0.1,1,10],
        "penalty":["l1","l2"],
        "solver":["liblinear"]
    }

    with mlflow.start_run():
        grid_search=GridSearchCV(LogisticRegression(),param_grid,cv=5,scoring="f1",n_jobs=-1)
        grid_search.fit(X_train,y_train)

        for params,mean_score,std_score in zip(grid_search.cv_results_["params"],grid_search.cv_results_["mean_test_score"],grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"LR with params : {params}",nested=True):
                model=LogisticRegression()
                model.fit(X_train,y_train)

                y_pred=model.predict(X_test)

                metrics={
                        "accuracy":accuracy_score(y_test,y_pred),
                        "precision":precision_score(y_test,y_pred),
                        "recall":recall_score(y_test,y_pred),
                        "f1":f1_score(y_test,y_pred),
                        "mean_cv_score":mean_score,
                        "std_cv_score":std_score
                        }
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(f"Params : {params} | Accuracy : {metrics['accuracy']} | precision : {metrics['precision']} | recall : {metrics['recall']}")
        

        best_params=grid_search.best_params_
        best_model=grid_search.best_estimator_
        best_f1=grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score",best_f1)
        mlflow.sklearn.log_model(best_model,"model")

        print(f"Best params : {best_params} | best f1 score {best_f1}")



if __name__ == "__main__":
    (X_train,X_test,y_train,y_test),vectorizer=load_data("notebooks/data.csv")
    train_and_log_model(X_train,X_test,y_train,y_test,vectorizer)