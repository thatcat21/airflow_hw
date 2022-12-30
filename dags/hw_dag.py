import datetime as dt
from datetime import datetime

import os
import sys
import logging
import dill
import pandas as pd

from airflow.models import DAG
from airflow.operators.python import PythonOperator

path = os.path.expanduser('~/airflow_hw')
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)

#from modules.pipeline import pipeline
# <YOUR_IMPORTS>

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df.drop(columns_to_drop, axis=1)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        bounds = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return bounds

    df = df.copy()
    boundaries = calculate_outliers(df['year'])
    df.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    def short_model(x):
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    df = df.copy()
    df.loc[:, 'short_model'] = df['model'].apply(short_model)
    df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df


def pipeline() -> None:
    df = pd.read_csv(f'{path}/data/train/homework.csv')

    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('outlier_remover', FunctionTransformer(remove_outliers)),
        ('feature_creator', FunctionTransformer(create_features)),
        ('column_transformer', column_transformer)
    ])

    models = [
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    ]

    best_score = .0
    best_pipe = None
    for model in models:

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        logging.info(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    logging.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

    best_pipe.fit(X, y)
    model_filename = f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'

    with open(model_filename, 'wb') as file:
        dill.dump(best_pipe, file)

    logging.info(f'Model is saved as {model_filename}')


def model_loading():
    path = os.path.expanduser('~/airflow_hw') + '/data/models'
    file_name = sorted(os.listdir(path))[-1]
    file_path = os.path.join(path, file_name)
    with open(file_path, 'rb') as file:
        object_to_load = dill.load(file)
    return object_to_load


def prediction_frame_creation(model):
    path = os.path.expanduser('~/airflow_hw') + '/data/train/homework.csv'
    df_start = pd.read_csv(path).drop(['price_category'], axis=1)

    path = os.path.expanduser('~/airflow_hw') + '/data/test'
    file_name = sorted(os.listdir(path))
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for i in file_name:
        file_path = os.path.join(path, i)
        df = pd.read_json(file_path, encoding='utf-8', orient='index').T
        df = df.loc[:, df_start.columns]
        predict = model.predict(df)[0]
        car_id = i[:i.find('.json')]
        df_pred.loc[len(df_pred.index)] = [car_id, predict]
    return df_pred


def csv_saving(df):
    path = os.path.expanduser('~/airflow_hw') + f'/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df.to_csv(path, encoding='utf-8', index=False)


def predict():
    model = model_loading()
    prediction = prediction_frame_creation(model)
    csv_saving(prediction)




with DAG(
        dag_id='car_price_prediction',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:
    fitting_pipeline = PythonOperator(
        task_id='fitting_pipeline',
        python_callable=pipeline,
        dag=dag
    )

    prediction_pipeline = PythonOperator(
        task_id='prediction_pipeline',
        python_callable=predict,
        dag=dag
    )

    fitting_pipeline >> prediction_pipeline
