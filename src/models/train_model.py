from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import pickle
from src.data import explore_data


def __preprocess_data(df):
    """preprocessing data to used in fitting
    parameters:
        df: dataframe containing data to fit on
    Return:
        X: data features
        y: Labels
    """
    df = df.drop(['timestamp', 'sld', 'longest_word'], axis = 1)
    y = df.iloc[:,-1]
    X = df.drop(['Label'], axis = 1)
    with open('../../data/interim/data_used_for_training.csv', encoding='utf-8', mode='a') as f: # saving each record in CSV
        df.to_csv(f, index=False, header=f.tell()==0, line_terminator='\n')
    return X, y

def __randomforest_fit(df):
    """ fitting a randomforest model
    parameters:
        df: dataframe containing data to fit on
    Return:
        model_rf: model instance
    """
    X, y = __preprocess_data(df)
    model_rf = RandomForestClassifier(n_estimators = 2000, bootstrap = False, max_depth = None, 
                                   max_features = 'auto', random_state = 50)
    model_rf = model_rf.fit(X, y)
    return model_rf

def __catboost_fit(data, logger):
    """ fitting a catboostmodel model
    parameters:
        df: dataframe containing data to fit on
    Return:
        model_cb: model instance
    """
    logger.info('===preparing data for training..')
    X, y = __preprocess_data(data) # preprocessing
    logger.info('===Data prepared successfully')
    logger.info('===Training..')
    model_cb = CatBoostClassifier(iterations = 1000, learning_rate = 0.05, verbose = False, random_state = 50)
    model_cb.fit(X, y, verbose=False)
    return model_cb

def train_rf_model():
    """Initiating the model
    parameters:
        data: dataframe containing data to fit on
    output:
        pickled model
    """
    pkl_path = '../../models/rf_model.pkl'
    data =  pd.read_csv(f'../../data/external/training_dataset.csv') # reading the training data
    rf_model = __randomforest_fit(data) # training the model
    pickle.dump(rf_model, open(pkl_path, 'wb')) # pickling the model

def train_cb_model(logger):
    """Initiating the model
    parameters:
        data: dataframe containing data to fit on
    output:
        pickled model
    """
    pkl_path = '../../models/cb_model.pkl'
    logger.info('=== Reading training data')
    data =  pd.read_csv(f'../../data/external/training_dataset.csv') # reading the training data
    #explore_data.explore(data)
    cb_model = __catboost_fit(data, logger) # fitting the model
    logger.info('===Extracting important visualizations from data..')
    explore_data.explore(data, cb_model, logger) # exploring data and visualzie 
    pickle.dump(cb_model, open(pkl_path, 'wb')) # pickling the model
    logger.info('=== model saved and ready for impplementation')