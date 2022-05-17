# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from libcst import Continue
from src.features import build_features
from src.models import predict_model, train_model
import pandas as pd
from kafka import KafkaConsumer
from os.path import exists

field_names = ['raw_domain', 'timestamp', 'FQDN_count','subdomain_length',  'upper', 'lower','numeric','entropy', 'special', 'labels', 
                'labels_max', 'labels_average', 'longest_word', 'sld', 'len', 'subdomain', 'predicted label', 'confidence score']
# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(): #input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('===Checking models availability')
    #checking for the model existance
    file_exists = exists('../../models/cb_model.pkl')
    if file_exists: # checking if the pickled model is exist in the  specified path
        logger.info('===Your model is ready')
        Continue
    else: # if the model is not exist in the specified path it will train and pickle a new model
        logger.info('===Training catboost classifier..') 
        train_model.train_cb_model(logger) # training model
        logger.info('=== model trained successfuly')
    


    logger.info('===initializing consumer')
    #initializing the KafkaConsumer
    consumer = KafkaConsumer( 
        'ml-raw-dns',
        bootstrap_servers = ['localhost:9092'],
        auto_offset_reset = 'earliest',
        enable_auto_commit = False
    )
    logger.info('===Consumer successfully initialized')
    logger.info('===making final data set from raw data')
    #dataframe = pd.DataFrame()
    # looping over each record # Note: I did not specify a condition to stop the consumer, so it can run in a real environment (the model will always wait for new ingestions)
    for domain in consumer:
        dataframe = pd.DataFrame() # initializing dataframe to save the whole columns
        dataframe['raw_domain'] = [domain.value.decode('utf-8')] # saving the raw domain
        dataframe = make_dataset(domain.value.decode("utf-8"), domain.timestamp, dataframe, logger) #calling the make_dataset function which will return 14 columns
        logger.info('===Load Model & predict')
        dataframe['predicted label'], dataframe['confidence score'] = predict_model.predict_cb(dataframe, logger) # prediction and confidence score
        logger.info('===Successfully predicted')
        logger.info('===Writing CSV file\n')
        
        # saving each predicted record in CSV (Saving is record by record)
        with open('../../data/processed/final_output_dataset.csv', encoding='utf-8', mode='a') as f: 
            dataframe.to_csv(f, index=False, header=f.tell()==0, line_terminator='\n')
        logger.info('=== CSV file written')

    

#making dataset
def make_dataset(domain, timestamp, df, logger):
    """ returns a dataframe of of 15 columns of features
    Parameters:
        domain: domain name record
        timestamp: time stamp of 
    """
    logger.info('extract features..')
    df['FQDN_count'] = [build_features.count_char(domain)]
    df['subdomain_length'] = [build_features.subdomain_length(domain)]
    df['upper'] = [build_features.count_upper(domain)]
    df['lower'] = [build_features.count_lower(domain)]
    df['numeric'] = [build_features.count_digits(domain)]
    df['entropy'] = [build_features.entropy(domain)]
    df['special'] = [build_features.count_special_character(domain)]
    df['labels'] = [build_features.count_labels(domain)]
    df['labels_max'] = [build_features.labels_max(domain)]
    df['labels_average'] = [build_features.labels_average(domain)]
    df['longest_word'] = [build_features.longest_word(domain)]
    df['sld'] = [build_features.second_level_domain(domain)]
    df['len'] = [build_features.length(domain)]
    df['subdomain'] = [build_features.contain_subdomain(domain)]
    logger.info('Features extracted successfully')
    
    return df    
    
    
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
    
    
    
    

