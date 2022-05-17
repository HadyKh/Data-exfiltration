
import pandas as pd
import numpy as np
from src.visualization import visualize


def explore(data, model, logger):
    """ Running the visualize.py script
    Parameters:
        data: dataframe
        model: model fitted for the shap library
        logger: the identified logger to log the steps
    """
    logger.info('===Dataset visualization')
    visualize.visualize_dataset(data)
    logger.info('===Chaeck balance')
    visualize.check_balance(data)
    logger.info('===Chaeck correlation')
    visualize.correlation_map(data)
    logger.info('===feature importance')
    visualize.implementing_shap(data, model)
