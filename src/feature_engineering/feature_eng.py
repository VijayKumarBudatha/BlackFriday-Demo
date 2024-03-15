import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

import logging 
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]



def feature_encoding(df:pd.DataFrame):
    """Label encoding three columns"""

    logger.info('Label encoding three columns')
    label_encoder_gender = LabelEncoder()
    df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])

    label_encoder_age = LabelEncoder()
    df['Age'] = label_encoder_age.fit_transform(df['Age'])

    label_encoder_city = LabelEncoder()
    df['City_Category'] = label_encoder_city.fit_transform(df['City_Category'])
    

    return df