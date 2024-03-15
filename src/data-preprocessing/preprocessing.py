import pandas as pd
import numpy as np

from src.data_preprocessing.utils import copy_df
from sklearn.preprocessing import LabelEncoder

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]

def drop_cols(df:pd.DataFrame):
    """Dropping unimportant columns"""

    logger.info('DROPPING unimportant COLUMNS...')
    columns_to_drop = ['Product_Category_3','User_ID','Product_ID','Gender','City_Category','Marital_Status']
    df = df.drop(columns_to_drop, axis = 1)
    
    return df


def replace_plus(df:pd.DataFrame):
    """Replacing '+' in 'Age' and 'Stay_In_Current_City_Years'"""

    logger.info("Replacing '+' in 'Age' and 'Stay_In_Current_City_Years...")
    df['Age'] = df['Age'].apply(lambda x : str(x).replace('55+', '55'))
    df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].apply(lambda x : str(x).replace('4+', '4'))


    return df


# def feature_encoding(df:pd.DataFrame):
#     """Label encoding three columns"""

#     logger.info('Label encoding three columns')
#     label_encoder_gender = LabelEncoder()
#     df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])

#     label_encoder_age = LabelEncoder()
#     df['Age'] = label_encoder_age.fit_transform(df['Age'])

#     label_encoder_city = LabelEncoder()
#     df['City_Category'] = label_encoder_city.fit_transform(df['City_Category'])
    

#     return df


def fixing_null_values(df:pd.DataFrame):
    """Fixing null values in Product_category_2"""

    logger.info('Fixing null values in Product_category_2...')
    df['Product_Category_2'].fillna(df['Product_Category_2'].median(), inplace = True)

    return df

def convert_to_correct_dtype(df:pd.DataFrame):
    df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype('int')
    return df

def data_preprocessing_pipeline(df:pd.DataFrame):

    df = df.pipe(copy_df)\
         .pipe(drop_cols)\
         .pipe(replace_plus)\
         .pipe(fixing_null_values)\
         .pipe(convert_to_correct_dtype)

    return df


