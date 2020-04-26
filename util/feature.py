import pandas as pd
import numpy as np
from datetime import datetime as dt
import uuid
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from toto_logger.logger import TotoLogger

logger = TotoLogger()

class FeatureEngineering: 

    def __init__(self): 
        pass
    
    def __dummizer(self, encoded_id, id_encoder):
        """
        This method creates the dummies when an encoder has already been defined
        This is usefull to maintain the same dummy features after the training
        Using a pd.get_dummies() wouldn't work after training (e.g. in the scoring process), because
        the available data differs from the training data, so the aliments can be different and using the pd.get_dummies()
        would end up having different features during scoring.

        This method takes the original aliments and creates the dummy columns
        """
        id_cols = list(map(lambda x : 'id_' + str(x), id_encoder.transform(id_encoder.classes_)))

        vals = [0 for x in range(len(id_cols))]

        vals[encoded_id] = 1

        return pd.Series(vals, index=id_cols)


    def do_for_predict(self, data, id_encoder, context): 
        """
        Engineers the features for the prediction

        Parameters
        ----------
        data (dict)
            Requires the following keys to be present: 
            - alimentId (str)
        """
        aliment_id = data['alimentId']

        if aliment_id not in id_encoder.classes_: 
            return pd.DataFrame()

        aliment_encoded_id = id_encoder.transform([aliment_id])[0]

        id_cols = list(map(lambda x : 'id_' + str(x), id_encoder.transform(id_encoder.classes_)))

        vals = [0 for x in range(len(id_cols))]
        vals[aliment_encoded_id] = 1

        features_df = pd.DataFrame([vals], columns=id_cols)

        return features_df

    def do_for_scoring(self, folder, raw_data_filename, context, model): 
        """
        Feature engineering to support the scoring process
        """        
        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Starting feature engineering'.format(context=context.process), 'info')

        # Load the data
        raw_data_df = pd.read_csv(raw_data_filename, index_col=0)
        raw_data_df.fillna(0, inplace=True)

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Starting with a raw data shape of {s}'.format(context=context.process, s=raw_data_df.shape), 'info')

        # 2. Encode the id 
        id_encoder = joblib.load(model.files['id-encoder'])

        # IMPORTANT!
        # Remove the aliments that have not been encoded in the training process
        raw_data_df = raw_data_df[raw_data_df['id'].isin(id_encoder.classes_)]

        # Encode the id
        raw_data_df['encoded_id'] = id_encoder.transform(raw_data_df['id'])

        # 3. Convert the encoded_id into dummies
        id_dummies = raw_data_df['encoded_id'].apply(self.__dummizer, args=[id_encoder])

        raw_data_df = pd.concat([raw_data_df, id_dummies], axis=1)
        
        # 4. Make amounts categorical!!
        raw_data_df = pd.concat([raw_data_df, pd.get_dummies(raw_data_df['amountGr'], prefix='amountGr')], axis=1)
        raw_data_df = pd.concat([raw_data_df, pd.get_dummies(raw_data_df['amountMl'], prefix='amountMl')], axis=1)
        raw_data_df = pd.concat([raw_data_df, pd.get_dummies(raw_data_df['amount'], prefix='amount')], axis=1)
        
        # 5. Drop useless columns
        raw_data_df.drop(columns=['name', 'date', 'id', 'time', 'encoded_id', 'amountGr', 'amountMl', 'amount'], inplace=True)
        
        raw_data_df.drop(columns=['amount_0.0', 'amountGr_0.0', 'amountMl_0.0'], inplace=True)
        
        # 6. Save to file
        features_filename = '{folder}/features.csv'.format(folder=folder);

        raw_data_df.to_csv(features_filename)

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Feature engineering completed. Features Shape: {r}'.format(context=context.process, r=raw_data_df.shape), 'info')

        # Return the file and the vectorizer
        return features_filename


    def do_for_train(self, folder, raw_data_filename, context): 
        """
        Engineers the features
        """
        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Starting feature engineering'.format(context=context.process), 'info')

        # Load the data
        raw_data_df = pd.read_csv(raw_data_filename, index_col=0)
        raw_data_df.fillna(0, inplace=True)

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Starting with a raw data shape of {s}'.format(context=context.process, s=raw_data_df.shape), 'info')

        # 2. Encode the id 
        id_encoder = LabelEncoder()
        id_encoder.fit(raw_data_df['id'])

        raw_data_df['encoded_id'] = id_encoder.transform(raw_data_df['id'])

        # 3. Convert the encoded_id into dummies
        raw_data_df = pd.concat([raw_data_df, pd.get_dummies(raw_data_df['encoded_id'], prefix='id')], axis=1)
        
        # 4. Make amounts categorical!!
        raw_data_df = pd.concat([raw_data_df, pd.get_dummies(raw_data_df['amountGr'], prefix='amountGr')], axis=1)
        raw_data_df = pd.concat([raw_data_df, pd.get_dummies(raw_data_df['amountMl'], prefix='amountMl')], axis=1)
        raw_data_df = pd.concat([raw_data_df, pd.get_dummies(raw_data_df['amount'], prefix='amount')], axis=1)
        
        # 5. Drop useless columns
        raw_data_df.drop(columns=['name', 'date', 'id', 'time', 'encoded_id', 'amountGr', 'amountMl', 'amount'], inplace=True)
        
        raw_data_df.drop(columns=['amount_0.0', 'amountGr_0.0', 'amountMl_0.0'], inplace=True)
        
        # 6. Save to file
        features_filename = '{folder}/features.csv'.format(folder=folder);

        raw_data_df.to_csv(features_filename)

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Feature engineering completed. Features Shape: {r}'.format(context=context.process, r=raw_data_df.shape), 'info')

        # Return the file and the vectorizer
        return (features_filename, id_encoder)

