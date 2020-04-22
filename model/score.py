import os
import uuid
import numpy as np
import pandas as pd
import joblib
from datetime import datetime as dt, timedelta

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from toto_logger.logger import TotoLogger

from util.history import HistoryDownloader
from util.feature import FeatureEngineering

from totoml.model import ModelScore

logger = TotoLogger()

class Scorer: 
    """
    This class does the batch scoring of the model
    """

    def __init__(self): 
        pass

    def score(self, model, context): 
        """
        Score the provided model 
        """
        scoring_period = (dt.today() - timedelta(days=90)).strftime("%Y%m%d")

        logger.compute(context.correlation_id, '[ {context} ] - Scoring model {m}.v{v}'.format(context=context.process, m=model.info['name'], v=model.info['version']), 'info')

        # Create the folder where to store all the data
        folder = "{tmp}/{model_name}/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], model_name=model.info['name'], fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)
        
        # 1. Download the data
        history_filename = HistoryDownloader(scoring_period).download(folder, context)

        # 2. Engineer features
        features_filename = FeatureEngineering().do_for_scoring(folder, history_filename, context, model)

        # 3. Score
        # Read the features
        features_df = pd.read_csv(features_filename, index_col=0)

        # Get the id_encoder
        id_encoder = joblib.load(model.files['id-encoder'])

        # Get he ids as a list
        ids = list(map(lambda x : 'id_' + str(x), id_encoder.transform(id_encoder.classes_).tolist()))

        # Extraction of X and y
        X = features_df[ids]
        Y = features_df.drop(columns=ids)

        trained_model = joblib.load(model.files['model'])

        Y_pred = trained_model.predict(X)

        # Score
        p1 = precision_score(Y, Y_pred, average='weighted')
        r1 = recall_score(Y, Y_pred, average='weighted')
        f1 = f1_score(Y, Y_pred, average='weighted')

        score = [
            {"name": "precision", "value" : p1},
            {"name": "recall", "value": r1},
            {"name": "f1", "value": f1}
        ]

        return ModelScore(score, [history_filename, features_filename])
