import joblib
import pandas as pd
import numpy as np
import pickle

from util.feature import FeatureEngineering
from toto_logger.logger import TotoLogger

from totoml.model import ModelPrediction

logger = TotoLogger()

class Predictor: 

    def __init__(self): 
        pass

    def predict (self, model, context, data):
        """
        Predicts the list of foods that are most likely to be chosen at a specific time
        Requires the following data to be passed in data: "date" (%Y%m%d), "time" (HH:mm)
        """
        # Load relevant model files
        col_names = pickle.load(open(model.files['target-categories'], 'rb'))
        id_encoder = joblib.load(model.files['id-encoder'])

        # 1. Feature engineering
        features_df = FeatureEngineering().do_for_predict(data, id_encoder, context)

        if len(features_df) == 0: 
            return ModelPrediction(prediction={})

        # 2. Load model & other required files
        trained_model = joblib.load(model.files['model'])

        # 3. Predict
        Y = pd.DataFrame(trained_model.predict(features_df), columns=col_names)

        try: 
            idx_true = Y.loc[0].tolist().index(1)

            prediction = float(col_names[idx_true].split('_')[1])
            predicted_type = col_names[idx_true].split('_')[0]
            
            # Return the prediction
            # predicted_type is for example amountGr, amountMl, amount
            # prediction is the value (e.g. 200)
            return ModelPrediction(prediction={predicted_type: prediction})

        except ValueError:
            
            return ModelPrediction(prediction={})
