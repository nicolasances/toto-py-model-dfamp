from flask import Flask
from model.dfamp import DFAMP

from totoml.controller import ModelController
from totoml.config import ControllerConfig

app = Flask(__name__)

model_controller = ModelController(DFAMP(), app, ControllerConfig(enable_batch_predictions_events=False, enable_single_prediction_events=False))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)