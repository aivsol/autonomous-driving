import os
from flask import Flask
from api.detect import detection_api 
from api.config import cfg

app = Flask(__name__, template_folder=cfg.template_directory)
app.register_blueprint(detection_api, url_prefix='/detect')
