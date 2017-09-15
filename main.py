import os
from flask import Flask
from api.detect import detection_api 
from api.config import api_config

app = Flask(__name__, template_folder=api_config.template_directory)
app.register_blueprint(detection_api, url_prefix='/detect')

app.run(host='0.0.0.0')
