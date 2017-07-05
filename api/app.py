#!/usr/bin/env python

import os
from flask import Flask, render_template, request, redirect, \
                    url_for, send_from_directory
from werkzeug import secure_filename
import time
import wrapper
from config import cfg

# Initialize the Flask application
app = Flask(__name__)

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in cfg.allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')


# Route that will process the detect signs request
@app.route('/faster_rcnn/detect/signs', methods=['POST'])
def detect_signs():
    # Get the name of the uploaded file
    file = request.files['file']
    CONF_THRESHOLD = float(request.form['conf_threshold'])
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        tic = time.clock()
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        path = os.path.join(cfg.upload_folder, filename)
        # Move the file form the temporal folder to the upload folder we setup
        file.save(path)
        wrapper.detect_signs(path, CONF_THRESHOLD)
        # Redirect the user to the resulting video route, which
        # will basicaly show on the browser the processed video
        toc = time.clock()
        print ('Processing took {:.3f}s'.format(toc-tic))
        return redirect(url_for('uploaded_file',
                                filename=filename))

# Route that will process the detect vehicle request
@app.route('/faster_rcnn/detect/vehicles', methods=['POST'])
def detect_vehicles():
    # Get the name of the uploaded file
    file = request.files['file']
    CONF_THRESHOLD = float(request.form['conf_threshold'])
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        
        tic = time.clock()
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        path = os.path.join(cfg.upload_folder, filename)
        # Move the file form the temporal folder to the upload folder we setup
        file.save(path)
        wrapper.detect_vehicles(path, CONF_THRESHOLD)
        # Redirect the user to the resulting video route, which
        # will basicaly show on the browser the processed video
        toc = time.clock()
        print ('Processing took {:.3f}s'.format(toc-tic))
        return redirect(url_for('uploaded_file',
                                filename=filename))

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(cfg.result_folder,
                               filename)

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("80"),
        debug=True
    )
