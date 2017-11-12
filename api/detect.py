#!/usr/bin/env python
import _init
import os
from flask import render_template, request, redirect, \
                    url_for, send_from_directory, Blueprint, jsonify
from werkzeug import secure_filename
import time
import classes

# from faster_rcnn import FasterRCNN
from tf_faster_rcnn import TFFasterRCNN
from config import api_config
# from FCRN_depth import FCRNDepth
#from monocular_depth import MonocularDepth
detection_api = Blueprint('detection_api', __name__)


#FCRNDepth_estimator = FCRNDepth(api_config.FCRN_depth_model_path, api_config.FCRN_depth_weights_path)
#monoculardepth_estimator = MonocularDepth(api_config.monocular_depth_model_path, api_config.monocular_depth_weights_path)

if api_config.vehicle_framework == "TF":
    vehicle_detector = TFFasterRCNN('vehicle',
                                    api_config.vehicle_net,
                                    api_config.vehicle_tfmodel,
                                    classes.VOC_CLASSES)

elif api_config.vehicle_framework == "CAFFE":
    vehicle_detector = FasterRCNN(api_config.vehicle_prototxt,
                                  api_config.vehicle_caffemodel,
                                  classes.VOC_CLASSES,
                                  api_config.cpu_mode)
else:
    raise ValueError("Only TF and CAFFE implementations supported")

# if api_config.sign_framework == "TF":
#     sign_detector = TFFasterRCNN('sign',
#                                  api_config.sign_net,
#                                  api_config.sign_tfmodel,
#                                  classes.SIGNS_CLASSES)
#
# elif api_config.sign_framework == "CAFFE":
#     sign_detector = FasterRCNN(api_config.sign_prototxt,
#                                api_config.sign_caffemodel,
#                                classes.SIGNS_CLASSES,
#                                api_config.cpu_mode)
# else:
#     raise ValueError("Only TF and CAFFE implementations supported")

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in api_config.allowed_extensions


@detection_api.route('/')
def index():
    return render_template('index.html')


# Route that will process the detect signs request
@detection_api.route('/faster_rcnn/signs', methods=['POST'])
def detect_signs():
    # Get the name of the uploaded file
    file = request.files['file']
    CONF_THRESHOLD = float(request.form['conf_threshold'])
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        tic = time.clock()
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        path = os.path.join(api_config.upload_folder, filename)
        # Move the file form the temporal folder to the upload folder we setup
        file.save(path)
        sign_detector.detect(path, CONF_THRESHOLD)
        # Redirect the user to the resulting video route, which
        # will basicaly show on the browser the processed video
        toc = time.clock()
        print ('Processing took {:.3f}s'.format(toc-tic))
        return redirect(url_for('detection_api.uploaded_file',
                                filename=filename))
    return 0


# Route that will process the detect vehicle request
@detection_api.route('/faster_rcnn/vehicles', methods=['POST'])
def detect_vehicles():
    # # Get the name of the uploaded file
    # file = request.files['file']
    # CONF_THRESHOLD = float(request.form['conf_threshold'])
    # # Check if the file is one of the allowed types/extensions
    # if file and allowed_file(file.filename):
    #     tic = time.clock()
    #     # Make the filename safe, remove unsupported chars
    #     filename = secure_filename(file.filename)
    #     path = os.path.join(api_config.upload_folder, filename)
    #     # Move the file form the temporal folder to the upload folder we setup
    #     file.save(path)
    #     vehicle_detector.detect(path, CONF_THRESHOLD)
    #     # Redirect the user to the resulting video route, which
    #     # will basicaly show on the browser the processed video
    #     toc = time.clock()
    #     print ('Processing took {:.3f}s'.format(toc-tic))
    #     return redirect(url_for('detection_api.uploaded_file',
    #                             filename=filename))
    CONF_THRESHOLD = float(request.form['conf_threshold'])
    vehicle_detector.detect(os.path.join(api_config.upload_folder, 'samples', 'frames'), CONF_THRESHOLD)
    return jsonify(message='Detections done!')

# Route that will process the depth map request
@detection_api.route('/fcrn/depth', methods=['POST'])
def FCRNdepth_map():
    # Get the name of the uploaded file
    file = request.files['file']
    # CONF_THRESHOLD = float(request.form['conf_threshold'])
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        tic = time.clock()
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        path = os.path.join(api_config.upload_folder, filename)
        # Move the file form the temporal folder to the upload folder we setup
        file.save(path)
        FCRNDepth_estimator.detect(path)
        # Redirect the user to the resulting video route, which
        # will basicaly show on the browser the processed video
        toc = time.clock()
        print ('Processing took {:.3f}s'.format(toc - tic))
        return redirect(url_for('detection_api.uploaded_file',
                                    filename=filename))

# Route that will process monodepth map request
@detection_api.route('/monocular/depth', methods=['POST'])
def monoculardepth_map():
    # Get the name of the uploaded file
    file = request.files['file']
    # CONF_THRESHOLD = float(request.form['conf_threshold'])
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        tic = time.clock()
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        path = os.path.join(api_config.upload_folder, filename)
        # Move the file form the temporal folder to the upload folder we setup
        file.save(path)
        monoculardepth_estimator.detect(path)
        # Redirect the user to the resulting video route, which
        # will basicaly show on the browser the processed video
        toc = time.clock()
        print ('Processing took {:.3f}s'.format(toc - tic))
        return redirect(url_for('detection_api.uploaded_file',
                                filename=filename))



# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser
@detection_api.route('/uploads/<filename>')
def uploaded_file(filename):
    result_directory = os.path.join(api_config.upload_folder,
                                    filename.split(".")[0],
                                    "result")
    return send_from_directory(result_directory,
                               filename)
