from easydict import EasyDict as edict
import os
import os.path

__C = edict()

api_config = __C

__C.cpu_mode = True

__C.root_directory = os.getcwd()

__C.template_directory = os.path.join(os.getcwd(), "api/templates")

__C.upload_folder = 'api/uploads/'

__C.result_folder = 'api/uploads/result/'

__C.frames_folder = 'api/uploads/frames'

__C.annotated_frames_folder = 'api/uploads/annotated-frames'

__C.allowed_extensions = set(['avi', 'mp4'])

# sign_framework could be "TF" OR "CAFFE"
__C.sign_framework = "CAFFE"

# SIGN CAFFE CONFIG
__C.sign_prototxt = os.path.join(__C.root_directory, "api/resources/"
                                 "faster-rcnn/models/gtsdb"
                                 "/ZF/test.prototxt")

__C.sign_caffemodel = os.path.join(__C.root_directory, "api/resources/"
                                   "faster-rcnn/weights/GTSDB/"
                                   "zf_faster_rcnn_final.caffemodel")

# SIGN TF CONFIG
__C.sign_tfmodel = os.path.join(__C.root_directory, "api/resources/"
                                "tffaster-rcnn/weights/GTSDB/"
                                "VGGnet_fast_rcnn_iter_15000.ckpt")

__C.sign_net = "VGGnet_test"

# vehicle_framework could be "TF" OR "CAFFE"
__C.vehicle_framework = "CAFFE"

# VECHILE CAFFE CONFIG
__C.vehicle_prototxt = os.path.join(__C.root_directory, "api/resources/"
                                    "faster-rcnn/models/pascal_voc/"
                                    "ZF/faster_rcnn_test.pt")

__C.vehicle_caffemodel = os.path.join(__C.root_directory, "api/"
                                      "resources/faster-rcnn/weights/"
                                      "pascal_voc/"
                                      "ZF_faster_rcnn_final.caffemodel")

# VEHICLE TF CONFIG
__C.vehicle_tfmodel = os.path.join(__C.root_directory, "api/resources/"
                                   "tffaster-rcnn/weights/pascal_voc/"
                                   "VGGnet_fast_rcnn_iter_150000.ckpt")

__C.vehicle_net = "VGGnet_test"
