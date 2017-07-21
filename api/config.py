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

__C.allowed_extensions = set(['avi'])

__C.sign_prototxt = os.path.join(__C.root_directory, "api/resources/"
                                 "faster-rcnn/models/gtsdb"
                                 "/ZF/test.prototxt")

__C.sign_caffemodel = os.path.join(__C.root_directory, "api/resources/"
                                   "faster-rcnn/weights/GTSDB/"
                                   "zf_faster_rcnn_final.caffemodel")

__C.vehicle_prototxt = os.path.join(__C.root_directory, "api/resources/"
                                    "faster-rcnn/models/pascal_voc/"
                                    "ZF/test.prototxt")

__C.vehicle_caffemodel = os.path.join(__C.root_directory, "api/"
                                      "resources/faster-rcnn/weights/"
                                      "pascal_voc/"
                                      "ZF_faster_rcnn_final.caffemodel")
