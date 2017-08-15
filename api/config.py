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
__C.sign_framework = "TF"

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
__C.vehicle_framework = "TF"

# VECHILE CAFFE CONFIG
__C.vehicle_prototxt = os.path.join(__C.root_directory, "api/resources/"
                                    "faster-rcnn/models/pascal_voc/"
                                    "ZF/test.prototxt")

__C.vehicle_caffemodel = os.path.join(__C.root_directory, "api/"
                                      "resources/faster-rcnn/weights/"
                                      "pascal_voc/"
                                      "ZF_faster_rcnn_final.caffemodel")

#Depth configurations
__C.FCRN_depth_model_path = os.path.join(__C.root_directory, "core/"\
                    "sub-modules/FCRN_depth")
__C.FCRN_depth_weights_path = os.path.join(__C.root_directory, "api/"\
                    "resources/FCRN_depth/weights/NYU_ResNet-UpProj.npy")
__C.monocular_depth_model_path = os.path.join(__C.root_directory, "core/"\
                    "sub-modules/monocular_depth")
__C.monocular_depth_weights_path = os.path.join(__C.root_directory, "api/"\
                    "resources/monocular_depth/weights/model_cityscapes")

__C.vehicle_tfmodel = os.path.join(__C.root_directory, "api/resources/"
                                   "tffaster-rcnn/weights/pascal_voc/"
                                   "VGGnet_fast_rcnn_iter_150000.ckpt")

__C.vehicle_net = "VGGnet_test"

