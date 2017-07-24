from easydict import EasyDict as edict
import os
import os.path

__C = edict()

cfg = __C

__C.cpu_mode = True

__C.root_directory = os.getcwd()

__C.template_directory = os.path.join(os.getcwd(), "api/templates")

__C.upload_folder = 'api/uploads/'

__C.input_path = 'api/uploads/test.avi'

__C.result_folder = 'api/uploads/result/'

__C.frames_folder = 'api/uploads/frames'

__C.annotated_frames_folder = 'api/uploads/annotated-frames'

__C.allowed_extensions = set(['avi', 'mp4'])

__C.sign_prototxt = os.path.join(__C.root_directory, "core/"\
                    "py-faster-rcnn/models/gtsdb/ZF/faster_rcnn_end2end/"\
                    "test.prototxt")

__C.sign_caffemodel = os.path.join(__C.root_directory, "core/"\
                    "py-faster-rcnn/data/GTSDB/TrainedModel/"\
                    "zf_faster_rcnn_final.caffemodel")

__C.vehicle_prototxt = os.path.join(__C.root_directory, "core/"\
                    "py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/"\
                    "faster_rcnn_test.pt")

__C.vehicle_caffemodel = os.path.join(__C.root_directory, "core/"\
                    "py-faster-rcnn/data/faster_rcnn_models/"\
                    "ZF_faster_rcnn_final.caffemodel")
#Depth configurations
__C.depth_model_path = os.path.join(__C.root_directory, "core/"\
                    "tensorflow/models/NYU_ResNet-UpProj.npy")
__C.monodepth_model_path = os.path.join(__C.root_directory, "core/"\
                    "tensorflow/models/monodepth/model_kitti")

