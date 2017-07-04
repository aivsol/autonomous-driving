from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.upload_folder = 'uploads/'
__C.result_folder = 'uploads/result/'
__C.allowed_extensions = set(['avi'])

