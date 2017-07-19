"""Set up paths for Faster R-CNN."""
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe py-faster-rcnn to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'core', 'sub-modules', 'py-faster-rcnn', 
                                'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'core', 'sub-modules', 'py-faster-rcnn', 'lib')
add_path(lib_path)
