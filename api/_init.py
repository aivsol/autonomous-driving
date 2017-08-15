"""Set up paths for Faster R-CNN."""
import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# Add api directory to PYTHONPATH
this_dir = osp.dirname(__file__)
add_path(this_dir)

# Add caffe py-faster-rcnn to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'core', 'sub-modules',
                      'py-faster-rcnn', 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'core', 'sub-modules',
                    'py-faster-rcnn', 'lib')
add_path(lib_path)


# ADD TFFRCNN to Python Path
tffrcnn_path = osp.join(this_dir, '..', 'core', 'sub-modules',
                        'aivsol-TFFRCNN')
add_path(tffrcnn_path)

# Add ml-algos to PYTHONPATH
ml_algos_path = osp.join(this_dir, '..', 'core', 'ml-algos')
add_path(ml_algos_path)

submodules_path = osp.join(this_dir, '..', 'core', 'sub-modules')
add_path(submodules_path)


# Add repository's root to PYTHONPATH
root_path = osp.join(this_dir, '..')
add_path(root_path)

