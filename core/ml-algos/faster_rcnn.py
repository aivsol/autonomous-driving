#!/usr/bin/env python
import numpy as np
import caffe
import os
import glob as glob
import classes as CLS
from PIL import Image
import matplotlib.pyplot as plt

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from apputils.utilities import video_to_frames
from apputils.utilities import frames_to_video
from ml_algorithm import MLAlgorithm
from config import api_config

plt.switch_backend('agg')


class FasterRCNN(MLAlgorithm):

    def __init__(self, prototxt_path, caffemodel_path, classes, cpu_mode):
        MLAlgorithm.__init__(self)
        cfg.TEST.HAS_RPN = True
        self.prototxt = prototxt_path
        self.caffemodel = caffemodel_path
        self.cpu_mode = cpu_mode
        self.classes = classes
        if cpu_mode:
            caffe.set_mode_cpu()
        else:
            # TODO: Allow GPU ID to be set through API
            caffe.set_mode_gpu()
            caffe.set_device(0)
            cfg.GPU_ID = 0
        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)
        print '\n\nLoaded network {:s}'.format(self.caffemodel)

    def detect(self, path, conf_thresh=0.8):

        video_name = path.split("/")[-1]
        frames_folder = video_to_frames(video_name)

        im_names = glob.glob(os.path.join(frames_folder, '*.ppm'))
        im_names.sort()
        for im_name in im_names:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo {}'.format(im_name)
            self. process_frame(video_name, im_name, self.classes, conf_thresh)

        frames_to_video(video_name)

    def draw(self, im_file, class_name, dets, ax, thresh=0.5):

        if class_name in CLS.IGNORE:
            return

        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                         bbox[2] - bbox[0],
                         bbox[3] - bbox[1], fill=False,
                         edgecolor='red', linewidth=3.5))
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        plt.axis('off')
        plt.tight_layout()

    def process_frame(self, video_name, image_name, CLASSES, CONF_THRESH):
        # Output frame path
        im_name = image_name.split('/')[-1].replace('.ppm', '.jpg')
        im_path_ = os.path.join(api_config.upload_folder,
                                video_name.split(".")[0],
                                "annotated-frames", im_name)
        im = np.array(Image.open(image_name))
        im = im[:, :, ::-1]
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time,
                                               boxes.shape[0])

        NMS_THRESH = 0.3
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            self.draw(im_path_, cls, dets, ax, thresh=CONF_THRESH)
        plt.savefig(im_path_, bbox_inches='tight')
        plt.close()
