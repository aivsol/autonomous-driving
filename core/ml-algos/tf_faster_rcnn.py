#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import glob as glob
import classes as CLS
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.cElement as ET

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

from apputils.utilities import video_to_frames
from apputils.utilities import frames_to_video
from apputils.utilities import xml_setup
from apputils.utilities import xml_add_object
from apputils.utilities import xml_write

from ml_algorithm import MLAlgorithm
from config import api_config


class TFFasterRCNN(MLAlgorithm):

    def __init__(self, model_name, demo_net, demo_model, classes):
        MLAlgorithm.__init__(self)
        cfg.TEST.HAS_RPN = True
        g = tf.Graph()
        with g.as_default():
            # init session
            self.sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
            self.net = get_network(demo_net, len(classes))
            saver = tf.train.Saver()
            saver.restore(self.sess, demo_model)
        self.classes = classes
        print 'Loaded network {:s}'.format(demo_model)

    def detect(self, path, conf_thresh=0.8):

        video_name = os.path.basename(path)
        frames_folder = video_to_frames(video_name)

        im_names = glob.glob(os.path.join(frames_folder, '*.jpg'))
        im_names.sort()
        for im_name in im_names:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo TF {}'.format(im_name)
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

            frame_name = os.path.basename(im_file)
            if frame_name in self.xml.keys():
                self.xml[frame_name] += [class_name, bbox[0], bbox[1],
                                         bbox[2], bbox[3]]
            else:
                self.xml[frame_name] = [class_name, bbox[0], bbox[1],
                                        bbox[2], bbox[3]]
            xml_add_object(self.annotation, frame_name.split(".")[0],
                           class_name, self.classes.index(class_name),
                           bbox)

        plt.axis('off')
        plt.tight_layout()

    def process_frame(self, video_name, im_name, CLASSES, CONF_THRESH):
        # Output frame path
        im_path_ = os.path.join(api_config.upload_folder,
                                video_name.split(".")[0],
                                "annotated-frames", os.path.basename(im_name))
        im = np.array(Image.open(im_name))
        im = im[:, :, ::-1]
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.sess, self.net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time,
                                               boxes.shape[0])

        NMS_THRESH = 0.3
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        self.annotation = xml_setup(im_name, im.shape)
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            self.draw(im_path_, cls, dets, ax, thresh=CONF_THRESH)
        xml_write(video_name, os.path.basename(im_name), self.annotation)
        plt.savefig(im_path_, bbox_inches='tight')
        plt.close()
