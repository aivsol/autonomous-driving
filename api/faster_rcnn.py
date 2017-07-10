#!/usr/bin/env python
#TODO: Remove unnecessary imports
import _init
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from utilities import remove_previous_frames, video_to_frames, \
                        remove_previous_results, frames_to_video
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import glob as glob
import os
import sys
import classes
from PIL import Image

class FasterRCNN:

    def __init__ (self, input_path, prototxt_path, caffemodel_path, classes, cpu_mode, conf_thresh=0.8):

        cfg.TEST.HAS_RPN = True
        self.prototxt = prototxt_path
        self.caffemodel = caffemodel_path
        self.cpu_mode = cpu_mode
        self.conf_thresh = conf_thresh
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
    
    def detect (self):
        
        remove_previous_frames()
	video_to_frames()
	
	im_names = glob.glob(os.path.join('uploads','frames','*.ppm'))
	for im_name in im_names:
	    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	    print 'Demo {}'.format(im_name)
	    self. process_frame(im_name, self.classes, self.conf_thresh)
	
	remove_previous_results()
	frames_to_video()

    def draw_detections(self, im_file, class_name, dets, ax, thresh=0.5): 

	if class_name in classes.IGNORE:
	    return

	inds = np.where(dets[:, -1] >= thresh)[0]
	if len(inds) == 0:
	    return

	for i in inds:
	    bbox = dets[i, :4]
	    score = dets[i, -1]

	    ax.add_patch(
		plt.Rectangle((bbox[0], bbox[1]),
			      bbox[2] - bbox[0],
			      bbox[3] - bbox[1], fill=False,
			      edgecolor='red', linewidth=3.5)
		)
	    ax.text(bbox[0], bbox[1] - 2,
		    '{:s} {:.3f}'.format(class_name, score),
		   bbox=dict(facecolor='blue', alpha=0.5),
		   fontsize=14, color='white')

	plt.axis('off')
	plt.tight_layout()
	plt.draw()

    def process_frame(self, image_name, CLASSES, CONF_THRESH):
	
	#im = cv2.imread(image_name)
	im = np.array(Image.open(image_name))
	im = im[:,:,::-1]
	timer = Timer()
	timer.tic()
	scores, boxes = im_detect(self.net, im)
	timer.toc()
	print ('Detection took {:.3f}s for '
	       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

	NMS_THRESH = 0.3
	im_path = os.path.join('uploads', 'annotated-frames')
	im = im[:, :, (2, 1, 0)]
	fig, ax = plt.subplots(figsize=(12, 12)) 
	ax.imshow(im, aspect='equal')
	for cls_ind, cls in enumerate(CLASSES[1:]):
	    cls_ind += 1 # because we skipped background
	    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
	    cls_scores = scores[:, cls_ind]
	    dets = np.hstack((cls_boxes,
			      cls_scores[:, np.newaxis])).astype(np.float32)
	    keep = nms(dets, NMS_THRESH)
	    dets = dets[keep, :]
	    im_name = image_name.split('/')[-1].replace('.ppm','.jpg')
	    im_path_ = os.path.join('uploads', 'annotated-frames', im_name)
	    self.draw_detections(im_path_, cls, dets, ax, thresh=CONF_THRESH)
	plt.savefig(im_path_, bbox_inches='tight')


input_path = "uploads/test.avi"

prototxt_path = "/home/zaheer/aivsol/stage/autonomous-driving/core/py-faster-rcnn/models/gtsdb/ZF/faster_rcnn_end2end/test.prototxt"
caffemodel_path = "/home/zaheer/aivsol/stage/autonomous-driving/core/py-faster-rcnn/data/GTSDB/TrainedModel/zf_faster_rcnn_final.caffemodel"

sign_detector = FasterRCNN(input_path, prototxt_path, caffemodel_path, classes.SIGNS_CLASSES, True, 0.6)

raw_input()

prototxt_path = "/home/zaheer/aivsol/stage/autonomous-driving/core/py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt"
caffemodel_path = "/home/zaheer/aivsol/stage/autonomous-driving/core/py-faster-rcnn/data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel"

vehicle_detector = FasterRCNN(input_path, prototxt_path, caffemodel_path, classes.VOC_CLASSES, True, 0.6)

raw_input()

vehicle_detector.detect()

raw_input()

sign_detector.detect()

raw_input()
