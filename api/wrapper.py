#!/usr/bin/env python

import _init
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import glob as glob
import os
import sys
import classes

VOC_NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

SIGNS_NETS = {'vgg16': ('VGG16',
                    'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                    'zf_faster_rcnn_final.caffemodel')}

def draw_detections(im_file, class_name, dets, ax, thresh=0.5): 

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

def process_frame(net, image_name, CLASSES):
    
    im = cv2.imread(image_name)
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    CONF_THRESH = 0.8
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
        draw_detections(im_path_, cls, dets, ax, thresh=CONF_THRESH)
    plt.savefig(im_path_, bbox_inches='tight')

# TODO: Make the next four functions flexible
def remove_previous_frames():
    os.system('rm uploads/frames/*')
    
def video_to_frames():
    os.system('ffmpeg -i uploads/test.avi uploads/frames/%5d.ppm')

def remove_previous_results():
    os.system('rm uploads/result/*')

def frames_to_video():
    os.system('ffmpeg -start_number 1 -framerate 2 -i ' + \
                'uploads/annotated-frames/%5d.jpg -vcodec mpeg4 ' + \
                '-pix_fmt yuvj422p uploads/result/test.avi')
    
def detect_signs(path, cpu_mode=True):
    cfg.TEST.HAS_RPN = True
    
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'gtsdb', SIGNS_NETS['zf'][0],
                    'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'GTSDB','TrainedModel',
                              SIGNS_NETS['zf'][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    
    remove_previous_frames()
    video_to_frames()
    
    im_names = glob.glob(os.path.join('uploads','frames','*.ppm'))
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo {}'.format(im_name)
        process_frame(net, im_name, classes.SIGNS_CLASSES)
    
    remove_previous_results()
    frames_to_video()

def detect_vehicles(path, cpu_mode=True):
    
    cfg.TEST.HAS_RPN = True
    prototxt = os.path.join(cfg.MODELS_DIR, VOC_NETS['zf'][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                             VOC_NETS['zf'][1])
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    remove_previous_frames()
    video_to_frames()
    
    im_names = glob.glob(os.path.join('uploads','frames','*.ppm'))
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo {}'.format(im_name)
        process_frame(net, im_name, classes.VOC_CLASSES)
    
    remove_previous_results()
    frames_to_video()
