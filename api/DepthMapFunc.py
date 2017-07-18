#!/usr/bin/env python
import _init

import argparse
from moviepy.editor import VideoFileClip
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
# from utils.timer import Timer
import models
from config import cfg as CFG
# from utils.timer import Timer
import numpy as np
import glob as glob
import classes as CLS


class DepthEstimation:
    def __init__(self, model_path):

        self.model_data_path=model_path
        #Parameters for the input image to the network
        self.depth_input_height = 228
        self.depth_input_width = 304
        self.depth_input_channels = 3
        self.depth_input_batch_size = 1

        #Create input node placeholder
        self.input_node = tf.placeholder(tf.float32, shape=(None, self.depth_input_height, self.depth_input_width, self.depth_input_channels))

        # Construct the network
        self.net = models.ResNet50UpProj({'data': self.input_node}, self.depth_input_batch_size)
        self.sess = tf.Session()
        # Load the converted parameters
        print('Loading the model')
        self.net.load(self.model_data_path, self.sess)

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        self.sess.run(init_new_vars_op)
        print('Depth Estimation Model loaded successfully')


    def predict(self, model_data_path, image):
        w, h, c = image.shape
        image_holder = tf.placeholder(tf.float32, shape=(w, h, c))
        pred_res = tf.cast(tf.image.resize_images(image, [self.depth_input_height, self.depth_input_width]), tf.uint8)
        image1 = self.sess.run(pred_res, feed_dict={image_holder: image})

        img = image1
        img = np.array(img).astype('float32')
        img = np.expand_dims(np.asarray(img), axis=0)

        resize_ph = tf.placeholder(tf.float32, shape=(None, 128, 160, 1))
        pred = self.sess.run(self.net.get_output(), feed_dict={self.input_node: img})
        pred_res = tf.image.resize_images(resize_ph, [w, h])
        pred1 = self.sess.run(pred_res, feed_dict={resize_ph: pred})

        smallest = np.min(pred1[0, :, :, 0])
        largest = np.max(pred1[0, :, :, 0])
        pred_norm = 255 * (pred1[0, :, :, 0] - smallest) / (largest - smallest)
        pred_norm = np.dstack((pred_norm, pred_norm, pred_norm))

        place1 = tf.placeholder(tf.float32, shape=(w, h, c))
        pre = tf.cast(place1, tf.uint8)
        ss = tf.concat([image, pre], 1)

        dd = self.sess.run(ss, feed_dict={place1: pred_norm})

        return dd


    def process_start(self, image):
        pred = self.predict(self.model_data_path, image)
        return pred


    def detect(self, path):
        vid_output = CFG.result_folder + 'out.mp4'

        # Location of the input video
        clip1 = VideoFileClip(path)
        # clip = clip1.resize((width, height))
        vid_clip = clip1.fl_image(self.process_start)
        vid_clip.write_videofile(vid_output, audio=False)