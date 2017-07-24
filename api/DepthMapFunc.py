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



# Necessary files for monodepth
from monodepth_model import *
from average_gradients import *





class MonoDepthEstimation:
    def __init__(self, model_path):
        print('Loading Monodepth Estimation Model')
        encoder = 'vgg'
        input_height = 256
        input_width = 512
        use_deconv = False

        params = monodepth_parameters(
            encoder=encoder,
            height=input_height,
            width=input_width,
            use_deconv=use_deconv
        )

        self.left = tf.placeholder(tf.float32, [2, 256, 512, 3])
        self.right = tf.placeholder(tf.float32, [2, 256, 512, 3])
        # left = tf.zeros([2, 256, 512, 3], "float32")
        # right=left_p

        self.model = MonodepthModel(params, 'test', self.left, self.right)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess1 = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        self.sess1.run(tf.global_variables_initializer())
        self.sess1.run(tf.local_variables_initializer())

        # print args.model_name
        # model_path = 'model/model_kitti'
        # if args.checkpoint_path == '':
        #     restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
        # else:
        restore_path = model_path

        # print restore_path
        train_saver.restore(self.sess1, restore_path)
        print('Monodepth Estimation Model Loaded Successfully')

    def post_process_disparity(self, disp):
        _, h, w = disp.shape
        l_disp = disp[0, :, :]
        r_disp = np.fliplr(disp[1, :, :])
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
        r_mask = np.fliplr(l_mask)
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

    def process_start(self, image):
        im = tf.placeholder("float32", [720, 1280, 3])
        left_res = tf.image.resize_images(im, [256, 512])
        # left1=tf.expand_dims(left1, 0)
        left_in = tf.stack([left_res, tf.image.flip_left_right(left_res)])
        left_frame = self.sess1.run(left_in, feed_dict={im: image}) / 255
        disp = self.sess1.run(self.model.disp_left_est[0], feed_dict={self.left: left_frame})
        # disparities = disp[0].squeeze()
        disparities_pp = self.post_process_disparity(disp.squeeze())
        mind = np.min(disparities_pp)
        maxd = np.max(disparities_pp)
        d = 255 * (disparities_pp - mind) / (maxd - mind)
        d = np.dstack([d, d, d])

        resize_ph = tf.placeholder(tf.float32, shape=(256, 512, 3))
        place1 = tf.image.resize_images(resize_ph, [720, 1280])
        pre = tf.cast(place1, tf.uint8)
        ss = tf.concat([image, pre], 1)
        disparity = self.sess1.run(ss, feed_dict={resize_ph: d})

        return disparity

    def detect(self, path):
        vid_output = CFG.result_folder + 'out.mp4'
        # Location of the input video
        clip1 = VideoFileClip(path)
        # clip = clip1.resize((width, height))
        vid_clip = clip1.fl_image(self.process_start)
        vid_clip.write_videofile(vid_output, audio=False)

    # def predict(self, model_data_path, image):
    #     w, h, c = image.shape
    #     image_holder = tf.placeholder(tf.float32, shape=(w, h, c))
    #     pred_res = tf.cast(tf.image.resize_images(image, [self.depth_input_height, self.depth_input_width]), tf.uint8)
    #     image1 = self.sess.run(pred_res, feed_dict={image_holder: image})
    #
    #     img = image1
    #     img = np.array(img).astype('float32')
    #     img = np.expand_dims(np.asarray(img), axis=0)
    #
    #     resize_ph = tf.placeholder(tf.float32, shape=(None, 128, 160, 1))
    #     pred = self.sess.run(self.net.get_output(), feed_dict={self.input_node: img})
    #     pred_res = tf.image.resize_images(resize_ph, [w, h])
    #     pred1 = self.sess.run(pred_res, feed_dict={resize_ph: pred})
    #
    #     smallest = np.min(pred1[0, :, :, 0])
    #     largest = np.max(pred1[0, :, :, 0])
    #     pred_norm = 255 * (pred1[0, :, :, 0] - smallest) / (largest - smallest)
    #     pred_norm = np.dstack((pred_norm, pred_norm, pred_norm))
    #
    #     place1 = tf.placeholder(tf.float32, shape=(w, h, c))
    #     pre = tf.cast(place1, tf.uint8)
    #     ss = tf.concat([image, pre], 1)
    #
    #     dd = self.sess.run(ss, feed_dict={place1: pred_norm})
    #
    #     return dd
    #
    #
    # def process_start(self, image):
    #     pred = self.predict(self.model_data_path, image)
    #     return pred
    #
    #
    # def detect(self, path):
    #     vid_output = CFG.result_folder + 'out.mp4'
    #
    #     # Location of the input video
    #     clip1 = VideoFileClip(path)
    #     # clip = clip1.resize((width, height))
    #     vid_clip = clip1.fl_image(self.process_start)
    #     vid_clip.write_videofile(vid_output, audio=False)


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