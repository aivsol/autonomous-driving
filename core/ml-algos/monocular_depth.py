#!/usr/bin/env python
import _init

import argparse
from moviepy.editor import VideoFileClip
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
# from utils.timer import Timer
# import models
from ml_algorithm import MLAlgorithm
from config import api_config
import numpy as np
import glob as glob
# from utils.timer import Timer
from apputils.utilities import video_to_frames
from apputils.utilities import frames_to_video
plt.switch_backend('agg')

# Necessary files for monodepth
import monoculardepth


class MonocularDepth(MLAlgorithm):
    def __init__(self, model_path, weight_path):
        MLAlgorithm.__init__(self)
        print('Loading Monodepth Estimation Model')
        encoder = 'vgg'
        input_height = 256
        input_width = 512
        use_deconv = False

        params = monoculardepth.monodepth_parameters(
            encoder=encoder,
            height=input_height,
            width=input_width,
            use_deconv=use_deconv
        )
        g_monocular = tf.Graph()
        self.graph = g_monocular
        with self.graph.as_default():
            self.left = tf.placeholder(tf.float32, [2, 256, 512, 3])
            self.right = tf.placeholder(tf.float32, [2, 256, 512, 3])

            self.model = monoculardepth.MonodepthModel(params, 'test', self.left, self.right)

            # SESSION
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess1 = tf.Session(config=config, graph=self.graph)

            # SAVER
            train_saver = tf.train.Saver()

            # INIT
            self.sess1.run(tf.global_variables_initializer())
            self.sess1.run(tf.local_variables_initializer())

            # print restore_path
            train_saver.restore(self.sess1, weight_path)
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
        with self.graph.as_default():
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
        #Using Moviepy script - Start
        # vid_output = api_config.result_folder + 'out.mp4'
        # # Location of the input video
        # clip1 = VideoFileClip(path)
        # vid_clip = clip1.fl_image(self.process_start)
        # vid_clip.write_videofile(vid_output, audio=False)
        # Using Moviepy script - End

        #ffmpeg based script - Start
        video_name = os.path.basename(path)
        frames_folder = video_to_frames(video_name)

        im_names = glob.glob(os.path.join(frames_folder, '*.ppm'))
        im_names.sort()
        for img_name in im_names:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo {}'.format(img_name)
            # timer = Timer()
            # timer.tic()
            # Output frame path
            im_name = img_name.split('/')[-1].replace('.ppm', '.jpg')
            im_path_ = os.path.join(api_config.upload_folder,
                                    video_name.split(".")[0],
                                    "annotated-frames", im_name)
            im = np.array(Image.open(img_name))
            im = im[:, :, ::-1]
            disparity = self.process_start(im)
            fig, ax = plt.subplots(figsize=(18, 18))
            ax.imshow(im, aspect='equal')
            plt.imshow(disparity)
            plt.tight_layout()
            plt.axis('off')
            plt.savefig(im_path_, bbox_inches='tight')
            plt.close()
            # timer.toc()
            # print ('Detection took {:.3f}s')
        frames_to_video(video_name)
        # ffmpeg based script - End

