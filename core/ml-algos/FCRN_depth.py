import _init
import sys
import argparse
from moviepy.editor import VideoFileClip
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from ml_algorithm import MLAlgorithm
from config import api_config
import numpy as np
import glob as glob
import FCRNdepth
# from utils.timer import Timer
from apputils.utilities import video_to_frames
from apputils.utilities import frames_to_video
plt.switch_backend('agg')


class FCRNDepth(MLAlgorithm):
    def __init__(self, model_path, weights_path):
        MLAlgorithm.__init__(self)
        self.weights_path = weights_path
        #Parameters for the input image to the network
        self.depth_input_height = 228
        self.depth_input_width = 304
        self.depth_input_channels = 3
        self.depth_input_batch_size = 1

        #Create input node placeholder
        self.input_node = tf.placeholder(tf.float32, shape=(None, self.depth_input_height, self.depth_input_width, self.depth_input_channels))

        # Construct the network
        self.net = FCRNdepth.ResNet50UpProj({'data': self.input_node}, self.depth_input_batch_size)
        self.sess = tf.Session()
        # Load the converted parameters
        print('Loading the model')
        self.net.load(self.weights_path, self.sess)

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        self.sess.run(init_new_vars_op)
        print('Depth Estimation Model loaded successfully')


    def predict(self, image):
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
        pred = self.predict(image)
        return pred


    def detect(self, path):
        # # Moviepy based script - Start
        # vid_output = api_config.result_folder + 'out.mp4'
        # # Location of the input video
        # clip1 = VideoFileClip(path)
        # vid_clip = clip1.fl_image(self.process_start)
        # vid_clip.write_videofile(vid_output, audio=False)
        # # Moviepy based script - End

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
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(im, aspect='equal')
            plt.imshow(disparity)
            plt.axis('off')
            plt.savefig(im_path_, bbox_inches='tight')
            plt.close()
            # timer.toc()
            # print ('Detection took {:.3f}s')
        frames_to_video(video_name)
        # ffmpeg based script - End