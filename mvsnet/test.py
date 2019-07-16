#!/usr/bin/env python
"""
Copyright 2018, Yao Yao, HKUST.
Training script.
"""

from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import sys
import math
import argparse
import numpy as np
import tensorflow.contrib.eager as tfe

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import tensorflow as tf

sys.path.append("../")
from tools.common import Notify
from preprocess import *
from model import *
from mvsnet import *
from homography_warping import repeat_int
from graphicsHelper import *
import pdb

# params for datasets
tf.app.flags.DEFINE_string('dense_folder', '../TEST_DATA_FOLDER/', 
                           """Root path to dense folder.""")
# params for input
tf.app.flags.DEFINE_integer('view_num', 2,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('default_depth_start', 1,
                            """Start depth when training.""")
tf.app.flags.DEFINE_integer('default_depth_interval', 1, 
                            """Depth interval when training.""")
tf.app.flags.DEFINE_integer('max_d', 32, 
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 1024, 
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 768, 
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25, 
                            """Downsample scale for building cost volume (W and H).""")
tf.app.flags.DEFINE_float('interval_scale', 1, 
                            """Downsample scale for building cost volume (D).""")
tf.app.flags.DEFINE_integer('base_image_size', 32, 
                            """Base image size to fit the network.""")
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            """training batch size""")

# params for config
tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', 
                           '../MODEL_FOLDER/model.ckpt',
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 70000,
                            """ckpt step.""")
FLAGS = tf.app.flags.FLAGS

class MVSGenerator:
    """ data generator class, tf only accept generator without param """
    def __init__(self, sample_list, view_num):
        self.sample_list = sample_list
        self.view_num = view_num
        self.sample_num = len(sample_list)
        self.counter = 0
    
    def __iter__(self):
        while True:
            for data in self.sample_list: 
                scaled_images, centered_images, scaled_cams, real_cams, image_index = self.readData(data)
                yield (scaled_images, centered_images, scaled_cams, real_cams, image_index)

    def readData(self, data):
        # read input data
        images = []
        cams = []
        image_index = int(os.path.splitext(os.path.basename(data[0]))[0])
        selected_view_num = int(len(data) / 2)

        for view in range(min(self.view_num, selected_view_num)):
            # image = cv2.imread(data[2 * view])
            image_file = file_io.FileIO(data[2 * view], mode='rb')
            image = scipy.misc.imread(image_file, mode='RGB')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cam = load_cam(open(data[2 * view + 1]))
            cam_file = file_io.FileIO(data[2 * view + 1], mode='rb')
            cam = load_cam(cam_file)
            cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale
            images.append(image)
            cams.append(cam)

        if selected_view_num < self.view_num:
            for view in range(selected_view_num, self.view_num):
                # image = cv2.imread(data[0])
                image_file = file_io.FileIO(data[0], mode='rb')
                image = scipy.misc.imread(image_file, mode='RGB')
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # cam = load_cam(open(data[1]))
                cam_file = file_io.FileIO(data[1], mode='rb')
                cam = load_cam(cam_file)
                cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale
                images.append(image)
                cams.append(cam)

        # determine a proper scale to resize input 
        h_scale = float(FLAGS.max_h) / images[0].shape[0]
        w_scale = float(FLAGS.max_w) / images[0].shape[1]
        if h_scale > 1 or w_scale > 1:
            print ("max_h, max_w should < W and H!")
            exit()
        resize_scale = h_scale
        if w_scale > h_scale:
            resize_scale = w_scale
        scaled_input_images, scaled_input_cams = scale_mvs_input(images, cams, scale=resize_scale)

        # crop to fit network
        croped_images, croped_cams = crop_mvs_input(scaled_input_images, scaled_input_cams)

        # center images
        centered_images = []
        for view in range(self.view_num):
            centered_images.append(center_image(croped_images[view]))

        # sample cameras for building cost volume
        real_cams = np.copy(croped_cams) 
        scaled_cams = scale_mvs_camera(croped_cams, scale=FLAGS.sample_scale)
        
        # return mvs input
        scaled_images = []
        for view in range(self.view_num):
            scaled_images.append(scale_image(croped_images[view], scale=FLAGS.sample_scale))
        scaled_images = np.stack(scaled_images, axis=0)
        croped_images = np.stack(croped_images, axis=0)
        scaled_cams = np.stack(scaled_cams, axis=0)
        self.counter += 1
        return scaled_images, centered_images, scaled_cams, real_cams, image_index

def GetProjectedImages(centered_images, real_cams, ref_depth):
    ref_img = np.squeeze(centered_images[0,0,:,:,:]).swapaxes(0, 2).swapaxes(1, 2)
    x = np.linspace(0, ref_img.shape[2], ref_img.shape[2], axis = -1)   
    y = np.linspace(0, ref_img.shape[1], ref_img.shape[1], axis = -1)
    xv, yv = np.meshgrid(x, y)
    R_ref = np.squeeze(real_cams[0, 0, 0, :3, :3])
    t_ref = real_cams[0, 0, 0, :3, 3:4]
    K_ref = np.squeeze(real_cams[0, 0, 1, :3, :3])
    projected_images = np.zeros_like(centered_images)
    for c in range(1, centered_images.shape[1]):
        n_img = np.squeeze(centered_images[0,c,:,:,:]).swapaxes(0, 2).swapaxes(1, 2)
        R_n = np.squeeze(real_cams[0, 1, 0, :3, :3])
        t_n = real_cams[0, 1, 0, :3, 3:4]
        K_n = np.squeeze(real_cams[0, 1, 1, :3, :3])
        W = PixelCoordToWorldCoord(K_ref, R_ref, t_ref, xv, yv, ref_depth)
        xp, yp = WorldCoordTopixelCoord(K_n, R_n, t_n, W)
        projected_img = GetImageAtPixelCoordinates(n_img, xp, yp).swapaxes(0, 2).swapaxes(0, 1)
        projected_images[0,c,:,:,:] = projected_img
    return projected_images

def mvsnet_pipeline(mvs_list):
    print ('sample number: ', len(mvs_list))
    output_folder = os.path.join(FLAGS.dense_folder, 'depths_mvsnet')

    # create output folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    mvs_generator = MVSGenerator(mvs_list, FLAGS.view_num)
    for data in mvs_list:
        _, centered_images, scaled_cams, real_cams, image_index = mvs_generator.readData(data)

        centered_images = np.expand_dims(np.stack(centered_images, axis = 0), axis = 0).astype(np.float32)
        scaled_cams = np.expand_dims(np.stack(scaled_cams, axis = 0), axis = 0).astype(np.float32)
        real_cams = np.expand_dims(np.stack(real_cams, axis = 0), axis = 0).astype(np.float32)
        depth_start = np.expand_dims(np.squeeze(scaled_cams[:FLAGS.batch_size, 0:1, 1:2, 3:4, 0:1]).astype(np.float32), axis = 0)
        depth_interval = np.expand_dims(np.squeeze(scaled_cams[:FLAGS.batch_size, 0:1, 1:2, 3:4, 1:2]).astype(np.float32), axis = 0)
        cameras = []
        for c in range(FLAGS.view_num):
            R = np.squeeze(real_cams[0, c, 0, :3, :3])
            t = real_cams[0, c, 0, :3, 3:4]
            K = np.squeeze(real_cams[0, c, 1, :3, :3])
            currentCam = Camera(str(c), K, R, t, np.squeeze(centered_images[0, c, :, :, :]))
            cameras.append(currentCam)

        # depth map inference
        centered_images_tf = tf.convert_to_tensor(centered_images) 
        scaled_cams_tf = tf.convert_to_tensor(scaled_cams)
        depth_start_tf = tf.convert_to_tensor(depth_start)
        depth_interval_tf = tf.convert_to_tensor(depth_interval)
        init_depth_map, prob_map, depth_end = inference_mem(
            centered_images_tf, scaled_cams_tf, FLAGS.max_d, depth_start_tf, depth_interval_tf)

        # refinement 
        ref_image = tf.squeeze(tf.slice(tf.convert_to_tensor(centered_images), [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
        depth_map = depth_refine(init_depth_map, ref_image, FLAGS.max_d, depth_start_tf, depth_interval_tf)
        real_cams_tf = tf.convert_to_tensor(real_cams)

        # init option
        init_op = tf.global_variables_initializer()
        var_init_op = tf.local_variables_initializer()
        # GPU grows incrementally
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:   
            # initialization
            sess.run(var_init_op)
            sess.run(init_op)

            # load model
            if FLAGS.pretrained_model_ckpt_path is not None:
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(
                    sess, '-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
                print(Notify.INFO, 'Pre-trained model restored from %s' %
                    ('-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
                total_step = FLAGS.ckpt_step
        
            # run inference for each reference view
            start_time = time.time()
            try:
                out_depth_map, out_init_depth_map, out_prob_map, out_images, out_cams, out_depth_end = sess.run(
                    [depth_map, init_depth_map, prob_map, centered_images_tf, scaled_cams_tf, depth_end])
            except tf.errors.OutOfRangeError:
                print("all dense finished")  # ==> "End of dataset"
                break
            duration = time.time() - start_time
            print(Notify.INFO, 'depth inference %s finished. (%.3f sec/step)' % (data, duration), 
                    Notify.ENDC)
            
            ref_img = np.squeeze(out_images[0,0,:,:,:]).swapaxes(0, 2).swapaxes(1, 2)
            ref_depth = cv2.resize(np.squeeze(out_depth_map), (ref_img.shape[2],ref_img.shape[1]))
            ultra_refine_depth_map = inference_refine(centered_images_tf, real_cams_tf, tf.convert_to_tensor(ref_depth), depth_start_tf, depth_end_tf, FLAGS.max_d, depth_interval_tf)
            out_ultra_refine_depth_map = sess.run([ultra_refine_depth_map])
            out_ultra_refine_depth_map = cv2.resize(np.squeeze(out_ultra_refine_depth_map[0][0]), (ref_img.shape[2],ref_img.shape[1]))

            pdb.set_trace()
            # projected_images = GetProjectedImages(centered_images, real_cams, ref_depth)
            # projected_images = GetProjectedImagesTF(tf.convert_to_tensor(centered_images), tf.convert_to_tensor(real_cams), tf.convert_to_tensor(ref_depth))

            fig, ax0 = plt.subplots(nrows=1, ncols=3)
            ax0[0].imshow(np.swapaxes(ref_img, 0, 2).swapaxes(0, 1))
            ax0[1].imshow(ref_depth)
            ax0[2].imshow(out_ultra_refine_depth_map + ref_depth)
            multi = MultiCursor(fig.canvas, (ax0[0], ax0[1], ax0[2]), color='r', lw=1, horizOn=True, vertOn=True)
            plt.show()
            plt.tight_layout()

def main(_):  # pylint: disable=unused-argument
    """ program entrance """
    # generate input path list
    mvs_list = gen_pipeline_mvs_list(FLAGS.dense_folder)
    # mvsnet inference
    mvsnet_pipeline(mvs_list)


if __name__ == '__main__':
    tf.app.run()