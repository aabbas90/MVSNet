#!/usr/bin/env python
"""
Copyright 2018, Yao Yao, HKUST.
Model architectures.
"""

import sys
import math
import tensorflow as tf
import numpy as np

sys.path.append("../")
from cnn_wrapper.mvsnet import *
from homography_warping import get_homographies, homography_warping
from homography_warping import repeat_int

FLAGS = tf.app.flags.FLAGS

def get_propability_map(cv, depth_map, depth_start, depth_interval):
    """ get probability map from cost volume """

    def _repeat_(x, num_repeats):
        """ repeat each element num_repeats times """
        x = tf.reshape(x, [-1])
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    shape = tf.shape(depth_map)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth = tf.shape(cv)[1]

    # byx coordinate, batched & flattened
    b_coordinates = tf.range(batch_size)
    y_coordinates = tf.range(height)
    x_coordinates = tf.range(width)
    b_coordinates, y_coordinates, x_coordinates = tf.meshgrid(b_coordinates, y_coordinates, x_coordinates)
    b_coordinates = _repeat_(b_coordinates, batch_size)
    y_coordinates = _repeat_(y_coordinates, batch_size)
    x_coordinates = _repeat_(x_coordinates, batch_size)

    # d coordinate (floored and ceiled), batched & flattened
    d_coordinates = tf.reshape((depth_map - depth_start) / depth_interval, [-1])
    d_coordinates_left0 = tf.clip_by_value(tf.cast(tf.floor(d_coordinates), 'int32'), 0, depth - 1)
    d_coordinates_left1 = tf.clip_by_value(d_coordinates_left0 - 1, 0, depth - 1)
    d_coordinates1_right0 = tf.clip_by_value(tf.cast(tf.ceil(d_coordinates), 'int32'), 0, depth - 1)
    d_coordinates1_right1 = tf.clip_by_value(d_coordinates1_right0 + 1, 0, depth - 1)

    # voxel coordinates
    voxel_coordinates_left0 = tf.stack(
        [b_coordinates, d_coordinates_left0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_left1 = tf.stack(
        [b_coordinates, d_coordinates_left1, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right0 = tf.stack(
        [b_coordinates, d_coordinates1_right0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right1 = tf.stack(
        [b_coordinates, d_coordinates1_right1, y_coordinates, x_coordinates], axis=1)

    # get probability image by gathering and interpolation
    prob_map_left0 = tf.gather_nd(cv, voxel_coordinates_left0)
    prob_map_left1 = tf.gather_nd(cv, voxel_coordinates_left1)
    prob_map_right0 = tf.gather_nd(cv, voxel_coordinates_right0)
    prob_map_right1 = tf.gather_nd(cv, voxel_coordinates_right1)
    prob_map = prob_map_left0 + prob_map_left1 + prob_map_right0 + prob_map_right1
    prob_map = tf.reshape(prob_map, [batch_size, height, width, 1])

    return prob_map


def inference(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction    
    if is_master_gpu:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=True)
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UniNetDS2({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []
        for d in range(depth_num):
            # compute cost (variation metric)
            ave_feature = ref_tower.get_output()
            ave_feature2 = tf.square(ref_tower.get_output())
            for view in range(0, FLAGS.view_num - 1):
                homography = tf.slice(view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            cost = ave_feature2 - tf.square(ave_feature)
            depth_costs.append(cost)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=False)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=True)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(
            tf.scalar_mul(-1, filtered_cost_volume), axis=1, name='prob_volume')
        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    return estimated_depth_map, prob_map#, filtered_depth_map, probability_volume

def inference_mem(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    feature_c = 32
    feature_h = FLAGS.max_h / 4
    feature_w = FLAGS.max_w / 4

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction    
    if is_master_gpu:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    else:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    ref_feature = ref_tower.get_output()
    ref_feature2 = tf.square(ref_feature)

    view_features = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UniNetDS2({'data': view_image}, is_training=True, reuse=tf.AUTO_REUSE)
        view_features.append(view_tower.get_output())
    view_features = tf.stack(view_features, axis=0)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)
    view_homographies = tf.stack(view_homographies, axis=0)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []

        for d in range(depth_num):
            # compute cost (standard deviation feature)
            ave_feature = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature2 = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave2', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature = tf.assign(ave_feature, ref_feature)
            ave_feature2 = tf.assign(ave_feature2, ref_feature2)

            def body(view, ave_feature, ave_feature2):
                """Loop body."""
                homography = tf.slice(view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                warped_view_feature = homography_warping(view_features[view], homography)
                ave_feature = tf.assign_add(ave_feature, warped_view_feature)
                ave_feature2 = tf.assign_add(ave_feature2, tf.square(warped_view_feature))
                view = tf.add(view, 1)
                return view, ave_feature, ave_feature2

            view = tf.constant(0)
            cond = lambda view, *_: tf.less(view, FLAGS.view_num - 1)
            _, ave_feature, ave_feature2 = tf.while_loop(
                cond, body, [view, ave_feature, ave_feature2], back_prop=False, parallel_iterations=1)

            ave_feature = tf.assign(ave_feature, tf.square(ave_feature) / (FLAGS.view_num * FLAGS.view_num))
            ave_feature2 = tf.assign(ave_feature2, ave_feature2 / FLAGS.view_num - ave_feature)
            depth_costs.append(ave_feature2)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    print(cost_volume)
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=tf.AUTO_REUSE)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=tf.AUTO_REUSE)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                           axis=1, name='prob_volume')

        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            ds = depth_start[i]
            de = depth_end[i]
            soft_1d = tf.linspace(ds, de, tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    # filtered_depth_map = tf.cast(tf.greater_equal(prob_map, 0.8), dtype='float32') * estimated_depth_map

    # return filtered_depth_map, prob_map
    return estimated_depth_map, prob_map, depth_end

def compute_ref_features(images, cams, is_master_gpu=True):
    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction    
    if is_master_gpu:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=True)
    else:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=True)
    ref_feature = ref_tower.get_output()
    ref_feature2 = tf.square(ref_feature)
    return ref_feature, ref_feature2

def compute_cost_volume(images, cams, current_depth, ref_feature, ref_feature2, is_master_gpu=True):
    feature_c = 32
    feature_h = FLAGS.max_h / 4
    feature_w = FLAGS.max_w / 4

    x = tf.linspace(0.5, tf.cast(FLAGS.max_w, 'float32') - 0.5, FLAGS.max_w)
    y = tf.linspace(0.5, tf.cast(FLAGS.max_h, 'float32') - 0.5, FLAGS.max_h)
    xv, yv = tf.meshgrid(x, y)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):

        # compute cost (standard deviation feature)
        ave_feature = tf.Variable(tf.zeros(
            [FLAGS.batch_size, feature_h, feature_w, feature_c]),
            name='ave', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        ave_feature2 = tf.Variable(tf.zeros(
            [FLAGS.batch_size, feature_h, feature_w, feature_c]),
            name='ave2', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        ave_feature = tf.assign(ave_feature, ref_feature)
        ave_feature2 = tf.assign(ave_feature2, ref_feature2)
        view_images_projected = GetProjectedImagesTF(images, cams, current_depth, xv, yv)

        def body(view, ave_feature, ave_feature2):
            """Loop body."""
            view_image = tf.gather(view_images_projected, view) # tf.slice(view_images_projected, [view, 0, 0, 0], [1, -1, -1, -1])
            view_tower = UniNetDS2({'data': view_image}, is_training=True, reuse=True)
            view_tower_out = view_tower.get_output()
            ave_feature = tf.assign_add(ave_feature, view_tower_out)
            ave_feature2 = tf.assign_add(ave_feature2, tf.square(view_tower_out))
            view = tf.add(view, 1)
            return view, ave_feature, ave_feature2

        view = tf.constant(0)
        cond = lambda view, *_: tf.less(view, FLAGS.view_num - 1)
        _, ave_feature, ave_feature2 = tf.while_loop(
            cond, body, [view, ave_feature, ave_feature2], back_prop=False, parallel_iterations=1)

        ave_feature = tf.assign(ave_feature, tf.square(ave_feature) / (FLAGS.view_num * FLAGS.view_num))
        ave_feature2 = tf.assign(ave_feature2, ave_feature2 / FLAGS.view_num - ave_feature)
        return ave_feature2, view_images_projected # depth_cost
    
    # cost_volume = tf.stack(depth_costs, axis=1)

def get_depth_from_cost_volume(cost_volume, depth_start, depth_interval, depth_end, is_master_gpu = True):
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=True)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=True)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                           axis=1, name='prob_volume')

        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(FLAGS.max_d, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        print(soft_4d)
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    return estimated_depth_map, prob_map, probability_volume

def inference_refine(images, cams, ref_depth, depth_start, depth_num, depth_interval, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    feature_c = 32
    feature_h = FLAGS.max_h / 4
    feature_w = FLAGS.max_w / 4

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction    
    if is_master_gpu:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    else:
        ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    ref_feature = ref_tower.get_output()
    ref_feature2 = tf.square(ref_feature)

    # view_features = []
    # for view in range(1, FLAGS.view_num):
    #     view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
    #     view_tower = UniNetDS2({'data': view_image}, is_training=True, reuse=True)
    #     view_features.append(view_tower.get_output())
    # view_features = tf.stack(view_features, axis=0)
    x = tf.constant(tf.linspace(0.5, tf.cast(FLAGS.max_w, 'float32') - 0.5, FLAGS.max_w))
    y = tf.constant(tf.linspace(0.5, tf.cast(FLAGS.max_h, 'float32') - 0.5, FLAGS.max_h))
    xv, yv = tf.meshgrid(x, y)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []

        for d in range(depth_num):
            # compute cost (standard deviation feature)
            current_depth = ref_depth + tf.cast(d, tf.float32) * depth_interval + tf.cast(depth_start, tf.float32) 
            ave_feature = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature2 = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave2', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature = tf.assign(ave_feature, ref_feature)
            ave_feature2 = tf.assign(ave_feature2, ref_feature2)
            view_images_projected = GetProjectedImagesTF(images, cams, current_depth, xv, yv)

            def body(view, ave_feature, ave_feature2):
                """Loop body."""
                view_image = tf.gather(view_images_projected, view) # tf.slice(view_images_projected, [view, 0, 0, 0], [1, -1, -1, -1])
                view_tower = UniNetDS2({'data': view_image}, is_training=True, reuse=tf.AUTO_REUSE)
                view_tower_out = view_tower.get_output()
                ave_feature = tf.assign_add(ave_feature, view_tower_out)
                ave_feature2 = tf.assign_add(ave_feature2, tf.square(view_tower_out))
                view = tf.add(view, 1)
                return view, ave_feature, ave_feature2

            view = tf.constant(0)
            cond = lambda view, *_: tf.less(view, FLAGS.view_num - 1)
            _, ave_feature, ave_feature2 = tf.while_loop(
                cond, body, [view, ave_feature, ave_feature2], back_prop=False, parallel_iterations=1)

            ave_feature = tf.assign(ave_feature, tf.square(ave_feature) / (FLAGS.view_num * FLAGS.view_num))
            ave_feature2 = tf.assign(ave_feature2, ave_feature2 / FLAGS.view_num - ave_feature)
            depth_costs.append(ave_feature2)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    print(cost_volume)
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=tf.AUTO_REUSE)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=tf.AUTO_REUSE)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)
    print(filtered_cost_volume)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                           axis=1, name='prob_volume')

        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        print(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        print(soft_4d)
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)
        print(soft_4d)

    # probability map
    # prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    # filtered_depth_map = tf.cast(tf.greater_equal(prob_map, 0.8), dtype='float32') * estimated_depth_map

    # return filtered_depth_map, prob_map
    return estimated_depth_map #, prob_map

def GetProjectedImagesTF(centered_images, real_cams, ref_depth, xv, yv):
    # ref_image = tf.get_variable("ref_image")
    # ref_img = tf.transpose(tf.squeeze(centered_images[0,0,:,:,:]), perm = [2, 0, 1])
    R_ref = tf.squeeze(real_cams[0, 0, 0, :3, :3])
    t_ref = real_cams[0, 0, 0, :3, 3:4]
    K_ref = tf.squeeze(real_cams[0, 0, 1, :3, :3])
    projected_images = [] #tf.get_variable("projected_images", shape = [centered_images.shape[0] - 1, centered_images.shape[1:]])
    for c in range(1, FLAGS.view_num):
        # n_img = tf.transpose(tf.squeeze(centered_images[0, c, :, :, :]), perm = [2, 0, 1])
        R_n = tf.squeeze(real_cams[0, c, 0, :3, :3])
        t_n = real_cams[0, c, 0, :3, 3:4]
        K_n = tf.squeeze(real_cams[0, c, 1, :3, :3])
        W = PixelCoordToWorldCoord(K_ref, R_ref, t_ref, xv, yv, ref_depth)
        xp, yp = WorldCoordTopixelCoord(K_n, R_n, t_n, W)
        projected_img = tf.expand_dims(tf.transpose(GetImageAtPixelCoordinates(tf.transpose(tf.squeeze(centered_images[0, c, :, :, :]), perm = [2, 0, 1]), xp, yp), perm = [1, 2, 0]), axis = 0)
        projected_img.set_shape([1, FLAGS.max_h, FLAGS.max_w, 3])
        projected_images.append(projected_img)
    return projected_images #tf.stack(projected_images, axis = 0)

def PixelCoordToWorldCoord(K, R, t, u, v, depth):
    a = tf.multiply(K[2,0], u) - K[0,0]
    b = tf.multiply(K[2,1], u) - K[0,1]
    c = tf.multiply(depth, (K[0,2] - np.multiply(K[2,2], u)))

    g = tf.multiply(K[2,0], v) - K[0,1]
    h = tf.multiply(K[2,1], v) - K[1,1]
    l = tf.multiply(depth, (K[1,2] - tf.multiply(K[2,2], v)))

    y = tf.divide(l - tf.divide(tf.multiply(g, c), a), h - tf.divide(tf.multiply(g, b), a))
    x = tf.divide((c - tf.multiply(b,y)), a)
    z = depth
    C = tf.stack((x, y, z), axis = 0)
    # W = np.matmul(R.T, C - t[:,:,np.newaxis])
    
    W1 = tf.reduce_sum(tf.expand_dims(R[:,0:1], axis = 2) * (C - tf.expand_dims(t, axis = 2)), axis = 0)
    W2 = tf.reduce_sum(tf.expand_dims(R[:,1:2], axis = 2) * (C - tf.expand_dims(t, axis = 2)), axis = 0)
    W3 = tf.reduce_sum(tf.expand_dims(R[:,2:3], axis = 2) * (C - tf.expand_dims(t, axis = 2)), axis = 0)
    # W2 = tf.sum(R[:,1:2,np.newaxis] * (C - t[:,:,np.newaxis]), axis = 0)
    # W3 = tf.sum(R[:,2:3,np.newaxis] * (C - t[:,:,np.newaxis]), axis = 0)
    W = tf.stack((W1, W2, W3), axis = 0)
    return W

def WorldCoordTopixelCoord(K, R, t, W):
    # C = t + np.matmul(R, W.T)
    R1 = tf.transpose(R[0:1, :])
    R2 = tf.transpose(R[1:2, :])
    R3 = tf.transpose(R[2:3, :])
    
    C1 = tf.reduce_sum(tf.expand_dims(R1, axis = 2) * W, axis = 0)
    C2 = tf.reduce_sum(tf.expand_dims(R2, axis = 2) * W, axis = 0)
    C3 = tf.reduce_sum(tf.expand_dims(R3, axis = 2) * W, axis = 0)
    C = tf.expand_dims(t, axis = 2) + tf.stack((C1, C2, C3), axis = 0)
    # p = np.matmul(K, C)
    K1 = tf.transpose(K[0:1, :])
    K2 = tf.transpose(K[1:2, :])
    K3 = tf.transpose(K[2:3, :])
    
    P1 = tf.reduce_sum(tf.expand_dims(K1, axis = 2) * C, axis = 0)
    P2 = tf.reduce_sum(tf.expand_dims(K2, axis = 2) * C, axis = 0)
    P3 = tf.reduce_sum(tf.expand_dims(K3, axis = 2) * C, axis = 0)
 
    Xp = tf.divide(P1, P3)
    Yp = tf.divide(P2, P3)
    return Xp, Yp


def interpolateS(image, x, y):
    image_shape = tf.shape(image)
    batch_size = image_shape[0]
    height =image_shape[1]
    width = image_shape[2]

    # image coordinate to pixel coordinate
    x = x - 0.5
    y = y - 0.5
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    max_y = tf.cast(height - 1, dtype='int32')
    max_x = tf.cast(width - 1,  dtype='int32')
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)
    b = repeat_int(tf.range(batch_size), height * width)

    indices_a = tf.stack([b, y0, x0], axis=1)
    indices_b = tf.stack([b, y0, x1], axis=1)
    indices_c = tf.stack([b, y1, x0], axis=1)
    indices_d = tf.stack([b, y1, x1], axis=1)

    pixel_values_a = tf.expand_dims(tf.gather_nd(image, indices_a), 1)
    pixel_values_b = tf.expand_dims(tf.gather_nd(image, indices_b), 1)
    pixel_values_c = tf.expand_dims(tf.gather_nd(image, indices_c), 1)
    pixel_values_d = tf.expand_dims(tf.gather_nd(image, indices_d), 1)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    area_a = tf.expand_dims(((y1 - y) * (x1 - x)), 1)
    area_b = tf.expand_dims(((y1 - y) * (x - x0)), 1)
    area_c = tf.expand_dims(((y - y0) * (x1 - x)), 1)
    area_d = tf.expand_dims(((y - y0) * (x - x0)), 1)
    output = tf.add_n([area_a * pixel_values_a,
                        area_b * pixel_values_b,
                        area_c * pixel_values_c,
                        area_d * pixel_values_d])
    return output

def GetImageAtPixelCoordinates(image, xp, yp):
    interpR = interpolateS(image[0:1, :, :], tf.reshape(xp, [-1]), tf.reshape(yp, [-1]))
    interpG = interpolateS(image[1:2, :, :], tf.reshape(xp, [-1]), tf.reshape(yp, [-1]))
    interpB = interpolateS(image[2:3, :, :], tf.reshape(xp, [-1]), tf.reshape(yp, [-1]))
    interpR = tf.reshape(interpR, shape=tf.shape(image[0:1, :, :]))
    interpG = tf.reshape(interpG, shape=tf.shape(image[0:1, :, :]))
    interpB = tf.reshape(interpB, shape=tf.shape(image[0:1, :, :]))
    return tf.concat((interpR, interpG, interpB), axis = 0)

def depth_refine(init_depth_map, image, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ refine depth image with the image """

    # normalization parameters
    depth_shape = tf.shape(init_depth_map)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    depth_start_mat = tf.tile(tf.reshape(
        depth_start, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_end_mat = tf.tile(tf.reshape(
        depth_end, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_scale_mat = depth_end_mat - depth_start_mat

    # normalize depth map (to 0~1)
    init_norm_depth_map = tf.div(init_depth_map - depth_start_mat, depth_scale_mat)

    # resize normalized image to the same size of depth image
    resized_image = tf.image.resize_bilinear(image, [depth_shape[1], depth_shape[2]])

    # refinement network
    if is_master_gpu:
        norm_depth_tower = RefineNet({'color_image': resized_image, 'depth_image': init_norm_depth_map},
                                        is_training=True, reuse=tf.AUTO_REUSE)
    else:
        norm_depth_tower = RefineNet({'color_image': resized_image, 'depth_image': init_norm_depth_map},
                                        is_training=True, reuse=tf.AUTO_REUSE)
    norm_depth_map = norm_depth_tower.get_output()

    # denormalize depth map
    refined_depth_map = tf.multiply(norm_depth_map, depth_scale_mat) + depth_start_mat

    return refined_depth_map

def non_zero_mean_absolute_diff(y_true, y_pred, interval):
    """ non zero mean absolute loss for one batch """
    with tf.name_scope('MAE'):
        shape = tf.shape(y_pred)
        interval = tf.reshape(interval, [shape[0]])
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
        masked_abs_error = tf.abs(mask_true * (y_true - y_pred))            # 4D
        masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2, 3])        # 1D
        masked_mae = tf.reduce_sum((masked_mae / interval) / denom)         # 1
    return masked_mae

def less_one_percentage(y_true, y_pred, interval):
    """ less one accuracy for one batch """
    with tf.name_scope('less_one_error'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true) + 1e-7
        interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_one_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 1.0), dtype='float32')
    return tf.reduce_sum(less_one_image) / denom

def less_three_percentage(y_true, y_pred, interval):
    """ less three accuracy for one batch """
    with tf.name_scope('less_three_error'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true) + 1e-7
        interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_three_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 3.0), dtype='float32')
    return tf.reduce_sum(less_three_image) / denom

def mvsnet_loss(estimated_disp_image, disp_image, depth_interval):
    """ compute loss and accuracy """
    # non zero mean absulote loss
    masked_mae = non_zero_mean_absolute_diff(disp_image, estimated_disp_image, depth_interval)
    # less one accuracy
    less_one_accuracy = less_one_percentage(disp_image, estimated_disp_image, depth_interval)
    # less three accuracy
    less_three_accuracy = less_three_percentage(disp_image, estimated_disp_image, depth_interval)

    return masked_mae, less_one_accuracy, less_three_accuracy