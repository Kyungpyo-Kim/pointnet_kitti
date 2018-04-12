import tensorflow as tf
import math
import time
import numpy as np
import os
import sys

from utils.tf_util import *
from utils.gen_kitti_h5_util import *

point_dim = INPUT_DIM
num_point = POINT_NUM

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, point_dim))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    batch_size = point_cloud.get_shape()[0].value
    #num_point = point_cloud.get_shape()[1].value
    #point_dim = point_cloud.get_shape()[2].value

    input_image = tf.expand_dims(point_cloud, -1)

    # CONV
    net = conv2d(input_image, 64, [1,point_dim], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    net = conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    points_feat1 = conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    print(points_feat1)


    # MAX
    pc_feat1 = max_pool2d(points_feat1, [num_point,1], padding='VALID', scope='maxpool1')

    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    pc_feat1 = fully_connected(pc_feat1, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)

    # CONCAT
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])

    # CONV
    net = conv2d(points_feat1_concat, 512, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv6')
    net = conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv7')
    net = dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = conv2d(net, OUT_DIM, [1,1], padding='VALID', stride=[1,1],
                         activation_fn=None, scope='conv8')
    net = tf.squeeze(net, [2])

    return net

def get_loss(pred, label):
    """ pred: B, N, Number of class
        label: B, N """

    # Weighted loss to solve unbalancing dataset
    # class number of all dataset
    #   [39301545, 22122293, 465511, 1516731]

    # class weights
    # log_test3: [3, 6, 240, 80]
    # log_test4: [4, 6, 240, 80]
    # log_test5: [4, 7, 240, 80]
    # log_test6: [5, 8, 240, 80]
    # log_test6: [6, 9, 240, 80]
    class_weights = tf.constant([6, 10, 240, 80], dtype = tf.float32)

    # one hot vector
    depth = 4
    label_onehot = tf.one_hot(label, depth, dtype = tf.float32)

    # weighted
    weighted_labels = label_onehot * class_weights
    weights = tf.reduce_sum(weighted_labels, axis = 2)

    # weighted loss
    unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = label_onehot)
    loss = unweighted_loss * weights
    #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)

    return tf.reduce_mean(loss)

if __name__ == "__main__":
    with tf.Graph().as_default():
        a = tf.placeholder(tf.float32, shape=(32,num_point,point_dim))
        print("------------------------")
        #exit()
        net = get_model(a, tf.constant(True))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            for i in range(100):
                print(i)
                sess.run(net, feed_dict={a:np.random.rand(32,num_point,point_dim)})
            print(time.time() - start)
