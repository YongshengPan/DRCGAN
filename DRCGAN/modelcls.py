# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random

from layers import *

img_layer = 1


def build_resnet_block(inputres, dim, change_dimension=False, block_stride=2, name="resnet"):
    with tf.variable_scope(name):
        if change_dimension:
            short_cut_conv = general_conv3d(inputres, dim, 1, 1, 1, block_stride, block_stride, block_stride, 0.02, "VALID", "sc", do_relu=False)
        else:
            short_cut_conv = inputres
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")
        if change_dimension:
            out_res = general_conv3d(out_res, dim, 3, 3, 3, block_stride, block_stride, block_stride, 0.02, "VALID", "c1")
        else:
            out_res = general_conv3d(out_res, dim, 3, 3, 3, 1, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv3d(out_res, dim, 3, 3, 3, 1, 1, 1, 0.02, "VALID", "c2", do_relu=False)
        return tf.nn.relu(out_res + short_cut_conv)


def build_generator(inputgen, dim, numofres = 6, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv3d(pad_input, dim, f, f, f, 1, 1, 1, 0.02, name="c1")
        o_c2 = general_conv3d(o_c1, dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv3d(o_c2, dim * 4, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c3")
        o_rb = o_c3
        for idd in range(numofres):
            o_rb = build_resnet_block(o_rb, dim * 4, name='r{0}'.format(idd))
        o_c4 = general_deconv3d(o_rb, [1, 64, 64, 64, dim * 2], dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv3d(o_c4, [1, 128, 128, 128, dim], dim, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c5")
        o_c5_pad = tf.pad(o_c5, [[0, 0], [ks, ks], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c6 = general_conv3d(o_c5_pad, img_layer, f, f, f, 1, 1, 1, 0.02, "VALID", "c6", do_relu=False)
        out_gen = tf.nn.tanh(o_c6, "t1")
        return out_gen


def build_gen_discriminator(inputdisc, dim, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        o_c1 = general_conv3d(inputdisc, dim, f, f, f, 2, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = general_conv3d(o_c1, dim * 2, f, f, f, 2, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv3d(o_c2, dim * 4, f, f, f, 2, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv3d(o_c3, dim * 8, f, f, f, 1, 1, 1, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv3d(o_c4, 1, f, f, f, 1, 1, 1, 0.02, "SAME", "c5", do_norm=False, do_relu=False)
        return o_c5


def build_feature_vgg_3d(in_x, dim, name="vgg_3d"):
    with tf.variable_scope(name):
        ks = 3
        o_c0 = general_conv3d(in_x, dim, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c0", do_norm=True)
        o_p0 = tf.nn.max_pool3d(o_c0, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c1 = general_conv3d(o_p0, dim*2, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c1", do_norm=True)
        o_p1 = tf.nn.max_pool3d(o_c1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c2 = general_conv3d(o_p1, dim*4, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c2", do_norm=True)
        o_p2 = tf.nn.max_pool3d(o_c2, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c3 = general_conv3d(o_p2, dim*4, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c3", do_norm=True)
        o_p3 = tf.nn.max_pool3d(o_c3, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c4 = general_conv3d(o_p3, dim*4, 3, 3, 3, 1, 1, 1, 0.2, "SAME", name="c4", do_norm=True, do_relu=True)
        o_p4 = tf.nn.avg_pool3d(o_c4, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='VALID')
        return o_p4


def build_feature_resnet18(inputgen, dim, name="res18"):
    with tf.variable_scope(name):
        ks = 3
        oc_1 = general_conv3d(inputgen, dim * 1, ks, ks, ks, 2, 2, 2, 0.02, "SAME", name="oc1")
        op_1 = tf.nn.max_pool3d(oc_1, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding="SAME", name="op1")
        r1_1 = build_resnet_block(op_1, dim * 1, name="r11")
        r1_2 = build_resnet_block(r1_1, dim * 1, name="r12")
        r2_1 = build_resnet_block(r1_2, dim * 2, True, name="r21")
        r2_2 = build_resnet_block(r2_1, dim * 2, name="r22")
        r3_1 = build_resnet_block(r2_2, dim * 4, True, name="r31")
        r3_2 = build_resnet_block(r3_1, dim * 4, name="r32")
        r4_1 = build_resnet_block(r3_2, dim * 8, True, name="r41")
        r4_2 = build_resnet_block(r4_1, dim * 8, name="r42")
        return r4_2


def build_classifier_old(input, dim, order=(1, 2), name="discriminator"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        ks = 3
        o_c0 = general_conv3d(input, dim, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c0", do_norm=True)
        o_p0 = tf.nn.max_pool3d(o_c0, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c1 = general_conv3d(o_p0, dim*2, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c1", do_norm=True)
        o_p1 = tf.nn.max_pool3d(o_c1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c2 = general_conv3d(o_p1, dim*4, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c2", do_norm=True)
        o_p2 = tf.nn.max_pool3d(o_c2, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c3 = general_conv3d(o_p2, dim*4, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c3", do_norm=True)
        o_p3 = tf.nn.max_pool3d(o_c3, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c4 = general_conv3d(o_p3, dim*4, 3, 3, 3, 1, 1, 1, 0.2, "SAME", name="c4", do_norm=True, do_relu=True)
        o_p4 = tf.nn.avg_pool3d(o_c4, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='VALID')

        # means = tf.get_variable("means", shape=o_p4.get_shape()[1::], dtype=tf.float32, initializer=tf.zeros_initializer())
        descs = []
        if 1 in order:
            descs.append(o_p4)
        if 2 in order:
            descs.append(tf.square(o_p4) - 1)
        if 3 in order:
            descs.append(tf.square(o_p4)*o_p4)
        descs = tf.nn.l2_normalize(tf.concat(descs, axis=-1), axis=-1)
        # descs = tf.nn.l2_normalize(tf.concat((o_p4, tf.square(o_p4) - 1), axis=-1), axis=-1)

        # weights = tf.Variable(tf.ones([1, 4, 5, 4, 1], dtype=np.float32), name="attention", trainable=True)

        # feats = tf.reduce_mean(descs, axis=(1, 2, 3))
        feats = tf.nn.l2_normalize(tf.reshape(descs, (-1, descs.shape[1]*descs.shape[2]*descs.shape[3]*descs.shape[4])), axis=-1)
        logits, prob = fc_op(feats, "fc_layer", 2, activation=tf.nn.softmax)
        return logits, prob, [o_c0, o_c1, o_c2, o_c3, o_c4]


def build_classifier(input, dim, order=(1, 2), name="discriminator"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        ks = 3
        o_c0 = general_conv3d(input, dim, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c0", do_norm=True)
        o_p0 = tf.nn.max_pool3d(o_c0, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c1 = general_conv3d(o_p0, dim*2, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c1", do_norm=True)
        o_p1 = tf.nn.max_pool3d(o_c1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c2 = general_conv3d(o_p1, dim*4, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c2", do_norm=True)
        o_p2 = tf.nn.max_pool3d(o_c2, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c3 = general_conv3d(o_p2, dim*4, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c3", do_norm=True)
        o_p3 = tf.nn.max_pool3d(o_c3, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c4 = general_conv3d(o_p3, dim*4, 3, 3, 3, 1, 1, 1, 0.2, "SAME", name="c4", do_norm=True, do_relu=True)
        o_p4 = tf.nn.avg_pool3d(o_c4, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='VALID')

        if 0 in order:
            feats = tf.reduce_mean(o_p4, axis=(1, 2, 3))
        else:
            descs = []
            if 1 in order:
                descs.append(o_p4)
            if 2 in order:
                descs.append(tf.square(o_p4) - 1)
            if 3 in order:
                descs.append(tf.square(o_p4) * o_p4)
            descs = tf.nn.l2_normalize(tf.concat(descs, axis=-1), axis=-1)
            feats = tf.nn.l2_normalize(
                tf.reshape(descs, (-1, descs.shape[1] * descs.shape[2] * descs.shape[3] * descs.shape[4])), axis=-1)
        logits, prob = fc_op(feats, "fc_layer", 2, activation=tf.nn.softmax)
        return logits, prob, [o_c0, o_c1, o_c2, o_c3, o_c4]

