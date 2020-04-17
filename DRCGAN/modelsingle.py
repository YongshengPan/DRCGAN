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



def build_resnet_block(inputres, dim, change_dimension=False, last_relu = True, name="resnet"):
    with tf.variable_scope(name):
        if change_dimension:
            short_cut_conv = general_conv3d(inputres, dim, 1, 1, 1, 2, 2, 2, 0.02, "VALID", "sc", do_relu=False)
        else:
            short_cut_conv = inputres
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")
        if change_dimension:
            out_res = general_conv3d(out_res, dim, 3, 3, 3, 2, 2, 2, 0.02, "VALID", "c1")
        else:
            out_res = general_conv3d(out_res, dim, 3, 3, 3, 1, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv3d(out_res, dim, 3, 3, 3, 1, 1, 1, 0.02, "VALID", "c2", do_relu=False)
        if last_relu:
            return tf.nn.relu(out_res + short_cut_conv)
        else:
            return out_res + short_cut_conv


def build_resnet50_block(inputres, dim, change_dimension=False, block_stride=2, name="resnet"):
    with tf.variable_scope(name):
        if change_dimension:
            short_cut_conv = general_conv3d(inputres, dim*4, 1, 1, 1, block_stride, block_stride, block_stride, 0.02, "VALID", "sc", do_relu=False)
        else:
            short_cut_conv = inputres
        if change_dimension:
            out_res = general_conv3d(inputres, dim, 1, 1, 1, block_stride, block_stride, block_stride, 0.02, "VALID","c1")
        else:
            out_res = general_conv3d(inputres, dim, 1, 1, 1, 1, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv3d(out_res, dim, 3, 3, 3, 1, 1, 1, 0.02, "VALID", "c2")
        out_res = general_conv3d(out_res, dim*4, 1, 1, 1, 1, 1, 1, 0.02, "VALID", "c3", do_relu=False)
        return tf.nn.relu(out_res + short_cut_conv)



def build_feature_basic_2d(in_x, dim, name="discriminator"):
    with tf.variable_scope(name):
        ks = 3
        o_c0 = general_conv3d(in_x, dim, 3, ks, ks, 1, 1, 1, 0.2, "VALID", name="c0", do_norm=True)
        o_c1 = general_conv3d(o_c0, dim, 1, ks, ks, 1, 1, 1, 0.2, "VALID", name="c1", do_norm=True)
        o_p0 = tf.nn.max_pool3d(o_c1, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], padding='VALID') #13
        o_c2 = general_conv3d(o_p0, dim*2, 1, 3, 3, 1, 1, 1, 0.2, "VALID", name="c2", do_norm=True)
        o_c3 = general_conv3d(o_c2, dim*2, 1, 2, 2, 1, 1, 1, 0.2, "VALID", name="c3", do_norm=True)
        o_p1 = tf.nn.max_pool3d(o_c3, [1, 1, 2, 2, 1], [1, 1, 2, 2, 1], padding='VALID') #5
        o_c4 = general_conv3d(o_p1, dim*4, 1, 3, 3, 1, 1, 1, 0.2, "VALID", name="c4", do_norm=True)
        o_c5 = general_conv3d(o_c4, dim*4, 1, 2, 2, 1, 1, 1, 0.2, "VALID", name="c5", do_norm=True)
        o_p2 = tf.nn.avg_pool3d(o_c5, [1, 1, 2, 2, 1], [1, 1, 1, 1, 1], padding='VALID')
        print(o_p2)
        return o_p2

def build_feature_vgg_2d(in_x, dim, name="discriminator"):
    with tf.variable_scope(name):
        ks = 3
        o_c0 = general_conv3d(in_x, dim, 3, ks, ks, 1, 1, 1, 0.2, "SAME", name="c0", do_norm=True)
        o_p0 = tf.nn.max_pool3d(o_c0, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], padding='SAME')
        o_c1 = general_conv3d(o_p0, dim * 2, 1, ks, ks, 1, 1, 1, 0.2, "SAME", name="c1", do_norm=True)
        o_p1 = tf.nn.max_pool3d(o_c1, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], padding='SAME')
        o_c2 = general_conv3d(o_p1, dim * 4, 1, ks, ks, 1, 1, 1, 0.2, "SAME", name="c2", do_norm=True)
        o_p2 = tf.nn.max_pool3d(o_c2, [1, 1, 2, 2, 1], [1, 1, 2, 2, 1], padding='SAME')
        o_c3 = general_conv3d(o_p2, dim * 4, 1, ks, ks, 1, 1, 1, 0.2, "SAME", name="c3", do_norm=True)
        o_p3 = tf.nn.max_pool3d(o_c3, [1, 1, 2, 2, 1], [1, 1, 2, 2, 1], padding='SAME')
        o_c4 = general_conv3d(o_p3, dim * 4, 1, 2, 2, 1, 1, 1, 0.2, "VALID", name="c4", do_norm=True)
        print(o_c4)
        return o_c4

def build_feature_vgg_3d(in_x, dim, name="discriminator"):
    with tf.variable_scope(name):
        ks = 3
        o_c0 = general_conv3d(in_x, dim, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c0", do_norm=True)
        o_p0 = tf.nn.max_pool3d(o_c0, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c1 = general_conv3d(o_p0, dim*2, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c1", do_norm=True)
        o_p1 = tf.nn.max_pool3d(o_c1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c2 = general_conv3d(o_p1, dim*4, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c2", do_norm=True)
        o_p2 = tf.nn.max_pool3d(o_c2, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c3 = general_conv3d(o_p2, dim*4, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c3", do_norm=True)
        o_p3 = tf.nn.max_pool3d(o_c3, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c4 = general_conv3d(o_p3, dim*4, 3, 3, 3, 1, 1, 1, 0.2, "SAME", name="c4", do_norm=True, do_relu=True)
        return o_c4


def build_feature_basic(in_x, dim, name="discriminator"):
    with tf.variable_scope(name):
        ks = 3
        o_c0 = general_conv3d(in_x, dim, ks, ks, ks, 1, 1, 1, 0.2, "VALID", name="c0", do_norm=True)
        o_c1 = general_conv3d(o_c0, dim, ks, ks, ks, 1, 1, 1, 0.2, "VALID", name="c1", do_norm=True)
        o_p0 = tf.nn.max_pool3d(o_c1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='VALID')#13
        o_c2 = general_conv3d(o_p0, dim*2, 3, 3, 3, 1, 1, 1, 0.2, "VALID", name="c2", do_norm=True)
        o_c3 = general_conv3d(o_c2, dim*2, 2, 2, 2, 1, 1, 1, 0.2, "VALID", name="c3", do_norm=True)
        o_p1 = tf.nn.max_pool3d(o_c3, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='VALID')#5
        o_c4 = general_conv3d(o_p1, dim*4, 2, 2, 2, 1, 1, 1, 0.2, "VALID", name="c4", do_norm=True)
        o_c5 = general_conv3d(o_c4, dim*4, 2, 2, 2, 1, 1, 1, 0.2, "VALID", name="c5", do_norm=True)
        o_p2 = tf.nn.avg_pool3d(o_c5, [1, 2, 2, 2, 1], [1, 1, 1, 1, 1], padding='VALID')  # 5
        return o_p2



def build_feature_resnet18(inputgen, dim, name="generator"):
    with tf.variable_scope(name):
        ks = 3
        oc_1 = general_conv3d(inputgen, dim * 1, ks, ks, ks, 1, 1, 1, 0.02, "SAME", name="oc1")
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


def build_feature_resnet50(inputgen, dim, name="generator"):
    with tf.variable_scope(name):
        ks = 3
        oc_1 = general_conv3d(inputgen, dim * 1, ks, ks, ks, 1, 1, 1, 0.02, "SAME", name="oc1")
        op_1 = tf.nn.max_pool3d(oc_1, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding="SAME", name="op1")
        r1_1 = build_resnet50_block(op_1, dim * 1, True, 1, name="r11")
        r1_2 = build_resnet50_block(r1_1, dim * 1, name="r12")
        r1_3 = build_resnet50_block(r1_2, dim * 1, name="r13")

        r2_1 = build_resnet50_block(r1_3, dim * 2, True, name="r21")
        r2_2 = build_resnet50_block(r2_1, dim * 2, name="r22")
        r2_3 = build_resnet50_block(r2_2, dim * 2, name="r23")
        r2_4 = build_resnet50_block(r2_3, dim * 2, name="r24")

        r3_1 = build_resnet50_block(r2_4, dim * 4, True, name="r31")
        r3_2 = build_resnet50_block(r3_1, dim * 4, name="r32")
        r3_3 = build_resnet50_block(r3_2, dim * 4, name="r33")
        r3_4 = build_resnet50_block(r3_3, dim * 4, name="r34")
        r3_5 = build_resnet50_block(r3_4, dim * 4, name="r35")
        r3_6 = build_resnet50_block(r3_5, dim * 4, name="r36")

        r4_1 = build_resnet50_block(r3_6, dim * 8, True, name="r41")
        r4_2 = build_resnet50_block(r4_1, dim * 8, name="r42")
        r4_3 = build_resnet50_block(r4_2, dim * 8, name="r43")
        return r4_3


def build_multinstance(input, dim, numldmk, backbone='fv_vgg3d', means=None, vars=None, name="discriminator"):
    do_scale = False
    do_norm = True
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        dbank = []
        fbank =[]
        for idx in range(numldmk):
            if 'res50' in backbone:
                descs = build_feature_resnet50(input[idx], dim, name='descs{0}'.format(1))
            elif 'res18' in backbone:
                descs = build_feature_resnet18(input[idx], dim, name='descs{0}'.format(1))
            elif 'vgg3d' in backbone:
                descs = build_feature_vgg_3d(input[idx], dim, name='descs{0}'.format(1))
            elif 'basic' in backbone:
                descs = build_feature_basic(input[idx], dim, name='descs{0}'.format(1))
            elif 'view' in backbone:
                descs = build_feature_basic_2d(input[idx], dim, name='descs{0}'.format(1))
            else:
                descs = build_feature_vgg_2d(input[idx], dim, name='descs{0}'.format(1))

            if 'fv_res50' in backbone:
                fv_c = fv_block_layers_50(descs, do_norm=True, name='fv{0}'.format(idx))
            elif 'fv' in backbone:
                with tf.variable_scope('fv_layer', reuse=tf.AUTO_REUSE) as scope:
                    if means is None:
                        # x_m = descs
                        # x_s = 0.717 * (tf.square(x_m) - 1)
                        # fv_c = tf.reduce_mean(tf.concat((x_m, x_s), axis=-1), axis=(1, 2, 3))
                        # if do_norm:
                        #     fv_c = tf.nn.l2_normalize(fv_c, axis=-1)
                        fv_c = fv_block_layers(descs, do_norm=do_norm, name='fl{0}'.format(idx))
                    else:
                        x_m = tf.div(descs - means[idx], tf.sqrt(vars[idx] + 1e-5))
                        x_s = 0.717 * (tf.square(x_m) - 1)
                        fv_c = tf.reduce_mean(tf.concat((x_m, x_s), axis=-1), axis=(1, 2, 3))
                        if do_norm:
                            fv_c = tf.nn.l2_normalize(fv_c, axis=-1)
                    if do_scale:
                        scale = tf.get_variable('scale{0}'.format(idx), [1], initializer=tf.ones_initializer())
                        fv_c = scale*fv_c
            else:
                fv_c = tf.reduce_mean(descs, axis=(1, 2, 3))

            dbank.append(descs)
            fbank.append(fv_c)

        feats = tf.concat(fbank, axis=-1, name='feats')
        print(fbank)

        logits, prob = fc_op(feats, "fc_layer", 2, activation=tf.nn.softmax)
        return logits, prob, dbank

def build_speed_multinstance(input, numldmk, backbone = 'fv_vgg', means=None, vars=None, name="discriminator"):
    do_scale = False
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        print(name)
        fbank = []
        for idx in range(numldmk):
            if 'fv_res50'in backbone:
                fv_c = fv_block_layers_50(input[idx], do_norm=True, name='fl{0}'.format(idx))
            elif 'fv'in backbone:
                with tf.variable_scope('fv_layer', reuse=tf.AUTO_REUSE) as scope:
                    if means is None:
                        fv_c = fv_block_layers(input[idx], do_norm=False, name='fl{0}'.format(idx))
                    else:
                        x_m = tf.div(input[idx] - means[idx], tf.sqrt(vars[idx] + 1e-5))
                        x_s = 0.717 * (tf.square(x_m) - 1)
                        fv_c = tf.reduce_mean(tf.concat((x_m, x_s), axis=-1), axis=(1, 2, 3))
                    if do_scale:
                        scale = tf.get_variable('scale{0}'.format(idx), [1], initializer=tf.ones_initializer())
                        fv_c = scale*fv_c
            elif 'fc'in backbone:
                fv_c = tf.reduce_mean(input[idx], axis=(1, 2, 3))
            else:
                fv_c = tf.reduce_mean(input[idx], axis=(1, 2, 3))
            fbank.append(fv_c)
        feats = tf.concat(fbank, axis=-1, name='feats')
        logits, prob = fc_op(feats, "fc_layer", 2, activation=tf.nn.softmax)
        return logits, prob

def build_discriminator(fbank, numldmk, name="discriminator"):
    with tf.variable_scope(name):
        dbank = []
        for idx in range(numldmk):
            dbank.append(fbank[idx])
        feats = tf.concat(dbank, axis=-1, name='res')
        # feats = tf.concat(fbank, axis=-1, name='feats')

        print(feats)
        logits, prob = fc_op(feats, "fc_layer", 2, activation=tf.nn.softmax)
        return logits, prob
